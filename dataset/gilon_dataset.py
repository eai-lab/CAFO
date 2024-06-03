import os
import pandas as pd
import numpy as np
import joblib
import pickle
from typing import Optional
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from utils import bcolors
from dataset.constant_sampler import ConstantRandomSampler
from dataset.make_images import generate_rp, generate_gramian, generate_markov


class CAFODataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train = GilonDataset("train", self.cfg)
            self.val = GilonDataset("val", self.cfg)
            self.train_sampler = ConstantRandomSampler(self.train)
            # self.val_sampler = ConstantRandomSampler(self.val)
        elif stage == "test":
            self.test = GilonDataset("test", self.cfg)
            # self.test_sampler = ConstantRandomSampler(self.train)
        elif stage == "predict":
            self.test = GilonDataset("test", self.cfg)
            self.test_sampler = ConstantRandomSampler(self.test)
        else:
            raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            sampler=self.train_sampler,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            sampler=self.test_sampler,
            shuffle=False,
            drop_last=False,
        )


class GilonDataset(Dataset):
    """Load Gilon sensor data and meta data using global_id value. index is mapped to global id through label_df"""

    def __init__(self, mode, cfg):
        print(bcolors.OKBLUE + bcolors.BOLD + f"{mode} Mode" + bcolors.ENDC + bcolors.ENDC)
        self.mode = mode
        self.cfg = cfg
        self.cv_val_num = cfg.task.validation_cv_num
        self.exp_subset = cfg.task["exp_subset"]

        if self.cfg.task.task_name == "gilon_activity":
            self.activity_map = activity_map

        # Create train dataset. Val is just a subset of train dataset
        if mode in ("train", "val"):
            self.label_df = pd.read_csv(f"data/{cfg.task.features_save_dir}/train_label_chunk40_window160_72users.csv")
            sensor_feature_save_path = f"data/{cfg.task.features_save_dir}/weight_train_sensors.pkl"
            meta_feature_save_path = f"data/{cfg.task.features_save_dir}/weight_train_meta.pkl"
            if os.path.isfile(sensor_feature_save_path):
                print(f"{sensor_feature_save_path} Exists!, loading..")
                with open(sensor_feature_save_path, "rb") as f:
                    self.sensor_dict = pickle.load(f)
                with open(meta_feature_save_path, "rb") as m:
                    self.meta_dict = pickle.load(m)

            cv_dict_path = f"data/{cfg.task.features_save_dir}/cv_dict.pkl"
            with open(cv_dict_path, "rb") as c:
                self.cv_dict = pickle.load(c)

            if mode in {"train"}:
                self.label_df = self.label_df[~self.label_df["ID"].isin(self.cv_dict[self.cv_val_num])]
            elif mode in {"val"}:
                self.label_df = self.label_df[self.label_df["ID"].isin(self.cv_dict[self.cv_val_num])]

        elif mode in ("test"):
            self.label_df = pd.read_csv(f"data/{cfg.task.features_save_dir}/test_label_chunk40_window160_72users.csv")
            sensor_test_feature_save_path = f"data/{cfg.task.features_save_dir}/weight_test_sensors.pkl"
            meta_test_feature_save_path = f"data/{cfg.task.features_save_dir}/weight_test_meta.pkl"
            if os.path.isfile(sensor_test_feature_save_path):
                print(f"{sensor_test_feature_save_path} Exists!, loading..")
                with open(sensor_test_feature_save_path, "rb") as f:
                    self.sensor_dict = pickle.load(f)
                with open(meta_test_feature_save_path, "rb") as m:
                    self.meta_dict = pickle.load(m)

        self.label_df = self.label_df[self.label_df["EXP"].isin(self.exp_subset)]
        """If you want to perform any ablation on the datasets, please do it here. all the features will be based on the label_df"""

        self.label_df = self.label_df[self.label_df["Weight"] < 95]
        get_label_statistics(self.label_df)
        self.label_df = self.label_df.reset_index(drop=True)

        if mode == "test":
            self.label_df = self.label_df[["ID", "GLOBAL_ID", "EXP", "Weight", "SPEED", "ACTIVITY"]]
            self.label_df.to_csv(f"{self.cfg.save_output_path}/test_label.csv", index=False)

        if self.cfg.add_random_channel_idx:
            self.meaningless_feat_dict = self._load_meaning_less_sensor_features(mode, cfg.random_channel_type)

    def _load_meaning_less_sensor_features(self, mode, sensortype):
        if mode in ["train", "val"]:
            with open(f"data/meaningless_feature/train_meaningless_sensor_{sensortype}_dict.pkl", "rb") as f:
                meaningless_feat = pickle.load(f)
        elif mode in ["test"]:
            with open(f"data/meaningless_feature/test_meaningless_sensor_{sensortype}_dict.pkl", "rb") as f:
                meaningless_feat = pickle.load(f)
        else:
            raise ValueError(f"Unknown mode {mode}")
        return meaningless_feat

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        # TODO: Check if the weighted sampling is working properly
        global_id = self.label_df.iloc[index].GLOBAL_ID
        label = self.label_df[self.label_df["GLOBAL_ID"] == global_id]

        original_feature = self.sensor_dict[global_id]
        if self.cfg.task.task_name == "gilon_activity":
            activity_label = self.activity_map[label.EXP.item()]
            y_true = torch.tensor(activity_label, dtype=torch.long)
        elif self.cfg.task.task_name == "gilon_speed":
            speed_label = label.SPEED.item()
            y_true = torch.tensor(speed_label, dtype=torch.float32)
            # y_true = torch.tensor(self.speed_label_dict[speed_label], dtype=torch.long)
        elif self.cfg.task.task_name == "gilon_bodyweight":
            weight_label = label.Weight.item()
            y_true = torch.tensor(weight_label, dtype=torch.float32)

        if self.cfg.image_encoding == "rp":
            if self.cfg.add_random_channel_idx:
                feature = generate_rp(original_feature, self.cfg, global_id, self.meaningless_feat_dict[global_id])
            else:
                feature = generate_rp(original_feature, self.cfg, global_id)
        elif self.cfg.image_encoding == "gramian":
            feature = generate_gramian(original_feature, self.cfg, global_id, self.meaningless_feat_dict[global_id])
        elif self.cfg.image_encoding == "markov":
            feature = generate_markov(original_feature, self.cfg, global_id, self.meaningless_feat_dict[global_id])

        return {
            "feature": torch.tensor(feature, dtype=torch.float32),
            "y_true": y_true,
            "global_id": global_id,
        }


def get_label_statistics(label_df):
    """for sanity check"""
    weight_list = list(label_df.Weight.unique())
    weight_list.sort()
    print("-" * 50)
    print(f"Number of users:{len(label_df.ID.unique())}")
    print(f"Number of unique global ID: {len(label_df.GLOBAL_ID.unique())}")
    print(f"Unique weights: {weight_list}")
    print(f"Site Distribution:{label_df.groupby('ID').head(1).Site.value_counts()}")
    print("-" * 50)


activity_map = {
    "EXP01": 0,
    "EXP02": 0,
    "EXP03": 1,
    "EXP04": 1,
    "EXP05": 1,
    "EXP06": 1,
    "EXP07": 1,
    "EXP08": 1,
    "EXP09": 2,
    "EXP10": 2,
    "EXP11": 2,
    "EXP12": 3,
    "EXP13": 3,
    "EXP14": 4,
    "EXP15": 5,
    "EXP16": 6,
}
