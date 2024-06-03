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
from dataset.make_images import generate_rp


class CAFODataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train = MSHAR_Dataset("train", self.cfg)
            self.val = MSHAR_Dataset("val", self.cfg)
            self.train_sampler = ConstantRandomSampler(self.train)
            # self.val_sampler = ConstantRandomSampler(self.val)
        elif stage == "test":
            self.test = MSHAR_Dataset("test", self.cfg)
            # self.test_sampler = ConstantRandomSampler(self.train)
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
            shuffle=False,
            drop_last=False,
        )


class MSHAR_Dataset(Dataset):
    """Load MS HAR sensor data using global_id value. index is mapped to global id through label_df"""

    def __init__(self, mode, cfg):
        print(bcolors.OKBLUE + bcolors.BOLD + f"{mode} Mode" + bcolors.ENDC + bcolors.ENDC)
        self.mode = mode
        self.cfg = cfg
        self.cv_val_num = cfg.task.validation_cv_num
        assert self.cfg.task.task_name in ["microsoft_activity"]
        assert self.cv_val_num <= 4
        if mode in ("train", "val"):
            self.label_df = pd.read_pickle(
                f"data/{cfg.task.features_save_dir}/MS_train_labels_chunk200_window200_10class.pkl"
            )

        elif mode in ("test"):
            self.label_df = pd.read_pickle(
                f"data/{cfg.task.features_save_dir}/MS_test_labels_chunk200_window200_10class.pkl"
            )

        # Create train dataset. Val is just a subset of train dataset
        if mode in ("train", "val"):
            sensor_feature_save_path = f"data/{cfg.task.features_save_dir}/train_sensors.pkl"
            if os.path.isfile(sensor_feature_save_path):
                print(f"{sensor_feature_save_path} Exists!, loading..")
                with open(sensor_feature_save_path, "rb") as f:
                    self.sensor_dict = pickle.load(f)

            if mode in {"train"}:
                self.label_df = self.label_df[~self.label_df["CV_VAL"].isin([self.cv_val_num])]
            elif mode in {"val"}:
                self.label_df = self.label_df[self.label_df["CV_VAL"].isin([self.cv_val_num])]

        elif mode in ("test"):
            sensor_test_feature_save_path = f"data/{cfg.task.features_save_dir}/test_sensors.pkl"
            if os.path.isfile(sensor_test_feature_save_path):
                print(f"{sensor_test_feature_save_path} Exists!, loading..")
                with open(sensor_test_feature_save_path, "rb") as f:
                    self.sensor_dict = pickle.load(f)

        """If you want to perform any ablation on the datasets, please do it here. all the features will be based on the label_df"""
        get_label_statistics(self.label_df)
        self.label_df = self.label_df.reset_index(drop=True)

        if mode in ("test"):
            self.label_df.to_csv(f"{self.cfg.save_output_path}/test_label.csv", index=False)

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        # TODO: Check if the weighted sampling is working properly
        global_id = self.label_df.iloc[index].GLOBAL_ID
        label = self.label_df[self.label_df["GLOBAL_ID"] == global_id]

        original_feature = self.sensor_dict[global_id]
        activity_label = label.CLASS_LABEL.item()
        y_true = torch.tensor(activity_label, dtype=torch.long)

        feature = generate_rp(original_feature, self.cfg, global_id)
        return {
            "feature": torch.tensor(feature, dtype=torch.float32),
            "y_true": y_true,
            "global_id": global_id,
        }


def get_label_statistics(label_df):
    """for sanity check"""
    CLASS_LABEL_list = list(label_df.CLASS_LABEL.unique())
    CLASS_LABEL_list.sort()
    print("-" * 50)
    print(f"Number of users:{len(label_df.user_id.unique())}")
    print(f"Number of unique global CLASS_LABEL: {len(label_df.GLOBAL_ID.unique())}")
    print(f"Number of unique CLASS_LABELs: {len(label_df.CLASS_LABEL.unique())}")
    print("-" * 50)
