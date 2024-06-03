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
from dataset.make_images import generate_rp,generate_gramian


class CAFODataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train = SquidGameDataset("train", self.cfg)
            self.val = SquidGameDataset("val", self.cfg)
            self.train_sampler = ConstantRandomSampler(self.train)
            # self.val_sampler = ConstantRandomSampler(self.val)
        elif stage == "test":
            self.test = SquidGameDataset("test", self.cfg)
            # self.test_sampler = ConstantRandomSampler(self.train)
        else:
            raise ValueError(f"Unknown stage {stage}")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
            sampler=self.train_sampler,
            shuffle=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.cfg.dataset.batch_size,
            num_workers=self.cfg.dataset.num_workers,
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


class SquidGameDataset(Dataset):
    def __init__(self, mode, cfg):
        print(bcolors.OKBLUE + bcolors.BOLD + f"{mode} Mode" + bcolors.ENDC + bcolors.ENDC)
        self.mode = mode
        self.cfg = cfg
        self.cv_val_num = cfg.task.validation_cv_num
        assert self.cfg.task.task_name == "squid_game"
        assert int(self.cv_val_num) <= 4

        if mode in ("train", "val"):
            self.label_df = pd.read_csv(f"data/{cfg.task.features_save_dir}/train_label.csv")
        elif mode == "test":
            self.label_df = pd.read_csv(f"data/{cfg.task.features_save_dir}/test_label.csv")

        if mode in ("train", "val"):
            feature_dict_save_path = f"data/{cfg.task.features_save_dir}/feature_dict_train.pkl"
            if os.path.isfile(feature_dict_save_path):
                print(f"Loading feature_dict from {feature_dict_save_path}")
                with open(feature_dict_save_path, "rb") as f:
                    self.feature_dict = pickle.load(f)

            if mode in ("train"):
                self.label_df = self.label_df[self.label_df.cv_split != self.cv_val_num]
            elif mode in ("val"):
                self.label_df = self.label_df[self.label_df.cv_split == self.cv_val_num]

        elif mode == "test":
            feature_dict_save_path = f"data/{cfg.task.features_save_dir}/feature_dict_test.pkl"
            if os.path.isfile(feature_dict_save_path):
                print(f"Loading feature_dict from {feature_dict_save_path}")
                with open(feature_dict_save_path, "rb") as f:
                    self.feature_dict = pickle.load(f)

        self.label_df = self.label_df.reset_index(drop=True)

        if mode in ("test"):
            self.label_df.to_csv(f"{self.cfg.save_output_path}/test_label.csv", index=False)

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        global_id = self.label_df.iloc[idx]["GLOBAL_ID"]
        y_true = self.label_df.iloc[idx]["class"]
        original_feature = self.feature_dict[global_id]
        y_true = torch.tensor(y_true, dtype=torch.long)
        if self.cfg.image_encoding =='rp':
            feature = generate_rp(original_feature, self.cfg, global_id)
        elif self.cfg.image_encoding =='gramian':
            feature = generate_gramian(original_feature, self.cfg, global_id)
        else:
            raise ValueError(f"Unknown image_encoding {self.cfg.image_encoding}")
        return {
            "feature": torch.tensor(feature, dtype=torch.float32),
            "y_true": y_true,
            "global_id": global_id,
        }
