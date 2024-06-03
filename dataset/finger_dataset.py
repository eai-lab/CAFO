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
            self.train = FingerDataset("train", self.cfg)
            self.val = FingerDataset("val", self.cfg)
            self.train_sampler = ConstantRandomSampler(self.train)
        elif stage == "test":
            self.test = FingerDataset("test", self.cfg)
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


class FingerDataset(Dataset):
    """Load HAR data"""

    def __init__(self, mode, cfg):
        print(bcolors.OKBLUE + bcolors.BOLD + f"{mode} Mode" + bcolors.ENDC + bcolors.ENDC)
        self.mode = mode
        self.cfg = cfg
        self.cv_val_num = cfg.task.validation_cv_num
        assert self.cfg.task.task_name == "fingergesture"

        data_path = f"data/{cfg.task.features_save_dir}/fingergesture.pkl"
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)

        self.label_df = pd.read_csv(f"data/{cfg.task.features_save_dir}/label_df.csv")

        if mode == "train":
            self.label_df = self.label_df[
                (self.label_df["cv_num"] != self.cv_val_num) & (self.label_df["cv_num"] != -1)
            ]
            # shuffle
            self.label_df = self.label_df.sample(frac=1).reset_index(drop=True)
        elif mode == "val":
            self.label_df = self.label_df[self.label_df["cv_num"] == self.cv_val_num]
        elif mode == "test":
            self.label_df = self.label_df[self.label_df["cv_num"] == -1]

        self.label_df.reset_index(drop=True, inplace=True)

        if mode == "test":
            self.label_df.to_csv(f"{self.cfg.save_output_path}/test_label.csv", index=False)

        get_label_statistics(self.label_df)

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, index):
        # TODO: Check if the weighted sampling is working properly
        global_id = self.label_df.loc[index, "GLOBAL_ID"]
        y_true = self.label_df.loc[index, "y_true"]

        original_feature = self.data[global_id]

        if self.cfg.image_encoding == "rp":
            feature = generate_rp(original_feature, self.cfg, global_id)
        elif self.cfg.image_encoding == "gramian":
            feature = generate_gramian(original_feature, self.cfg, global_id)

        return {
            "feature": torch.tensor(feature, dtype=torch.float32),
            "y_true": y_true,
            "global_id": global_id,
        }


def get_label_statistics(label_df):
    """for sanity check"""
    print("-" * 50)
    label_subject_ids = label_df.user_id.unique()
    print(f"Number of users:{len(label_subject_ids)}")
    print(f"Subject IDs:{label_subject_ids}")
    print(f"Number of unique global ID: {len(label_df.GLOBAL_ID.unique())}")
    print(f"==========Label Distribution========")
    print(label_df.y_true.value_counts())
    print("-" * 50)
