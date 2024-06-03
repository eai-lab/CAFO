import pandas as pd
import numpy as np
import os
import glob
import shutil
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from model.shufflenet_cafo import ShuffleNet
from model.resnet9_cafo import ResNet9


from lightning_models.viz_confusion import make_confusion_matrix


class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        assert self.cfg.task.task_name in [
            "gilon_activity",
            "gilon_speed",
            "microsoft_activity",
            "microsoft_activity_new",
            "squid_game",
            "fingergesture",
            "SpokenArabicDigits",
        ]
        self.lr = cfg.task.optimizer.lr
        self.model_name = cfg.model.model_name
        if self.model_name == "shufflenet":
            self.model = ShuffleNet(cfg)
        elif self.model_name == "resnet9":
            self.model = ResNet9(cfg)
        elif self.model_name == "simplevit":
            from model.simpleVIT_cafo import SimpleVIT

            self.model = SimpleVIT(cfg)
        elif self.model_name == "mlpmixer":
            from model.mixer_cafo import MlpMixer

            self.model = MlpMixer(cfg)
        else:
            raise ValueError(f"Unknown model {self.model_name}")
        self.num_params = sum([p.numel() for p in self.model.parameters()])
        self.log("num_params", self.num_params)

        self.loss = nn.CrossEntropyLoss()
        self.predict_counter = 0

    def training_step(self, batch, batch_idx):
        feature, y_true, global_id = batch["feature"], batch["y_true"], batch["global_id"]
        if batch_idx == 0:
            y_pred, model_outputs = self.model(feature, return_model_outputs=True)
            channel_attention = model_outputs["channel_attention"]
            if self.cfg.channelattention.name == "simam":
                channel_attention = torch.mean(channel_attention.view(channel_attention.shape[:2] + (-1,)), dim=-1)
            self.save_attention(channel_attention, type="channel")
            if self.current_epoch == 0:
                # since sampling is fixed for each epoch, we only need to save the attention map for the first epoch
                self.save_feature(feature)
                self.attention_label = self.save_attention_labels(y_true, global_id)
            self.print_classwise_attention(channel_attention)
        else:
            y_pred, model_outputs = self.model(feature, self.current_epoch, return_model_outputs=True)

        loss = self.loss(y_pred.squeeze(), y_true)
        self.log("loss", loss.item(), on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        y_pred = self.model(feature, return_model_outputs=False)
        loss = self.loss(y_pred.squeeze(), y_true)
        return {"val_loss": loss, "y_pred": y_pred, "label": y_true}

    def validation_step_end(self, outputs):
        return outputs

    def validation_epoch_end(self, outputs):
        # Collect results from all validation batches
        y_preds = torch.cat([torch.argmax(x["y_pred"], dim=1) for x in outputs]).cpu()
        labels = torch.cat([x["label"] for x in outputs]).cpu()
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        accuracy = accuracy_score(y_preds, labels)
        print(f"val_loss: {val_loss:.4f}, accuracy: {accuracy:.4f}")
        self.log_dict({"val_loss": val_loss, "val_accuracy": accuracy})

    def test_step(self, batch, batch_idx):
        feature, y_true = batch["feature"], batch["y_true"]
        y_pred, model_outputs = self.model(feature, return_model_outputs=True)
        channel_attention = model_outputs["channel_attention"]
        if self.cfg.channelattention.name == "simam":
            channel_attention = torch.mean(channel_attention.view(channel_attention.shape[:2] + (-1,)), dim=-1)
        loss = self.loss(y_pred.squeeze(), y_true)
        return {
            "y_pred": y_pred,
            "y_true": y_true,
            "channel_attention": channel_attention,
            "test_loss": loss,
        }

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x["test_loss"] for x in outputs]).cpu().mean()
        y_preds = torch.cat([torch.argmax(x["y_pred"], 1) for x in outputs]).cpu()
        y_true = torch.cat([x["y_true"] for x in outputs]).cpu()
        channel_attention = torch.cat([x["channel_attention"] for x in outputs]).cpu()
        self.save_test_attention(channel_attention, type="channel")
        self.summarize_test_attention_npy_file(y_true)

        # Save the test results in the output directory
        test_label = pd.read_csv(f"{self.cfg.save_output_path}/test_label.csv")
        test_label["test_loss"] = test_loss.item()
        test_label["seed"] = self.cfg.seed
        test_label["exp_num"] = self.cfg.exp_num
        test_label["stop_epoch"] = self.trainer.early_stopping_callback.stopped_epoch
        test_label["y_pred"] = y_preds.numpy()
        test_label["y_true"] = y_true.numpy()
        test_label["cv_num"] = self.cfg.task.validation_cv_num
        test_label["model_name"] = self.cfg.model.model_name
        test_label["task_name"] = self.cfg.task.task_name
        test_label["project_name"] = self.cfg.logger.name
        test_label.to_csv(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_test_label.csv", index=False
        )

        activity_cf_mat = confusion_matrix(y_true, y_preds)
        make_confusion_matrix(
            activity_cf_mat, f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_confusion_matrix.png"
        )

        self.logger.experiment["test_labelfile"].upload(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_test_label.csv"
        )
        self.logger.experiment["confusion_matrix"].upload(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_confusion_matrix.png"
        )
        self.log_dict({"test_accuracy": accuracy_score(y_true, y_preds)})

    def predict_step(self, batch, batch_idx):
        """"""
        feature, y_true = batch["feature"], batch["y_true"]
        y_pred_original = self.model(feature, return_model_outputs=False)
        # shake-drop
        y_pred, model_outputs = self.model(feature, predict_mode=True, return_model_outputs=True)
        drop_mask = model_outputs["drop_mask"]
        return {
            "y_pred_original": y_pred_original,
            "y_pred": y_pred,
            "y_true": y_true,
            "channel_attention": model_outputs["channel_attention"],
            "drop_mask": drop_mask,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.cfg.task.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.task.scheduler_epochs)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def save_attention(self, attention_score, type="channel"):
        """save attention during training"""
        attention_score_npy = attention_score.cpu().detach().numpy()
        # remove file if exists
        if self.current_epoch == 0:
            if os.path.exists(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}"):
                print(f"remove {self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}")
                shutil.rmtree(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}")
                # remove files starting with cv{self.cfg.task.validation_cv_num}
                for f in os.listdir(self.cfg.save_output_path):
                    if f.startswith(f"cv{self.cfg.task.validation_cv_num}"):
                        os.remove(f"{self.cfg.save_output_path}/{f}")
        if not os.path.exists(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}"):
            os.makedirs(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}")

        with open(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/{type}_attention_{str(self.current_epoch).zfill(3)}.npy",
            "wb",
        ) as f:
            np.save(f, attention_score_npy)
        print(
            f"Save attention score in {self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/{type}_attention_{str(self.current_epoch).zfill(3)}.npy"
        )
        # print class-wise attention scores

    def print_classwise_attention(self, attention_score):
        attention_score_npy = attention_score.cpu().detach().numpy()
        attention_scores_df = pd.DataFrame(attention_score_npy.squeeze())
        epoch_attention_mean_file = (
            pd.concat([self.attention_label[["y_true"]], attention_scores_df], axis=1)
            .groupby("y_true")
            .mean()
            .reset_index()
        )
        print(f"current epoch: {self.current_epoch}")
        print(epoch_attention_mean_file)

    def save_test_attention(self, attention_score, type="channel"):
        """save attention for test set"""
        print(f"Save attention score in {self.cfg.save_output_path}/test_{type}_attention.npy")
        attention_score_npy = attention_score.cpu().detach().numpy()
        # remove file if exists
        if os.path.exists(f"{self.cfg.save_output_path}/test_cv{self.cfg.task.validation_cv_num}"):
            print(f"remove {self.cfg.save_output_path}/test_{self.cfg.task.validation_cv_num}")
            shutil.rmtree(f"{self.cfg.save_output_path}/test_{self.cfg.task.validation_cv_num}")
        if not os.path.exists(f"{self.cfg.save_output_path}/test_{self.cfg.task.validation_cv_num}"):
            os.makedirs(f"{self.cfg.save_output_path}/test_{self.cfg.task.validation_cv_num}")

        with open(
            f"{self.cfg.save_output_path}/test_{self.cfg.task.validation_cv_num}/{type}_test_attention.npy", "wb"
        ) as f:
            np.save(f, attention_score_npy)
        print(
            f"Save attention score in {self.cfg.save_output_path}/test_{self.cfg.task.validation_cv_num}/{type}_test_attention.npy"
        )

    def save_feature(self, feature):
        feature_npy = feature.cpu().detach().numpy()
        with open(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/feature.npy", "wb") as f:
            np.save(f, feature_npy)
        print(f"Save feature in {self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/feature.npy")

    def save_attention_labels(self, label, global_ids):
        """save true label and global id to project the attention score back to the original data"""
        label = label.cpu().detach().numpy()
        global_ids = global_ids.cpu().detach().numpy()
        attention_label = pd.DataFrame({"global_id": global_ids, "y_true": label})
        attention_label.to_csv(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/attention_label.csv", index=False
        )
        print(
            f"saved attention label in {self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/attention_label.csv"
        )
        return attention_label

    def summarize_attention_npy_files(self):
        """summarize attention score files in the output directory"""
        attention_label = pd.read_csv(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/attention_label.csv"
        )
        attention_score_files = glob.glob(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/channel_attention_*.npy"
        )

        attention_score_files.sort()
        attention_scores = []
        for attention_score_file in attention_score_files:
            attention_score = np.load(attention_score_file)
            attention_scores_df = pd.DataFrame(attention_score.squeeze())
            epoch_attention_mean_file = (
                pd.concat([attention_label[["y_true"]], attention_scores_df], axis=1)
                .groupby("y_true")
                .mean()
                .reset_index()
            )
            epoch_attention_mean_file["epoch"] = int(attention_score_file.split("_")[-1].split(".")[0])
            attention_scores.append(epoch_attention_mean_file)
        attention_scores = pd.concat(attention_scores)
        attention_scores.to_csv(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/channel_attention_scores.csv",
            index=False,
        )

        return attention_scores

    def flush_attention_files(self, best_ckpt_path):
        best_epoch = int(best_ckpt_path.split("=")[-1].split(".")[0])
        attention_score_files = glob.glob(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/channel_attention_*.npy"
        )
        attention_score_files.sort()
        for attention_score_file in attention_score_files:
            if int(attention_score_file.split("_")[-1].split(".")[0]) > best_epoch:
                print(f"remove {attention_score_file}")
                os.remove(attention_score_file)

    def summarize_test_attention_npy_file(self, y_true):
        """summarize attention score for inference results"""
        attention_score_files = glob.glob(
            f"{self.cfg.save_output_path}/test_{self.cfg.task.validation_cv_num}/channel_test_attention.npy"
        )
        attention_score = np.load(attention_score_files[0])
        attention_scores_df = pd.DataFrame(attention_score.squeeze())
        attention_scores_df["y_true"] = y_true
        attention_scores_df.to_csv(
            f"{self.cfg.save_output_path}/test_{self.cfg.task.validation_cv_num}/channel_test_attention_scores.csv",
            index=False,
        )
