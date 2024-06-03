import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from easydict import EasyDict as edict
from importance_visualizer.base_visualizer import BaseVisualizer

MS_feature_lists = ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]


class MicrosoftActivityVisualizer(BaseVisualizer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _load_feature_lists(self):
        return MS_feature_lists

    def _define_class_map(self):
        class_map = {
            0: "Bicep Curl",
            1: "Biceps Curl (band)",
            2: "Jump Rope",
            3: "Plank",
            4: "Pushups",
            5: "Squat",
            6: "Squat (hands behind head)",
            7: "Squat Jump",
            8: "Walk",
            9: "Walking lunge",
        }
        return class_map

    def _load_model_performance(self):
        """load cv label"""
        cv_label = pd.read_csv(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_test_label.csv")
        # calculate accuracy by class and store accuracy as string fmt
        class_accuracy = cv_label.groupby("y_true").apply(lambda x: np.sum(x["y_pred"] == x["y_true"]) / len(x))
        class_accuracy = class_accuracy.apply(lambda x: f"{x*100:.2f}")
        return class_accuracy

    def plot_global_attention(self):
        # Construct dataframe with feature names and global importance
        df = pd.DataFrame({"Feature": self.feature_lists, "Importance Score": self.global_importance})
        df = df.sort_values(by="Importance Score", ascending=False)
        # plot seaborn barplot for both df and test_df side by side
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.barplot(x="Importance Score", y="Feature", data=df, ax=ax, palette="Blues_d")
        ax.set_title("Train Global Importance")
        plt.tight_layout()

        plt.savefig(self.global_attention_path)
        print(f"Global Attention plot saved at {self.global_attention_path}")

    def plot_classwise_attention(self):
        # Plot heatmap of classwise_relative_importance
        class_accuracy = self._load_model_performance()

        fig, ax1 = plt.subplots(figsize=(10, 6))
        multiplier = self.find_decimal_factor(self.test_class_wise_relative_importance.max())
        ax2 = ax1.twinx()

        cax = inset_axes(
            ax1,
            width="40%",  # width: 40% of parent_bbox width
            height="10%",  # height: 10% of parent_bbox height
            loc="lower left",
            bbox_to_anchor=(0.6, 1.2, 1, 1),
            bbox_transform=ax2.transAxes,
            borderpad=0,
        )
        # convert self.class_wise_relative_importance to dataframe and set index to class name
        df = pd.DataFrame(self.test_class_wise_relative_importance, index=self.class_map.values())

        sns.heatmap(
            multiplier * df,
            center=0.0,
            annot=True,
            fmt=".3f",
            cmap="bwr",
            linewidths=0.5,
            ax=ax1,
            cbar_ax=cax,
            cbar_kws={"orientation": "horizontal"},
        )
        start, end = ax2.get_ylim()

        ax2.set_yticks(
            np.arange(start, end, end / len(class_accuracy)) + 0.07, labels=class_accuracy.iloc[::-1]
        )  # Need to check this label
        ax1.set_xticks([1, 2, 3, 4, 5, 6], labels=["acc_x", "acc_y", "acc_z", "gy_x", "gy_y", "gy_z"], rotation=45)
        ax1.axvline(x=3, color="r", ls="--", lw=2, label="accelerometer Right")
        ax1.axvline(x=6, color="b", ls="--", lw=2, label="accelerometer Left")
        ax1.axvline(x=10, color="g", ls="--", lw=2, label="FSR Right")
        ax1.axvline(x=14, color="y", ls="--", lw=2, label="FSR Left")
        ax2.set_ylabel("Accuracy", rotation=0, labelpad=30)

        plt.savefig(
            self.classwise_attention_path,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[cax],
        )
        print(f"Classwise Attention plot saved at {self.classwise_attention_path}")


if __name__ == "__main__":
    exp_num = 7002
    cv_num = 2
    cfg = edict(
        {
            "save_classwise_attention_path": "outputs/fingergesture/classwise_attention",
            "save_global_attention_path": "outputs/fingergesture/global_attention",
            "exp_num": exp_num,
            "save_output_path": f"outputs/fingergesture/EXP{exp_num}",
            "task": {"validation_cv_num": cv_num, "in_channels": 10},
        }
    )
    summarizer = GilonActionVisualizer(cfg)
    summarizer.plot_train_global_attention()
    summarizer.plot_classwise_attention()
