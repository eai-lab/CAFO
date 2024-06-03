import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from easydict import EasyDict as edict
from importance_visualizer.base_visualizer import BaseVisualizer

GILON_14_FEATURE_LISTS = [
    "right_accel_x",
    "right_accel_y",
    "right_accel_z",
    "left_accel_x",
    "left_accel_y",
    "left_accel_z",
    "right_fsr1",
    "right_fsr2",
    "right_fsr3",
    "right_fsr4",
    "left_fsr1",
    "left_fsr2",
    "left_fsr3",
    "left_fsr4",
]

GILON_15_FEATURE_LISTS = [
    "right_accel_x",
    "right_accel_y",
    "right_accel_z",
    "left_accel_x",
    "left_accel_y",
    "left_accel_z",
    "right_fsr1",
    "right_fsr2",
    "right_fsr3",
    "right_fsr4",
    "left_fsr1",
    "left_fsr2",
    "left_fsr3",
    "left_fsr4",
    "pseudo_channel",
]


class GilonActionVisualizer(BaseVisualizer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _load_feature_lists(self):
        if self.cfg.task.in_channels == 14:
            feature_lists = GILON_14_FEATURE_LISTS
        elif self.cfg.task.in_channels == 15:
            feature_lists = GILON_15_FEATURE_LISTS
        else:
            raise ValueError("Invalid in_channels")
        return feature_lists

    def _define_class_map(self):
        class_map = {
            0: "Stand Still",
            1: "Treadmill Walk",
            2: "Treadmill Run",
            3: "Ground Walk",
            4: "Squat",
            5: "Lunge",
            6: "Jumping Jack",
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

        # Test dataframe with feature names and global importance
        test_df = pd.DataFrame({"Feature": self.feature_lists, "Importance Score": self.test_global_importance})
        test_df = test_df.sort_values(by="Importance Score", ascending=False)
        # plot seaborn barplot for both df and test_df side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        sns.barplot(x="Importance Score", y="Feature", data=df, ax=ax1, palette="Blues_d")
        sns.barplot(x="Importance Score", y="Feature", data=test_df, ax=ax2, palette="Blues_d")
        ax1.set_title("Train Global Importance")
        ax2.set_title("Test Global Importance")
        plt.tight_layout()
        plt.savefig(self.global_attention_path)

    def plot_train_global_attention(self):
        df = pd.DataFrame({"Feature": self.feature_lists, "Importance Score": self.global_importance})
        df = df.sort_values(by="Importance Score", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 20))
        sns.barplot(x="Importance Score", y="Feature", data=df, ax=ax, palette="Blues_d")
        ax.set_title("GI Score", fontsize=56, fontweight="bold", pad=20)
        # set feature names larger, bold font
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=24, fontweight="bold")
        # set y label larger
        ax.set_ylabel("Features", fontsize=32, labelpad=20, fontweight="bold")
        # remove x label
        ax.set_xlabel("")
        # set larger x ticks
        ax.tick_params(axis="x", labelsize=20)
        plt.savefig(self.global_attention_path, bbox_inches="tight", dpi=300)

    def plot_classwise_attention(self):
        # Plot heatmap of classwise_relative_importance
        class_accuracy = self._load_model_performance()

        fig, ax1 = plt.subplots(figsize=(15, 20))
        multiplier = self.find_decimal_factor(self.test_class_wise_relative_importance.max())
        ax2 = ax1.twinx()

        cax = inset_axes(
            ax1,
            width="40%",  # width: 40% of parent_bbox width
            height="10%",  # height: 10% of parent_bbox height
            loc="lower left",
            bbox_to_anchor=(0.6, 1.05, 1, 1),
            bbox_transform=ax2.transAxes,
            borderpad=0,
        )
        # convert self.test_class_wise_relative_importance to dataframe and set index to class name
        df = pd.DataFrame(self.test_class_wise_relative_importance, index=self.class_map.values())
        # set title, shift title a bit to the right
        ax1.set_title("CWRI Score", fontsize=56, fontweight="bold", pad=20, x=0.23, y=1.03)
        classwise_importance = multiplier * df
        sns.heatmap(
            classwise_importance,
            center=0.0,
            annot=True,
            fmt=".1f",
            cmap="bwr",
            linewidths=0.5,
            ax=ax1,
            cbar_ax=cax,
            cbar_kws={"orientation": "horizontal"},
            annot_kws={"fontsize": 14, "fontweight": "bold", "color": "k"},
        )
        start, end = ax2.get_ylim()
        # leg = ax1.get_legend()
        ax2.set_yticks(
            np.arange(start, end, end / len(class_accuracy)) + 0.07, labels=class_accuracy.iloc[::-1]
        )  # Need to check this label
        # remove x ticks
        ax1.set_xticks([])
        ax1.axvline(x=3, color="r", ls="--", lw=4, label="accelerometer Right")
        ax1.axvline(x=6, color="b", ls="--", lw=4, label="accelerometer Left")
        ax1.axvline(x=10, color="g", ls="--", lw=4, label="FSR Right")
        ax1.axvline(x=13.9, color="y", ls="--", lw=4, label="FSR Left")

        trans = ax1.get_xaxis_transform()
        ax1.plot([0.1, 2.9], [-0.011, -0.011], color="red", transform=trans, clip_on=False, linewidth=8)
        ax1.plot([3.1, 5.9], [-0.011, -0.011], color="blue", transform=trans, clip_on=False, linewidth=8)
        ax1.plot([6.1, 9.9], [-0.011, -0.011], color="green", transform=trans, clip_on=False, linewidth=8)
        ax1.plot([10.1, 13.9], [-0.011, -0.011], color="#F09436", transform=trans, clip_on=False, linewidth=8)
        # ax2.set_ylabel("Accuracy", rotation=0, labelpad=30)
        fig.subplots_adjust(bottom=0.1, wspace=0.33)
        ax1.legend(
            ["Accelerometer Right", "Accelerometer Left", "FSR Right", "FSR Left"],
            fontsize=24,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            fancybox=False,
            shadow=False,
            ncol=4,
        )
        leg = ax1.get_legend()

        for line in leg.get_lines():
            line.set_linewidth(10.0)
            # set solid line style
            line.set_linestyle("-")
        # get first line
        legLine1 = leg.get_lines()[0]
        legLine1.set_color("r")
        legLine2 = leg.get_lines()[1]
        legLine2.set_color("b")
        legLine3 = leg.get_lines()[2]
        legLine3.set_color("g")
        legLine4 = leg.get_lines()[3]
        legLine4.set_color("#F09436")

        # Set y label
        ax1.set_ylabel("Class", fontsize=32, labelpad=16, fontweight="bold")
        ax2.set_ylabel("Accuracy", fontsize=32, labelpad=40, rotation=270, fontweight="bold")
        # set larger y ticks
        # ax1.tick_params(axis="y", labelsize=24, rotation=0)
        # set to bold for ax2 y ticks
        ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=24, fontweight="bold", rotation=0)
        ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=24, fontweight="bold")
        plt.subplots_adjust(top=0.9)
        plt.savefig(
            self.classwise_attention_path,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=(leg,),
        )


if __name__ == "__main__":
    exp_num = 303
    cv_num = 2
    cfg = edict(
        {
            "save_classwise_attention_path": "outputs/gilon_activity/classwise_attention",
            "save_global_attention_path": "outputs/gilon_activity/global_attention",
            "exp_num": exp_num,
            "save_output_path": f"outputs/gilon_activity/EXP{exp_num}",
            "task": {"validation_cv_num": cv_num, "in_channels": 14},
        }
    )
    summarizer = GilonActionVisualizer(cfg)
    summarizer.plot_train_global_attention()
    summarizer.plot_classwise_attention()
