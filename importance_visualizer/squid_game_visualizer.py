import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from easydict import EasyDict as edict
from importance_visualizer.base_visualizer import BaseVisualizer

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, jaccard_score
import matplotlib.font_manager as font_manager


class SquidGameVisualizer(BaseVisualizer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _load_feature_lists(self):
        return [str(i) for i in range(30)]

    def _define_class_map(self):
        if self.cfg.task.num_class == 3:
            class_map = {
                0: "Class 1",
                1: "Class 2",
                2: "Class 3",
            }
        else:
            raise ValueError("Invalid number of classes")
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

    def plot_classwise_attention(self):
        # Plot heatmap of classwise_relative_importance
        class_accuracy = self._load_model_performance()

        fig, ax1 = plt.subplots(figsize=(20, 8))
        multiplier = self.find_decimal_factor(self.test_class_wise_relative_importance.max())
        ax2 = ax1.twinx()

        cax = inset_axes(
            ax1,
            width="50%",  # width: 40% of parent_bbox width
            height="10%",  # height: 10% of parent_bbox height
            loc="lower left",
            bbox_to_anchor=(0.24, 1.1, 1, 1),
            bbox_transform=ax2.transAxes,
            borderpad=0,
        )
        # convert self.test_class_wise_relative_importance to dataframe and set index to class name
        df = pd.DataFrame(self.test_class_wise_relative_importance, index=self.class_map.values())

        classwise_importance = multiplier * df
        sns.heatmap(
            classwise_importance,
            center=0.0,
            annot=True,
            fmt=".1f",
            cmap="bwr",
            linewidths=0.5,
            ax=ax1,
            robust=True,
            cbar_ax=cax,
            cbar_kws={"orientation": "horizontal"},
            annot_kws={
                "fontsize": 20,
                "fontweight": "bold",
                # 'color':'k',
                "rotation": 90,
            },
        )
        start, end = ax2.get_ylim()
        one_class_dim = int(len(self.feature_lists) / 3)
        shape_range_dim = [one_class_dim, one_class_dim * 2, one_class_dim * 3]
        ax2.set_yticks(
            np.arange(start, end, end / len(class_accuracy)) + 0.19,
            labels=class_accuracy.iloc[::-1],
            fontsize=24,
            rotation=270,
        )
        ax1.set_xticks(
            [shape_range_dim[0], shape_range_dim[1], shape_range_dim[2]],
            labels=["Circle_feature", "Triangle_feature", "Rectangle_feature"],
            rotation=45,
        )

        ax1.axvline(x=shape_range_dim[0], color="g", lw=8, label="Class 1")
        ax1.axvline(x=shape_range_dim[1], color="g", lw=8, label="Class 2")
        ax1.axvline(x=shape_range_dim[2], color="g", lw=8, label="Class 3")
        ax1.axhline(y=1, color="w", lw=10)
        ax1.axhline(y=2, color="w", lw=10)

        trans = ax1.get_xaxis_transform()
        ax1.plot([0.3, 9.7], [-0.022, -0.022], color="#283B7C", transform=trans, clip_on=False, linewidth=13)
        ax1.plot([10.3, 19.7], [-0.022, -0.022], color="#E69535", transform=trans, clip_on=False, linewidth=13)
        ax1.plot([20.3, 29.7], [-0.022, -0.022], color="#DB3831", transform=trans, clip_on=False, linewidth=13)
        # create legends for the axvlines
        fig.subplots_adjust(bottom=0.1, wspace=0.33)
        font = font_manager.FontProperties(weight="bold", size=30)
        ax1.legend(
            ["Circle Feature", "Triangle Feature", "Square Feature"],
            fontsize=20,
            loc="upper center",
            prop=font,
            bbox_to_anchor=(0.5, -0.1),
            fancybox=False,
            shadow=False,
            ncol=3,
        )
        leg = ax1.get_legend()

        for line in leg.get_lines():
            line.set_linewidth(10.0)
            # set solid line style
            line.set_linestyle("-")
        # get first line
        legLine1 = leg.get_lines()[0]
        legLine1.set_color("#283B7C")
        legLine2 = leg.get_lines()[1]
        legLine2.set_color("#E69535")
        legLine3 = leg.get_lines()[2]
        legLine3.set_color("#DB3831")

        # remove x-ticks
        ax1.set_xticks([])
        # set ax1 larger y-ticks font size, bold
        ax1.tick_params(axis="y", which="major", labelsize=30, labelcolor="black")

        # adjust ax2 ylabel position
        ax2.set_ylabel("Accuracy", rotation=270, labelpad=32, fontsize=30)
        # ax2.yaxis.set_label_coords(1.04, 1.03)
        plt.subplots_adjust(top=0.9)
        plt.savefig(
            self.classwise_attention_path,
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=(leg,),
        )

        # binarize classwise_importance
        classwise_importance = classwise_importance.applymap(lambda x: 1 if x > 0 else 0)
        answer_sheet = pd.read_csv(f"data/{self.cfg.task.features_save_dir}/answersheet.csv", index_col=0)
        # flatten answer_sheet to 1darray
        classwise_importance = classwise_importance.values.flatten()
        answer_sheet = answer_sheet.values.flatten()
        # calculate recall, precision, f1, accuracy
        recall = recall_score(answer_sheet, classwise_importance)
        precision = precision_score(answer_sheet, classwise_importance)
        f1 = f1_score(answer_sheet, classwise_importance)
        accuracy = accuracy_score(answer_sheet, classwise_importance)
        jaccard = jaccard_score(answer_sheet, classwise_importance)
        # there can be two answers to this problem
        inverse_answer_sheet = 1 - answer_sheet
        recall_inverse = recall_score(inverse_answer_sheet, classwise_importance)
        precision_inverse = precision_score(inverse_answer_sheet, classwise_importance)
        f1_inverse = f1_score(inverse_answer_sheet, classwise_importance)
        accuracy_inverse = accuracy_score(inverse_answer_sheet, classwise_importance)
        jaccard_inverse = jaccard_score(inverse_answer_sheet, classwise_importance)

        if f1 >= f1_inverse:
            max_recall = recall
            max_precision = precision
            max_f1 = f1
            max_accuracy = accuracy
            max_jaccard = jaccard
            flag = "original"
        else:
            max_recall = recall_inverse
            max_precision = precision_inverse
            max_f1 = f1_inverse
            max_accuracy = accuracy_inverse
            max_jaccard = jaccard_inverse
            flag = "inverse"

        print(
            f"Recall: {recall:.3f}, Precision: {precision:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}, Jaccard: {jaccard:.3f}"
        )
        print(
            f"Recall: {recall_inverse:.3f}, Precision: {precision_inverse:.3f}, F1: {f1_inverse:.3f}, Accuracy: {accuracy_inverse:.3f}, Jaccard: {jaccard_inverse:.3f}"
        )
        print(f"Flag:{flag}")
        # save to dataframe
        score_df = pd.DataFrame(
            {
                "Recall_original": [recall],
                "Precision_original": [precision],
                "F1_original": [f1],
                "Accuracy_original": [accuracy],
                "Jaccard_original": [jaccard],
                "Recall_inverse": [recall_inverse],
                "Precision_inverse": [precision_inverse],
                "F1_inverse": [f1_inverse],
                "Accuracy_inverse": [accuracy_inverse],
                "Jaccard_inverse": [jaccard_inverse],
                "Recall_max": [max_recall],
                "Precision_max": [max_precision],
                "F1_max": [max_f1],
                "Accuracy_max": [max_accuracy],
                "Jaccard_max": [max_jaccard],
                "Flag": [flag],
                "EXP_NAME": [self.cfg.exp_num],
                "cv_num": [self.cfg.task.validation_cv_num],
                "model_name": [self.cfg.model.model_name],
            }
        )
        score_df.to_csv(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_cwri_scores.csv", index=False
        )


if __name__ == "__main__":
    model_name = {
        6000: "shufflenet",
        6001: "resnet9",
        6002: "mlpmixer",
        6003: "simplevit",
        6100: "shufflenet",
        6101: "shufflenet",
        6102: "shufflenet",
        6103: "shufflenet",
        6200: "resnet9",
        6201: "resnet9",
        6202: "resnet9",
        6203: "resnet9",
        6300: "mlpmixer",
        6301: "mlpmixer",
        6302: "mlpmixer",
        6303: "mlpmixer",
        6400: "simplevit",
        6401: "simplevit",
        6402: "simplevit",
        6403: "simplevit",
    }
    for exp_num in [
        6000,
        6001,
        6002,
        6003,
        6100,
        6101,
        6102,
        6103,
        6200,
        6201,
        6202,
        6203,
        6300,
        6301,
        6302,
        6303,
        6400,
        6401,
        6402,
        6403,
    ]:
        for cv_num in [0, 1, 2, 3, 4]:
            print(exp_num, cv_num)
            cfg = edict(
                {
                    "save_classwise_attention_path": "outputs/squid_game/classwise_attention",
                    "save_global_attention_path": "outputs/squid_game/global_attention",
                    "exp_num": exp_num,
                    "save_output_path": f"outputs/squid_game/EXP{exp_num}",
                    "model": {"model_name": model_name[exp_num]},
                    "task": {
                        "features_save_dir": "num_3classsamples_per_class18000_time32_shape30",
                        "validation_cv_num": cv_num,
                        "in_channels": 30,
                        "num_class": 3,
                    },
                }
            )
            summarizer = SquidGameVisualizer(cfg)
            summarizer.plot_global_attention()
            summarizer.plot_classwise_attention()
