import pandas as pd
import numpy as np
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from easydict import EasyDict as edict
#from importance_visualizer.base_visualizer import BaseVisualizer
from base_visualizer import BaseVisualizer
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, jaccard_score

fingergesture_feature_lists = [
    "thumb_low",
    "thumb_high",
    "index_low",
    "index_high",
    "middle_low",
    "middle_high",
    "ring_low",
    "ring_high",
    "pinky_low",
    "pinky_high",
]


class FingerGestureVisualizer(BaseVisualizer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _load_feature_lists(self):
        return fingergesture_feature_lists

    def _define_class_map(self):
        class_map = {
            0: "thumb\nonly",
            1: "thumb\nexcept",
            2: "index\nonly",
            3: "index\nexcept",
            4: "middle\nonly",
            5: "middle\nexcept",
            6: "ring\nonly",
            7: "ring\nexcept",
            8: "pinky\nonly",
            9: "pinky\nexcept",
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

    def plot_classwise_attention(self):
        # Plot heatmap of classwise_relative_importance
        class_accuracy = self._load_model_performance()

        fig, ax1 = plt.subplots(figsize=(10, 10))
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
        # convert self.test_class_wise_relative_importance to dataframe and set index to class name
        df = pd.DataFrame(self.test_class_wise_relative_importance, index=self.class_map.values())

        classwise_importance = multiplier * df
        # set elements larger 
        sns.heatmap(
            classwise_importance,
            center=0.0,
            annot=True,
            fmt=".1f",
            cmap="bwr",
            linewidths=0.5,
            ax=ax1,
            cbar_ax=cax,
            vmin = -2,
            vmax = 2,
            cbar_kws={"orientation": "horizontal"},
            annot_kws={"size": 17, "fontweight": "bold"},

        )
        
        start, end = ax2.get_ylim()
        # make ax1 yticks bold
        ax1.set_yticklabels(ax1.get_yticklabels(), fontweight="bold", rotation=0, fontsize=20)

        ax2.set_yticks(
            np.arange(start, end, end / len(class_accuracy)) + 0.07, labels=class_accuracy.iloc[::-1], fontsize=16, fontweight="bold"
        )  # Need to check this label
        sensor_labels = ["Thumb \nSensors", "Index \nSensors ", "Middle \nSensors", "Ring \nSensors", "Pinky \nSensors"]
        # place xticks on top of heatmap
        ax1.xaxis.tick_top()
        ax1.set_xticks([1,  3, 5, 7, 9], labels=sensor_labels, fontsize=18, fontweight="bold")
        color_codes = ['#e9311a', '#F8D949', '#56BCBE', '#B023B9', '#026c80']
        for idx, x in enumerate([2,4,6,8,10]):
            ax1.axvline(x=x, color=color_codes[idx], ls="--", lw=9)
        # ax1.axvline(x=4, color="r", ls="--", lw=2, label="AB")
        # ax1.axvline(x=8, color="b", ls="--", lw=2, label="CD")
        ax2.set_ylabel("Accuracy", rotation=270, labelpad=30, fontsize=18, fontweight="bold")

        # # get colorbar
        cbar = ax1.collections[0].colorbar  
        # set colorbar ticks bold 
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontweight="bold", fontsize=15)
        # remove colorbar
        cax.remove()
        plt.savefig(
            self.classwise_attention_path,
            dpi=300,
            bbox_inches="tight",
            # bbox_extra_artists=[cax],
        )

        # plot the heatmap. blue for positive, red for negative
        plt.figure(figsize=(10,10))
        # rotate colorbar label
        sns.heatmap(classwise_importance, cmap='bwr_r', vmin=-1, vmax=1, center=0, square=True, linewidths=.5, 
                    cbar_kws={"shrink": .5, "ticks":[-1, -0.5, 0, 0.5, 1], "label": "Spearmanr", "fraction":0.1})

        # # get colorbar
        cbar = plt.gcf().axes[-1]
        cbar.yaxis.set_ticks_position('left')
        cbar.yaxis.set_label_position('right')

        # rotate colorbar
        cbar.set_yticklabels(cbar.get_yticklabels(), rotation=270, fontsize=20)

        # make bold
        cbar.set_yticklabels(['2', '1', '0', '-1', '-2'], fontsize=20, fontweight='bold')
        cbar.set_ylabel('CWRI', rotation=270, labelpad=40, fontsize=30, fontweight='bold')
        # set larger colorbar size
        cbar.set_position([.85, .11, .03, .77])
        # remove heatmap and leave only colorbar
        plt.gca().set_visible(False)
        # save 
        plt.savefig("colorbar.png", dpi=300, bbox_inches='tight')

        # binarize classwise_importance
        classwise_importance = classwise_importance.applymap(lambda x: 1 if x > 0 else 0)
        answer_sheet = pd.read_csv(f"data/fingergesture/answer_sheet.csv", index_col=0)
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
                "CV_NUM": [self.cfg.task.validation_cv_num],
                "Model_name": [self.cfg.model.model_name],
            }
        )
        score_df.to_csv(
            f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}_cwri_scores.csv", index=False
        )


if __name__ == "__main__":
    model_name = {
        7002: "mlpmixer",
    }

    for exp_num in [7002]:
        for cv_num in [2]:
            cfg = edict(
                {
                    "save_classwise_attention_path": "outputs/fingergesture/classwise_attention",
                    "save_global_attention_path": "outputs/fingergesture/global_attention",
                    "exp_num": exp_num,
                    "save_output_path": f"outputs/fingergesture/EXP{exp_num}",
                    "task": {"validation_cv_num": cv_num, "in_channels": 10},
                    "model": {"model_name": model_name[exp_num]},
                }
            )
            summarizer = FingerGestureVisualizer(cfg)
            summarizer.plot_global_attention()
            summarizer.plot_classwise_attention()
    # exp_num = 7002
    # cv_num = 2
    # cfg = edict(
    #     {
    #         "save_classwise_attention_path": "outputs/fingergesture/classwise_attention",
    #         "save_global_attention_path": "outputs/fingergesture/global_attention",
    #         "exp_num": exp_num,
    #         "save_output_path": f"outputs/fingergesture/EXP{exp_num}",
    #         "task": {"validation_cv_num": cv_num, "in_channels": 10},
    #     }
    # )
    # summarizer = FingerGestureVisualizer(cfg)
    # summarizer.plot_global_attention()
    # summarizer.plot_classwise_attention()
