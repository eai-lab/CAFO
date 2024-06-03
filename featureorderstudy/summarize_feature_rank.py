import os
import sys
import numpy as np
import pandas as pd
import pickle
import datetime
import glob
import hydra
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error


@hydra.main(version_base=None, config_path=".", config_name="summarize_rank")
def main(cfg: DictConfig) -> None:
    # check cfg file
    if cfg.model_name == -1:
        assert cfg.model_name != -1, "Please specify model name"
    if cfg.task_name == -1:
        assert cfg.task_name != -1, "Please specify task name"
    if cfg.base_exp_num == -1:
        assert cfg.base_exp_num != -1, "Please specify base exp num"
    if cfg.start_exp_num == -1:
        assert cfg.start_exp_num != -1, "Please specify start exp num"
    if cfg.end_exp_num == -1:
        assert cfg.end_exp_num != -1, "Please specify end exp num"

    assert cfg.exp_type in ["important", "unimportant", "random"]

    # collect base exp results
    base_exp_dirs = glob.glob(f"outputs/{cfg.task_name}/EXP{cfg.base_exp_num}/*_test_label.csv")
    base_exp_dirs.sort()

    # collect all accuracy for experiments from start_exp_num to end_exp_num
    ablation_exp_dirs = glob.glob(f"outputs/{cfg.task_name}/EXP*/*_test_label.csv")
    ablation_exp_dirs = [
        file
        for file in ablation_exp_dirs
        if int(file.split("/")[-2].split("EXP")[-1]) >= cfg.start_exp_num
        and int(file.split("/")[-2].split("EXP")[-1]) <= cfg.end_exp_num
    ]
    ablation_exp_dirs.sort()
    # count number of files for each EXP
    exp_num_dict = {}
    for exp_dir in ablation_exp_dirs:
        exp_num = int(exp_dir.split("/")[-2].split("EXP")[-1])
        if exp_num in exp_num_dict:
            exp_num_dict[exp_num] += 1
        else:
            exp_num_dict[exp_num] = 1
    print(exp_num_dict)
    print("-" * 100)
    print(f"Start Ablation EXP file: {ablation_exp_dirs[0]} and End Ablation EXP file: {ablation_exp_dirs[-1]}")
    print(f"Total Ablation EXP files: {len(ablation_exp_dirs)}")
    print("-" * 100)

    all_exp_dirs = base_exp_dirs + ablation_exp_dirs
    print(f"Total EXP files: {len(all_exp_dirs)}")

    all_exp_results = []

    if cfg.task_name in ["gilon_activity", "microsoft_activity_new", "SpokenArabicDigits"]:
        for exp_dir in all_exp_dirs:
            # get exp num
            exp_num = int(exp_dir.split("/")[-2].split("EXP")[-1])
            result_df = pd.read_csv(exp_dir)
            assert result_df["model_name"].iloc[0] == cfg.model_name, "Model name should be same"
            assert result_df["task_name"].iloc[0] == cfg.task_name, "Task name should be same"
            # get accuracy
            accuracy = accuracy_score(result_df["y_true"], result_df["y_pred"])
            test_loss = result_df["test_loss"].mean()

            all_exp_results.append(
                {
                    "exp_num": exp_num,
                    "accuracy": accuracy,
                    "test_loss": test_loss,
                }
            )

    else:
        raise NotImplementedError
    all_exp_df = pd.DataFrame(all_exp_results)
    # map exp_num to feature removed order
    unique_exp_num = all_exp_df["exp_num"].unique()
    unique_exp_num.sort()
    exp_num_to_order = {exp_num: i for i, exp_num in enumerate(unique_exp_num)}
    all_exp_df["feature_order"] = all_exp_df["exp_num"].map(exp_num_to_order)
    # map exp_type
    all_exp_df["exp_type"] = cfg.exp_type
    all_exp_df["model_name"] = cfg.model_name
    all_exp_df["base_exp_num"] = cfg.base_exp_num
    all_exp_df["opl_loss"] = cfg.opl_loss

    # Read importance_order_files
    importance_order_file = (
        f"importance_order_files/{cfg.task_name}/{cfg.model_name}/{cfg.out_file_name_base}-{cfg.exp_type}.pkl"
    )

    with open(importance_order_file, "rb") as f:
        importance_order = pickle.load(f)
    print(importance_order)
    all_exp_df["feature_num"] = str([])
    # map importance_order
    for i, _ in enumerate(importance_order):
        all_exp_df.loc[all_exp_df["feature_order"] == i + 1, "feature_num"] = str(importance_order[i])

    print(all_exp_df.head(15))
    # save
    if not os.path.exists(f"importance_result_files/{cfg.task_name}/{cfg.model_name}"):
        os.makedirs(f"importance_result_files/{cfg.task_name}/{cfg.model_name}")

    all_exp_df.to_csv(
        f"importance_result_files/{cfg.task_name}/{cfg.model_name}/{cfg.out_file_name_base}_{cfg.exp_type}_feature_order.csv",
        index=False,
    )
    print(
        f"Saved to: importance_result_files/{cfg.task_name}/{cfg.model_name}/{cfg.out_file_name_base}_{cfg.exp_type}_feature_order.csv"
    )


if __name__ == "__main__":
    main()
