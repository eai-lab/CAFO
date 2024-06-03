import os
import sys
import numpy as np
import pickle
import datetime
import glob
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".", config_name="create_rank")
def main(cfg: DictConfig) -> None:
    # check cfg file
    if cfg.model_name == -1:
        assert cfg.model_name != -1, "Please specify model name"
    if cfg.task_name == -1:
        assert cfg.task_name != -1, "Please specify task name"
    if cfg.base_exp_num == -1:
        assert cfg.base_exp_num != -1, "Please specify base exp num"

    # get base_exp_path
    base_exp_path = os.path.join(f"outputs/{cfg.task_name}/EXP{cfg.base_exp_num}")
    channel_attention_files = glob.glob(f"{base_exp_path}/test_*/channel_test_attention.npy")
    channel_attention_files.sort()
    print(channel_attention_files)
    # assert len(channel_attention_files) == 5, "There should be 5 test files"

    attention_npys = []
    for channel_attention_file in channel_attention_files:
        test_attention = np.load(channel_attention_file).squeeze().mean(axis=0)
        attention_npys.append(test_attention)
    # make sure all test files have same number of channels
    assert len(set([len(attention_npy) for attention_npy in attention_npys])) == 1, "All test files should have same number of channels"
    # get average attention
    mean_test_attention = np.array(attention_npys).mean(axis=0)

    # get list of channel importance idx
    channel_importance_idx = list(np.argsort(mean_test_attention)[::-1])

    important_feature_dict = {}
    unimportant_feature_dict = {}
    random_feature_dict = {}

    # get important feature
    important_channel_idx = []
    for idx, feature_idx in enumerate(channel_importance_idx):
        important_channel_idx.append(int(feature_idx))
        important_feature_dict[idx] = important_channel_idx.copy()

    # get unimportant feature
    unimportant_channel_idx = []
    for idx, feature_idx in enumerate(channel_importance_idx[::-1]):
        unimportant_channel_idx.append(int(feature_idx))
        unimportant_feature_dict[idx] = unimportant_channel_idx.copy()

    # get random feature
    np.random.seed(42)
    random_channel_list = list(np.random.permutation(channel_importance_idx))
    random_channel_idx = []
    for idx, feature_idx in enumerate(random_channel_list):
        random_channel_idx.append(int(feature_idx))
        random_feature_dict[idx] = random_channel_idx.copy()

    print("Important feature dict")
    print(important_feature_dict)

    print("Unimportant feature dict")
    print(unimportant_feature_dict)

    print("Random feature dict")
    print(random_feature_dict)

    # save
    save_dir = os.path.join(f"importance_order_files/{cfg.task_name}/{cfg.model_name}/")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save important feature dict
    important_feature_dict_path = os.path.join(save_dir, f"{cfg.out_file_name_base}-important.pkl")
    with open(important_feature_dict_path, "wb") as f:
        pickle.dump(important_feature_dict, f)

    # save unimportant feature dict
    unimportant_feature_dict_path = os.path.join(save_dir, f"{cfg.out_file_name_base}-unimportant.pkl")
    with open(unimportant_feature_dict_path, "wb") as f:
        pickle.dump(unimportant_feature_dict, f)

    # save random feature dict
    random_feature_dict_path = os.path.join(save_dir, f"{cfg.out_file_name_base}-random.pkl")
    with open(random_feature_dict_path, "wb") as f:
        pickle.dump(random_feature_dict, f)

    print(f"Save important feature dict to {important_feature_dict_path}")


if __name__ == "__main__":
    main()
