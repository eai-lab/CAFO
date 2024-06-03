import pandas as pd
import numpy as np
import glob
import os
from abc import ABC, abstractmethod


class BaseVisualizer(ABC):
    """This class is an abstract base class in visualizing the importance of the features
    To create the subclass, you need to implement the following methods:
    - _load_feature_lists: load the feature lists, define them as constants
    - _define_class_map: define the class map, this is used to map the class number to the class name
    - plot_global_attention: plot the global attention
    - plot_classwise_attention: plot the classwise attention
    """

    def __init__(self, cfg):
        self.cfg = cfg
        # check if directory to save files exists
        if not os.path.exists(self.cfg.save_classwise_attention_path):
            os.makedirs(self.cfg.save_classwise_attention_path)
        if not os.path.exists(self.cfg.save_global_attention_path):
            os.makedirs(self.cfg.save_global_attention_path)
        self.classwise_attention_path = (
            f"{self.cfg.save_classwise_attention_path}/{self.cfg.exp_num}_cv{self.cfg.task.validation_cv_num}_classwise_attention.png"
        )
        self.global_attention_path = f"{self.cfg.save_global_attention_path}/{self.cfg.exp_num}_cv{self.cfg.task.validation_cv_num}_global_attention.png"
        self.feature_lists = self._load_feature_lists()
        # loading train results
        self.last_ca = self.load_last_CA()
        self.attention_label = self.load_attention_label_csv()
        # loading test results
        self.test_attention_label = self.load_test_attention_scores_and_label_csv()
        # define the class map
        self.unique_classes = np.unique(self.attention_label["y_true"])
        self.class_map = self._define_class_map()
        self.class_average_importance = self.calculate_class_average_importance()
        self.global_importance = self.calculate_global_importance()
        self.test_global_importance = np.array(np.mean(self.test_attention_label.iloc[:, :-1]))
        self.class_wise_relative_importance = self.calculate_class_wise_relative_importance()
        self.test_class_wise_relative_importance = self.test_calculate_class_wise_relative_importance()

    def load_last_CA(self):
        """Load the last channel attention npy file at training"""
        ca_files = glob.glob(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/channel_attention_*.npy")
        ca_files.sort()
        last_ca = np.load(ca_files[-1]).squeeze()
        return last_ca

    def load_attention_label_csv(self):
        # Load the attention label csv file
        attention_label = pd.read_csv(f"{self.cfg.save_output_path}/cv{self.cfg.task.validation_cv_num}/attention_label.csv")
        return attention_label

    def load_test_attention_scores_and_label_csv(self):
        # Load the test attention label csv file
        test_attention_label = pd.read_csv(f"{self.cfg.save_output_path}/test_{self.cfg.task.validation_cv_num}/channel_test_attention_scores.csv")
        return test_attention_label

    def calculate_class_average_importance(self):
        """
        Calculate the class average importance, this is different from the class-wise relative importance
        """
        class_average_importance = np.zeros((len(self.unique_classes), self.cfg.task.in_channels))
        for i, class_ in enumerate(self.unique_classes):
            class_average_importance[i] = np.mean(self.last_ca[self.attention_label["y_true"] == class_], axis=0)
        return class_average_importance

    def calculate_global_importance(self):
        """Calculate the global importance for train data"""
        global_importance = np.mean(self.last_ca, axis=0)
        return global_importance

    def calculate_class_wise_relative_importance(self):
        """Calculate the class-wise relative importance for train data"""
        class_wise_relative_importance = np.zeros((len(self.unique_classes), self.cfg.task.in_channels))
        for i, class_ in enumerate(self.unique_classes):
            # calculate the average of the class_average_importance without the current class
            rest_class_average_importance = np.delete(self.class_average_importance, i, axis=0)
            rest_average = np.mean(rest_class_average_importance, axis=0)
            # calculate the relative importance
            class_wise_relative_importance[i] = self.class_average_importance[i] - rest_average
        return class_wise_relative_importance

    def test_calculate_class_wise_relative_importance(self):
        """Calculate the class-wise relative importance for test data"""
        class_wise_relative_importance = np.zeros((len(self.unique_classes), self.cfg.task.in_channels))
        for i, class_ in enumerate(self.unique_classes):
            # calculate the average of the rest
            rest_average = np.mean(self.test_attention_label.iloc[:, :-1][self.test_attention_label["y_true"] != class_], axis=0)
            # calculate the relative importance
            class_wise_relative_importance[i] = self.class_average_importance[i] - rest_average
        return class_wise_relative_importance

    def find_decimal_factor(self, number):
        # if number is really small return 100
        if number < 0.00001:
            return 100
        multiplier = 10
        while abs(number) <= 0.1:
            multiplier *= 10
            number *= 10
        return multiplier

    @abstractmethod
    def _load_feature_lists(self):
        pass

    @abstractmethod
    def _define_class_map(self):
        # implement in subclass
        pass

    @abstractmethod
    def plot_global_attention(self):
        # implement in subclass
        pass

    @abstractmethod
    def plot_classwise_attention(self):
        # implement in subclass
        pass
