import numpy as np
import sys
import pandas as pd
import os
from random import random
from random import randint
import argparse
import matplotlib.pyplot as plt
import cv2

import neptune.new as neptune


class SquidGameSyntheticData:
    def __init__(self, config, mu=0, sigma=1):
        """
        Args:
            config: config file
            mu: mean of gaussian distribution
            sigma: standard deviation of gaussian distribution
        """

        assert config.num_samples_per_class % 2 == 0, "num_samples_per_class should be divisible by 2"
        assert config.feature_dim % 3 == 0, "feature_dim must be divisible by 3"
        assert config.feature_dim >= 15, "feature_dim should be at least 15"
        assert config.time_dim >= 20, "recommend having time points with at least 20"
        self.config = config
        self.num_samples_per_class = config.num_samples_per_class
        self.feature_dim = config.feature_dim
        self.shape_max_dim = int(config.feature_dim / 3)
        self.time_dim = config.time_dim
        self.dataset = []

        # Hyperparameters
        self.max_length_rectange = config.max_length_rectange
        self.max_length_triangle = config.max_length_triangle  # From top y coordinate to low y coordinate
        self.min_radius_circle = config.min_radius_circle
        self.mu, self.sigma = mu, sigma
        self.GLOBAL_ID = 0
        self.GLBOAL_ID_LABEL_DICT = {}

    def generate_samples(self):
        print("Generating Samples")
        class1_samples = self._generate_class1()
        class2_samples = self._generate_class2()
        class3_samples = self._generate_class3()
        print("Finished Generating Samples")
        print(f"Total Global ID :{self.GLOBAL_ID}")

        class1_df = pd.concat(class1_samples)
        class2_df = pd.concat(class2_samples)
        class3_df = pd.concat(class3_samples)

        self.dataset = pd.concat([class1_df, class2_df, class3_df])
        self.dataset = self.dataset.sample(frac=1).reset_index(drop=True)

        # Create Label
        self.label_file = pd.DataFrame.from_dict(self.GLBOAL_ID_LABEL_DICT).T
        self.label_file.reset_index(drop=False, inplace=True)
        self.label_file.columns = [
            "GLOBAL_ID",
            "circle_radius",
            "triangle_base_len",
            "rectangle_height",
            "rectangle_width",
            "class",
        ]

    def split_train_test(self, train_ratio=0.8):
        """Split dataset into train and test"""
        train_ids = np.random.choice(self.label_file.GLOBAL_ID, int(len(self.label_file) * train_ratio), replace=False)
        test_ids = np.setdiff1d(np.arange(len(self.dataset)), train_ids)

        self.train_df = self.dataset.loc[self.dataset.GLOBAL_ID.isin(train_ids)]
        self.test_df = self.dataset.loc[self.dataset.GLOBAL_ID.isin(test_ids)]

        train_label_df = self.label_file.loc[self.label_file.GLOBAL_ID.isin(train_ids)]
        test_label_df = self.label_file.loc[self.label_file.GLOBAL_ID.isin(test_ids)]
        train_label_df = train_label_df.reset_index(drop=True)
        self.train_label_df = self.apply_cv_split(train_label_df)
        self.test_label_df = test_label_df.reset_index(drop=True)

    def save_dataset(self, path):
        self.train_label_df.to_csv(os.path.join(path, "train_label.csv"), index=False)
        self.test_label_df.to_csv(os.path.join(path, "test_label.csv"), index=False)

        self.train_df.to_csv(os.path.join(path, "train_data.csv"), index=False)
        self.test_df.to_csv(os.path.join(path, "test_data.csv"), index=False)
        print(f"saved dataset to {path}")

    def save_config_info(self, path):
        with open(os.path.join(path, "config.txt"), "w") as f:
            f.write(f"num_samples_per_class:{self.num_samples_per_class}\n")
            f.write(f"feature_dim:{self.feature_dim}\n")
            f.write(f"time_dim:{self.time_dim}\n")
            f.write(f"max_length_rectange:{self.max_length_rectange}\n")
            f.write(f"max_length_triangle:{self.max_length_triangle}\n")
            f.write(f"min_radius_circle:{self.min_radius_circle}\n")
            f.write(f"mu:{self.mu}\n")
            f.write(f"sigma:{self.sigma}\n")
        print(f"saved config file to {path}")

    def apply_cv_split(self, label_file, cv_split=5):
        """Apply cross validation split to label file"""
        label_file["cv_split"] = np.random.randint(0, cv_split, len(label_file))
        label_file.reset_index(drop=True, inplace=True)
        label_file.sort_values(by=["cv_split", "GLOBAL_ID"], inplace=True)
        return label_file

    def _generate_class1(self):
        """Generate class 1 samples"""
        full_feature_mask = np.ones((self.time_dim, self.shape_max_dim))
        empty_mask = np.zeros((self.time_dim, self.shape_max_dim))
        class1_samples = []
        for i in range(int(self.num_samples_per_class / 2)):
            if i % 100 == 0:
                print(f"Finished Generating sample_num:{i}")
            for j in range(2):
                if j == 0:
                    # Inside mask
                    circle_mask, radius = self.generate_circle_mask()
                    triangle_mask, base_len = self.generate_triangle_mask()

                    # Inside Feature
                    circle_feat = self.make_circle_feature()
                    triangle_feat = self.make_triangle_feature()
                    # outside mask
                    outside_circle_mask = full_feature_mask - circle_mask
                    outside_triangle_mask = full_feature_mask - triangle_mask

                    inside_feat = np.concatenate([circle_feat, triangle_feat, empty_mask], axis=1)
                    inside_mask = np.concatenate([circle_mask, triangle_mask, empty_mask], axis=1)

                    # Generate White Noise
                    outside_feat = np.random.normal(self.mu, self.sigma, size=(self.time_dim, self.feature_dim))
                    outside_mask = np.concatenate(
                        [outside_circle_mask, outside_triangle_mask, full_feature_mask], axis=1
                    )

                    sample = pd.DataFrame(outside_feat * outside_mask + inside_feat * inside_mask)
                    self.GLBOAL_ID_LABEL_DICT[self.GLOBAL_ID] = {
                        "circle_radius": radius,
                        "triangle_base_len": base_len,
                        "height": 0,
                        "width": 0,
                        "label": 0,
                    }

                else:
                    # Inside mask
                    circle_mask, radius = self.generate_circle_mask()
                    circle_feat = self.make_circle_feature()
                    outside_circle_mask = full_feature_mask - circle_mask

                    inside_feat = np.concatenate([circle_feat, empty_mask, empty_mask], axis=1)
                    inside_mask = np.concatenate([circle_mask, empty_mask, empty_mask], axis=1)

                    # Generate White Noise
                    outside_feat = np.random.normal(self.mu, self.sigma, size=(self.time_dim, self.feature_dim))
                    outside_mask = np.concatenate([outside_circle_mask, full_feature_mask, full_feature_mask], axis=1)

                    sample = pd.DataFrame(outside_feat * outside_mask + inside_feat * inside_mask)
                    self.GLBOAL_ID_LABEL_DICT[self.GLOBAL_ID] = {
                        "circle_radius": radius,
                        "triangle_base_len": 0,
                        "height": 0,
                        "width": 0,
                        "label": 0,
                    }
                sample["GLOBAL_ID"] = self.GLOBAL_ID
                class1_samples.append(sample)
                self.GLOBAL_ID += 1
        return class1_samples

    def _generate_class2(self):
        """generate class 2 samples"""
        full_feature_mask = np.ones((self.time_dim, self.shape_max_dim))
        empty_mask = np.zeros((self.time_dim, self.shape_max_dim))
        class2_samples = []
        for i in range(int(self.num_samples_per_class / 2)):
            if i % 100 == 0:
                print(f"Finished Generating sample_num:{i}")
            for j in range(2):
                if j == 0:
                    # Inside mask
                    triangle_mask, base_len = self.generate_triangle_mask()
                    rectange_mask, height, width = self.generate_rectangle_mask()

                    # Inside Feature
                    triangle_feat = self.make_triangle_feature()
                    rectangle_feat = self.make_rectangle_feature()

                    # outside mask
                    outside_triangle_mask = full_feature_mask - triangle_mask
                    outside_rectangle_mask = full_feature_mask - rectange_mask

                    inside_feat = np.concatenate([empty_mask, triangle_feat, rectangle_feat], axis=1)
                    inside_mask = np.concatenate([empty_mask, triangle_mask, rectange_mask], axis=1)

                    # Generate White Noise
                    outside_feat = np.random.normal(self.mu, self.sigma, size=(self.time_dim, self.feature_dim))
                    outside_mask = np.concatenate(
                        [full_feature_mask, outside_triangle_mask, outside_rectangle_mask], axis=1
                    )

                    sample = pd.DataFrame(outside_feat * outside_mask + inside_feat * inside_mask)
                    self.GLBOAL_ID_LABEL_DICT[self.GLOBAL_ID] = {
                        "circle_radius": 0,
                        "triangle_base_len": base_len,
                        "height": height,
                        "width": width,
                        "label": 1,
                    }

                else:
                    # Inside mask
                    triangle_mask, base_len = self.generate_triangle_mask()
                    triangle_feat = self.make_triangle_feature()
                    outside_triangle_mask = full_feature_mask - triangle_mask

                    inside_feat = np.concatenate([empty_mask, triangle_feat, empty_mask], axis=1)
                    inside_mask = np.concatenate([empty_mask, triangle_mask, empty_mask], axis=1)

                    # Generate White Noise
                    outside_feat = np.random.normal(self.mu, self.sigma, size=(self.time_dim, self.feature_dim))
                    outside_mask = np.concatenate(
                        [full_feature_mask, outside_triangle_mask, full_feature_mask], axis=1
                    )

                    sample = pd.DataFrame(outside_feat * outside_mask + inside_feat * inside_mask)
                    self.GLBOAL_ID_LABEL_DICT[self.GLOBAL_ID] = {
                        "circle_radius": 0,
                        "triangle_base_len": base_len,
                        "height": 0,
                        "width": 0,
                        "label": 1,
                    }

                sample["GLOBAL_ID"] = self.GLOBAL_ID
                class2_samples.append(sample)
                self.GLOBAL_ID += 1
        return class2_samples

    def _generate_class3(self):
        """generate class 3 samples"""
        full_feature_mask = np.ones((self.time_dim, self.shape_max_dim))
        empty_mask = np.zeros((self.time_dim, self.shape_max_dim))
        class3_samples = []
        for i in range(int(self.num_samples_per_class / 2)):
            if i % 100 == 0:
                print(f"Finished Generating sample_num:{i}")
            for j in range(2):
                if j == 0:
                    # Inside mask
                    circle_mask, radius = self.generate_circle_mask()
                    rectangle_mask, height, width = self.generate_rectangle_mask()

                    # Inside Feature
                    circle_feat = self.make_circle_feature()
                    rectangle_feat = self.make_rectangle_feature()

                    # outside mask
                    outside_circle_mask = full_feature_mask - circle_mask
                    outside_rectangle_mask = full_feature_mask - rectangle_mask

                    inside_feat = np.concatenate([circle_feat, empty_mask, rectangle_feat], axis=1)
                    inside_mask = np.concatenate([circle_mask, empty_mask, rectangle_mask], axis=1)

                    # Generate White Noise
                    outside_feat = np.random.normal(self.mu, self.sigma, size=(self.time_dim, self.feature_dim))
                    outside_mask = np.concatenate(
                        [outside_circle_mask, full_feature_mask, outside_rectangle_mask], axis=1
                    )

                    sample = pd.DataFrame(outside_feat * outside_mask + inside_feat * inside_mask)
                    self.GLBOAL_ID_LABEL_DICT[self.GLOBAL_ID] = {
                        "circle_radius": radius,
                        "triangle_base_len": 0,
                        "height": height,
                        "width": width,
                        "label": 2,
                    }
                else:
                    # Inside mask
                    rectangle_mask, height, width = self.generate_rectangle_mask()
                    rectangle_feat = self.make_rectangle_feature()

                    # outside mask
                    outside_rectangle_mask = full_feature_mask - rectangle_mask

                    inside_feat = np.concatenate([empty_mask, empty_mask, rectangle_feat], axis=1)
                    inside_mask = np.concatenate([empty_mask, empty_mask, rectangle_mask], axis=1)

                    # Generate White Noise
                    outside_feat = np.random.normal(self.mu, self.sigma, size=(self.time_dim, self.feature_dim))
                    outside_mask = np.concatenate(
                        [full_feature_mask, full_feature_mask, outside_rectangle_mask], axis=1
                    )

                    sample = pd.DataFrame(outside_feat * outside_mask + inside_feat * inside_mask)
                    self.GLBOAL_ID_LABEL_DICT[self.GLOBAL_ID] = {
                        "circle_radius": 0,
                        "triangle_base_len": 0,
                        "height": height,
                        "width": width,
                        "label": 2,
                    }

                sample["GLOBAL_ID"] = self.GLOBAL_ID
                class3_samples.append(sample)
                self.GLOBAL_ID += 1
        return class3_samples

    def generate_circle_mask(self):
        """Generate a circle mask with minimum radius of 2"""
        center_coord_x = randint(self.min_radius_circle, self.shape_max_dim - self.min_radius_circle - 1)
        center_coord_y = randint(self.min_radius_circle, self.time_dim - self.min_radius_circle - 1)
        # print(center_coord_x, self.shape_max_dim-center_coord_x, center_coord_y, self.time_dim-center_coord_y)
        radius = min(
            center_coord_x, self.shape_max_dim - center_coord_x, center_coord_y, self.time_dim - center_coord_y
        )

        base = np.zeros((self.time_dim, self.shape_max_dim))
        circle = cv2.circle(base, (center_coord_x, center_coord_y), radius, color=(1, 0, 0), thickness=-1)
        # print(circle.shape)
        return circle, radius

    def generate_rectangle_mask(self):
        """Generate a rectangle mask"""
        top_left_coord_x = randint(0, self.shape_max_dim - 4)
        top_left_coord_y = randint(0, self.time_dim - 4)
        bottom_right_coord_x = randint(top_left_coord_x + 3, self.shape_max_dim - 1)
        bottom_right_coord_y = min(
            randint(top_left_coord_y + 3, self.time_dim - 1), top_left_coord_y + self.max_length_rectange
        )

        height = bottom_right_coord_y - top_left_coord_y
        width = bottom_right_coord_x - top_left_coord_x

        base = np.zeros((self.time_dim, self.shape_max_dim))
        rectange = cv2.rectangle(
            base, (top_left_coord_x, top_left_coord_y), (bottom_right_coord_x, bottom_right_coord_y), (1, 0, 0), -1
        )
        # print(rectange.shape)
        return rectange, height, width

    def generate_triangle_mask(self):
        """Generate a triangle mask"""
        top_vertex_coord_x = randint(1, self.shape_max_dim - 2)
        top_vertex_coord_y = randint(0, self.time_dim - 4)

        bottom_vertex_coord_y = min(
            randint(top_vertex_coord_y + 1, self.time_dim - 2), top_vertex_coord_y + self.max_length_triangle
        )
        base_len = min(top_vertex_coord_x, self.shape_max_dim - 1 - top_vertex_coord_x)

        top_vertex = (top_vertex_coord_x, top_vertex_coord_y)
        left_vertex = (top_vertex_coord_x - base_len, bottom_vertex_coord_y)
        right_vertex = (top_vertex_coord_x + base_len, bottom_vertex_coord_y)
        pts = np.array([top_vertex, left_vertex, right_vertex])
        base = np.zeros((self.time_dim, self.shape_max_dim))
        triangle = cv2.fillPoly(base, [pts], (1, 0, 0))
        # print(top_vertex,left_vertex,right_vertex)
        # print(triangle)
        return triangle, base_len

    def make_circle_feature(self):
        """ "create array of shape (self.time_dim, self.shape_max_dim) of circle features"""

        circle_feat = []
        for i in range(self.shape_max_dim):
            random_num = randint(0, 360)
            x = random_num + np.linspace(0, 360, self.time_dim)
            circle_feat.append(13 * np.cos(13 * x))
        circle_feats = np.stack(circle_feat, 1)

        return circle_feats

    def make_triangle_feature(self):
        """ "create array of shape (self.time_dim, self.shape_max_dim) of triangle features"""

        triangle_feat = []
        for i in range(self.shape_max_dim):

            random_num = randint(-2, 2)
            x = random_num + np.linspace(-3, 3, self.time_dim)
            # rectangle_feat.append(2*np.sin(0.3* x) + 4*np.cos(0.7* x))
            triangle_feat.append(5 * np.sin(5 * x))
        triangle_feats = np.stack(triangle_feat, 1)

        return triangle_feats

    def make_rectangle_feature(self):
        """ "create array of shape (self.time_dim, self.shape_max_dim) of rectangle features"""

        rectangle_feat = []
        for i in range(self.shape_max_dim):
            # random_num = randint(-2, 2)
            # x = random_num + np.linspace(-3, 3, self.time_dim)
            # # rectangle_feat.append(2*np.sin(0.3* x) + 4*np.cos(0.7* x))
            # rectangle_feat.append(sigmoid(x))
            random_num = randint(-2, 2)
            x = random_num + np.linspace(-3, 3, self.time_dim)
            rectangle_feat.append(sigmoid(x))
        rectangle_feats = np.stack(rectangle_feat, 1)

        return rectangle_feats

    def save_answer_sheet(self, path):
        answer_sheet = pd.DataFrame(
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            )
        )
        answer_sheet.index = ["class1", "class2", "class3"]
        answer_sheet.to_csv(os.path.join(path, "answersheet.csv"))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def print_options(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    print(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decription", type=str, default="", help="description of the dataset")
    parser.add_argument("--num_samples_per_class", type=int, default=18000, help="should be divisible by 2")
    parser.add_argument("--time_dim", type=int, default=32, help="should have at least 20 samples")
    parser.add_argument("--feature_dim", type=int, default=30, help="need to be divisible by 3")
    parser.add_argument("--min_radius_circle", type=int, default=4)
    parser.add_argument("--max_length_rectange", type=int, default=6)
    parser.add_argument("--max_length_triangle", type=int, default=6)
    config = parser.parse_args()

    print_options(config)
    dataset = SquidGameSyntheticData(config=config)
    dataset.generate_samples()
    dataset.split_train_test()

    dataset_name = (
        f"num_3classsamples_per_class{config.num_samples_per_class}_time{config.time_dim}_shape{config.feature_dim}"
    )
    config.dataset_name = dataset_name
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)
    dataset.save_dataset(dataset_name)
    dataset.save_config_info(dataset_name)
    dataset.save_answer_sheet(dataset_name)
    print(f"Dataset saved to {dataset_name}")
