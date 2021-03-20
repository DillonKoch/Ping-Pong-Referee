# ==============================================================================
# File: training_dataloader.py
# Project: src
# File Created: Saturday, 13th March 2021 9:25:16 am
# Author: Dillon Koch
# -----
# Last Modified: Friday, 19th March 2021 11:35:58 pm
# Modified By: Dillon Koch
# -----
#
# -----
# loading data to train models
# ==============================================================================

import json
import os
import sys
from os.path import abspath, dirname

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


class PingPongDataset(Dataset):
    def __init__(self, train=True, limit=None):
        self.train = train

        ball_frame_lists, ball_label_lists = self.prepare_dataset()
        self.ball_frame_lists = ball_frame_lists if limit is None else ball_frame_lists[:limit]
        self.ball_label_lists = ball_label_lists if limit is None else ball_label_lists[:limit]

    def load_ball_markup(self, game_path):  # Top Level
        with open(game_path + "/ball_markup.json", 'r') as f:
            ball_dict = json.load(f)
        return {key: value for key, value in ball_dict.items() if value['x'] != -1 and value['y'] != -1}

    def _frame_to_frame_list(self, game_path, frame):  # Specific Helper data_from_ball_dict
        frame = int(frame)
        new_frame_list = [game_path + f"/images/frame_{i}.png" for i in range(frame - 8, frame + 1, 1)]
        return new_frame_list

    def _label_to_dist(self, val, full_dist_len, norm_dist_len):  # Specific Helper data_from_ball_dict
        norm_vals = np.linspace(-3, 3, norm_dist_len)
        norm_dist = stats.norm.pdf(norm_vals, 0, 1)
        full_dist = np.zeros(full_dist_len + 1000)
        full_dist[int(val + 500 - (norm_dist_len / 2)):int(val + 500 + (norm_dist_len / 2))] = norm_dist
        full_dist = full_dist[500:-500]
        return full_dist

    def data_from_ball_dict(self, game_path, ball_dict):  # Top Level
        ball_frame_lists = []
        ball_labels = []
        for frame, label in ball_dict.items():
            ball_frame_lists.append(self._frame_to_frame_list(game_path, frame))

            x = int(label['x'] * (320 / 1920))
            y = int(label['y'] * (128 / 1080))
            x_label = self._label_to_dist(x, 320, 200)
            y_label = self._label_to_dist(y, 128, 80)
            full_label = np.append(x_label, y_label)
            ball_labels.append(full_label)
        return ball_frame_lists, ball_labels

    def prepare_dataset(self):  # Run
        train_test_str = "Train" if self.train else "Test"
        game_paths = listdir_fullpath(ROOT_PATH + f"/Data/{train_test_str}/")
        ball_frame_lists = []
        ball_label_lists = []
        for game_path in tqdm(game_paths):
            ball_dict = self.load_ball_markup(game_path)
            new_frame_lists, new_label_lists = self.data_from_ball_dict(game_path, ball_dict)
            ball_frame_lists += new_frame_lists
            ball_label_lists += new_label_lists
        return ball_frame_lists, ball_label_lists

    def __getitem__(self, index):
        frame_paths = self.ball_frame_lists[index]
        frames = [cv2.imread(path) for path in frame_paths]
        frames = [cv2.resize(frame, (320, 128)) for frame in frames]
        frames = np.dstack(frames)
        frames = frames.transpose(2, 0, 1)
        frames = frames.reshape((1, 27, 128, 320))
        frames = torch.from_numpy(frames)
        frames = frames.float()
        frames /= 255.
        labels = self.ball_label_lists[index]
        labels = labels.reshape((1, 448))
        labels = torch.from_numpy(labels)
        labels = labels.float()
        return frames, labels

    def __len__(self):
        return len(self.ball_frame_lists)


if __name__ == '__main__':
    dataset = PingPongDataset()
    self = dataset
    x, y = dataset[1000]
