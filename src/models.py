# ==============================================================================
# File: models.py
# Project: src
# File Created: Saturday, 6th March 2021 6:09:05 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 6th March 2021 7:03:04 pm
# Modified By: Dillon Koch
# -----
# Collins Aerospace
#
# -----
# creating models for TTNet
# ==============================================================================

from os.path import abspath, dirname
import sys
import matplotlib.pyplot as plt
import numpy as np

from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Linear, Sigmoid, Dropout, MaxPool2d, Dropout2d
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import SGD

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from load_data import Load_Video


class Conv_Block(Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_Block, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Detect_Ball(Module):
    def __init__(self, dropout=0.2):
        super(Detect_Ball, self).__init__()
        self.conv1 = Conv2d(27, 64, kernel_size=1, stride=1, padding=0)
        self.batch_norm = BatchNorm2d(64)
        self.relu = ReLU()
        self.conv_block1 = Conv_Block(in_channels=64, out_channels=64)
        self.conv_block2 = Conv_Block(in_channels=64, out_channels=64)
        self.dropout_2d = Dropout2d(p=dropout)
        self.conv_block3 = Conv_Block(in_channels=64, out_channels=128)
        self.conv_block4 = Conv_Block(in_channels=128, out_channels=128)
        self.conv_block5 = Conv_Block(in_channels=128, out_channels=256)
        self.conv_block6 = Conv_Block(in_channels=256, out_channels=256)
        self.fc1 = Linear(in_features=2560, out_features=1792)
        self.fc2 = Linear(in_features=1792, out_features=896)
        self.fc3 = Linear(in_features=896, out_features=448)
        self.dropout_1d = Dropout(p=dropout)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # TODO may want to return more than just the final output!
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv_block1(x)
        out_block2 = self.conv_block2(x)
        x = self.dropout_2d(out_block2)
        out_block3 = self.conv_block3(x)
        out_block4 = self.conv_block4(out_block3)
        x = self.dropout_2d(out_block4)
        out_block5 = self.conv_block5(x)
        features = self.conv_block6(out_block5)
        x = self.dropout_2d(features)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_1d(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout_1d(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x, features, out_block2, out_block3, out_block4, out_block5


if __name__ == '__main__':
    ball_global = Detect_Ball()
    ball_local = Detect_Ball()
    lv = Load_Video(ROOT_PATH + "/Data/Test/Game1/gameplay.mp4")
    vid_gen = lv.run(resize=True)
    arr = next(vid_gen)
    arr = arr.reshape((1, 27, 128, 320))
    arr = torch.from_numpy(arr)
    arr = arr.float()
    x, features, out_block2, out_block3, out_block4, out_block5 = ball_global(arr)
    pred_ball_local, local_features, *_ = ball_local(arr)
