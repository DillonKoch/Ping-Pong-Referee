# ==============================================================================
# File: detect_ball_model.py
# Project: src
# File Created: Monday, 15th February 2021 1:11:12 pm
# Author: Dillon Koch
# -----
# Last Modified: Monday, 15th February 2021 3:45:35 pm
# Modified By: Dillon Koch
# -----
# Collins Aerospace
#
# -----
# creates the model for detecting the ball location in videos
# ==============================================================================


from os.path import abspath, dirname
import sys
import numpy as np

from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Linear, Sigmoid, Dropout, MaxPool2d, Dropout2d
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import SGD

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


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
    def __init__(self, dropout):
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
        x = self.conv_block2(x)
        x = self.dropout_2d(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.dropout_2d(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.dropout_2d(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout_1d(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout_1d(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# arr = np.zeros((3, 27, 320, 128))
tensor = [np.zeros((2, 27, 320, 128)) for i in range(3)]
labels = [np.zeros((448)) for i in range(3)]
# labels = np.zeros((3, 448))
# labels = torch.from_numpy(labels)

data = [(tensor[i], labels[i]) for i in range(len(labels))]


def train(data, model):
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(10):
        print(epoch)
        for i, (inputs, targets) in enumerate(data):
            print(i)
            inputs = torch.from_numpy(inputs).float()
            targets = torch.from_numpy(targets).float()
            print('here')
            print(inputs.shape)
            print(targets.shape)
            optimizer.zero_grad()
            yhat = model(inputs)
            print(yhat.shape)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    model = Detect_Ball(0.5)
    train(data, model)

    # actually train on some nonsense inputs just so I know I can load data how I want
