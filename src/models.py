# ==============================================================================
# File: models.py
# Project: src
# File Created: Saturday, 6th March 2021 6:09:05 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 13th March 2021 8:56:42 pm
# Modified By: Dillon Koch
# -----
#
# -----
# creating models for TTNet
# ==============================================================================

import sys
from os.path import abspath, dirname

import torch
import torch.nn as nn
from torch.nn import (BatchNorm2d, Conv2d, Dropout, Dropout2d, Linear,
                      MaxPool2d, Module, ReLU, Sigmoid)

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from training_dataloader import PingPongDataset


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


class Ball_Detection_Loss(nn.Module):
    def __init__(self):
        super(Ball_Detection_Loss, self).__init__()
        self.epsilon = 1e-9

    def forward(self, pred_ball_position, target_ball_position):
        x_pred = pred_ball_position[:, :320]
        y_pred = pred_ball_position[:, 320:]

        x_target = target_ball_position[:, :320]
        y_target = target_ball_position[:, 320:]

        loss_ball_x = - torch.mean(x_target * torch.log(x_pred + self.epsilon) + (1 - x_target)
                                   * torch.log(1 - x_pred + self.epsilon))
        loss_ball_y = - torch.mean(y_target * torch.log(y_pred + self.epsilon) + (1 - y_target)
                                   * torch.log(1 - y_pred + self.epsilon))

        return loss_ball_x + loss_ball_y


def train(model, epochs=100):  # Run
    dataset = PingPongDataset(limit=None)
    dataset_len = len(dataset)
    criterion = Ball_Detection_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model_path = ROOT_PATH + "/src/ball_detection_global.pth"

    for epoch in range(epochs):
        for i, (frames, labels) in enumerate(dataset):
            print(f"epoch {epoch} ({i}/{dataset_len})")

            ball_pred, *_ = model(frames)
            loss = criterion(ball_pred.float(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")

            # finding predicted (x,y)
            x_pred = torch.argmax(ball_pred[0, :320])
            x_label = torch.argmax(labels[0, :320])
            y_pred = torch.argmax(ball_pred[0, 320:])
            y_label = torch.argmax(labels[0, 320:])
            print(f"X pred: {x_pred}, X label: {x_label}")
            print(f"Y pred: {y_pred}, Y label: {y_label}")

            if i % 1 == 0:
                torch.save(model, model_path)
                print('model saved!')


if __name__ == '__main__':
    ball_global = Detect_Ball()
    # ball_global = torch.load(model_path)
    train(ball_global)
