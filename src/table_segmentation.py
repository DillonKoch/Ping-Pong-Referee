# ==============================================================================
# File: table_segmentation.py
# Project: src
# File Created: Friday, 19th March 2021 11:24:05 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 20th March 2021 12:54:01 am
# Modified By: Dillon Koch
# -----
#
# -----
# Segmenting the table from the video
# ==============================================================================


import sys
from os.path import abspath, dirname

import torch
import torch.nn as nn

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from training_dataloader import PingPongDataset


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeconvBlock, self).__init__()
        middle_channels = int(in_channels / 4)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=1, stride=1, padding=0)
        self.batchnorm1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU()
        self.batchnorm_tconv = nn.BatchNorm2d(middle_channels)
        self.tconv = nn.ConvTranspose2d(middle_channels, middle_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.tconv(x)
        x = self.batchnorm_tconv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        return x


class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.deconvblock5 = DeconvBlock(in_channels=256, out_channels=128)
        self.deconvblock4 = DeconvBlock(in_channels=128, out_channels=128)
        self.deconvblock3 = DeconvBlock(in_channels=128, out_channels=64)
        self.deconvblock2 = DeconvBlock(in_channels=64, out_channels=64)
        self.tconv = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0, output_padding=0)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=2, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, out_block2, out_block3, out_block4, out_block5):
        x = self.deconvblock5(out_block5)
        x = x + out_block4
        x = self.deconvblock4(x)
        x = x + out_block3
        x = self.deconvblock3(x)
        x = x + out_block2
        x = self.deconvblock2(x)
        x = self.relu(self.tconv(x))
        x = self.relu(self.conv1(x))
        out = self.sigmoid(self.conv2(x))
        return out


class DICE_Smotth_Loss(nn.Module):
    def __init__(self, epsilon=1e-9):
        super(DICE_Smotth_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred_seg, target_seg):
        return 1. - ((torch.sum(2 * pred_seg * target_seg) + self.epsilon) /
                     (torch.sum(pred_seg) + torch.sum(target_seg) + self.epsilon))


class BCE_Loss(nn.Module):
    def __init__(self, epsilon=1e-9):
        super(BCE_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred_seg, target_seg):
        return - torch.mean(target_seg * torch.log(pred_seg + self.epsilon) + (1 - target_seg)
                            * torch.log(1 - pred_seg + self.epsilon))


class Segmentation_Loss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(Segmentation_Loss, self).__init__()
        self.bce_criterion = BCE_Loss(epsilon=1e-9)
        self.dice_criterion = DICE_Smotth_Loss(epsilon=1e-9)
        self.bce_weight = bce_weight

    def forward(self, pred_seg, target_seg):
        target_seg = target_seg.float()
        loss_bce = self.bce_criterion(pred_seg, target_seg)
        loss_dice = self.dice_criterion(pred_seg, target_seg)
        loss_seg = (1 - self.bce_weight) * loss_dice + self.bce_weight * loss_bce
        return loss_seg


def train(model, epochs=100):  # Run
    dataset = PingPongDataset(limit=None)
    dataset_len = len(dataset)
    criterion = Segmentation_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model_path = ROOT_PATH + "/src/segmentation.pth"

    for epoch in range(epochs):
        for i, (frames, labels) in enumerate(dataset):
            print(f"epoch {epoch} ({i}/{dataset_len})")

            event_pred, *_ = model(frames)
            loss = criterion(event_pred.float(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss: {loss.item()}")

            if i % 1 == 0:
                torch.save(model, model_path)
                print('model saved!')


if __name__ == '__main__':
    segmentation = Segmentation()
    self = segmentation
    train(segmentation, epochs=200)
