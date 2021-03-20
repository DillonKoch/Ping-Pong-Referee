# ==============================================================================
# File: event_detection.py
# Project: src
# File Created: Friday, 19th March 2021 11:26:05 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 20th March 2021 12:53:52 am
# Modified By: Dillon Koch
# -----
#
# -----
# Detecting events from video
# ==============================================================================


import torch
import torch.nn as nn

from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from training_dataloader import PingPongDataset


class Events_Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Events_Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class Detect_Events(nn.Module):
    def __init__(self, dropout_p):
        super(Detect_Events, self).__init__()
        self.conv1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout2d = nn.Dropout2d(p=dropout_p)
        self.convblock = Events_Conv_Block(in_channels=64, out_channels=64)
        self.fc1 = nn.Linear(in_features=640, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, global_features, local_features):
        event_input = torch.cat((global_features, local_features), dim=1)
        x = self.conv1(event_input)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.dropout2d(x)
        x = self.convblock(x)
        x = self.dropout2d(x)
        x = self.convblock(x)
        x = self.dropout2d(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Event_Detection_Loss(nn.Module):
    def __init__(self, weights=(1, 3), num_events=2, epsilon=1e-9):
        super(Event_Detection_Loss, self).__init__()
        weights = torch.tensor(weights).view(1, 2)
        self.weights = weights / weights.sum()
        self.num_events = num_events
        self.epsilon = epsilon

    def forward(self, pred_events, target_events):
        self.weights = self.weights.cuda()
        return - torch.mean(self.weights * (target_events * torch.log(pred_events + self.epsilon) + (1. - target_events)
                                            * torch.log(1 - pred_events + self.epsilon)))


def train(model, epochs=100):  # Run
    dataset = PingPongDataset(limit=None)
    dataset_len = len(dataset)
    criterion = Event_Detection_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model_path = ROOT_PATH + "/src/event_detection.pth"

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
    event_detection = Detect_Events()
    self = event_detection
    train(event_detection, epochs=150)
