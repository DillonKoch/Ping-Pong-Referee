# ==============================================================================
# File: detect_ball_model.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:08:57 am
# Author: Dillon Koch
# -----
# Last Modified: Monday, 15th February 2021 1:09:58 pm
# Modified By: Dillon Koch
# -----
#
# -----
# Trains a model to detect the ball in a stack of 9 frames of ping pong video
# ==============================================================================


import json
import os
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Detect_Ball_Model:
    def __init__(self):
        self.train_paths = [ROOT_PATH + f"/Data/Train/Game{i+1}/gameplay.mp4" for i in range(5)]
        self.test_paths = [ROOT_PATH + f"/Data/Test/Game{i+1}/gameplay.mp4" for i in range(7)]

    def _remove_negative_positions(self, labels):
        new_labels = {frame: pos for frame, pos in labels.items() if ((pos['x'] != -1) and (pos['y'] != -1))}
        return new_labels

    def load_labels(self, game, train=True):  # Top Level
        """
        loads the ball_markup.json file for a given video that contains the
        xy positions of the ball in the given frame
        """
        train_test_str = "Train" if train else "Test"
        path = ROOT_PATH + f"/Data/{train_test_str}/Game{game}/ball_markup.json"
        assert os.path.exists(path)
        with open(path, 'r') as f:
            labels = json.load(f)
        labels = self._remove_negative_positions(labels)
        frames = list(labels.keys())
        frames = [frame for frame in frames if int(frame) > 10]
        positions = list(labels.values())
        return frames, positions

    def _downsample_frame(self, frame):  # Specific Helper
        """
        downsamples an image from the original 1920x1080 to 320x128
        """
        new_frame = cv2.resize(frame, dsize=(320, 128))
        return new_frame

    def load_frame_stacks_generator(self, vid_path, frames):  # Top Level
        """
        loads a generator of the frame stacks for a given video
        """
        cap = cv2.VideoCapture(vid_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        stack = [cap.read()[1] for i in range(8)]
        stack = [self._downsample_frame(frame) for frame in stack]
        for i in range(8, frame_count, 1):
            _, new_frame = cap.read()
            new_frame = self._downsample_frame(new_frame)
            stack.append(new_frame)
            if str(i) in frames:
                yield np.concatenate(stack, axis=0)
            stack = stack[1:]

    # def _conv_block(self):  # Specific Helper
    #     return [
    #         keras.layers.Conv2D(64, (1, 1), strides=(1, 1)),
    #         keras.layers.BatchNormalization(),
    #         keras.layers.ReLU(),
    #         keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
    #     ]

    # def create_global_model(self):  # Top Level
    #     model = keras.Sequential([
    #         keras.layers.Conv2D(64, (1, 1), strides=(1, 1), input_shape=(1152, 320, 3)),
    #         keras.layers.BatchNormalization(),
    #         keras.layers.ReLU(),
    #         *self._conv_block(),
    #         *self._conv_block(),
    #         keras.layers.Dropout(0.5),
    #         *self._conv_block(),
    #         *self._conv_block(),
    #         keras.layers.Dropout(0.5),
    #         *self._conv_block(),
    #         *self._conv_block(),
    #         keras.layers.Dropout(0.5),
    #         keras.layers.Flatten(),

    #         # FC - Relu - Dropout 1
    #         keras.layers.Dense(2560),
    #         keras.layers.ReLU(),
    #         keras.layers.Dropout(0.5),
    #         # FC - Relu - Dropout 2
    #         keras.layers.Dense(1792),
    #         keras.layers.ReLU(),
    #         keras.layers.Dropout(0.5),
    #         # FC - Relu - Dropout 3
    #         keras.layers.Dense(2, activation='sigmoid')
    #     ])
    #     model.compile(optimizer='adam', loss='mean_squared_error')
    #     model.summary()
    #     return model

    # def create_local_model(self):  # Top Level
    #     pass

    # def train_local(self, vid_path):  # Run
    #     # load a video
    #     # load labels
    #     # create training set to work with
    #     # initialize model
    #     # set up training
    #     frames, positions = self.load_labels(1, train=False)
    #     frame_stacks_gen = self.load_frame_stacks_generator(vid_path, frames)

    #     first_stack = next(frame_stacks_gen)
    #     print(first_stack.shape)
    #     global_model = self.create_global_model()
    #     print(global_model.predict(np.expand_dims(first_stack, 0)))
    #     return frames, positions, frame_stacks_gen, global_model


if __name__ == '__main__':
    x = Detect_Ball_Model()
    self = x
    vid_path = x.test_paths[0]
    frames, positions, frame_stacks_gen, global_model = x.train_local(vid_path)
    # first_stack = x.train_local(vid_path)
