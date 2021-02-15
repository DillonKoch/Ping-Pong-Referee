# ==============================================================================
# File: load_data.py
# Project: src
# File Created: Tuesday, 19th January 2021 8:16:12 pm
# Author: Dillon Koch
# -----
# Last Modified: Friday, 5th February 2021 2:32:20 pm
# Modified By: Dillon Koch
# -----
#
# -----
# Loading a video into a numpy array
# ==============================================================================

import os
import sys
from os.path import abspath, dirname

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Load_Data:
    def __init__(self):
        pass

    def load_cap(self, vid_path):  # Top Level
        assert os.path.exists(vid_path), f"path {vid_path} does not exist"
        cap = cv2.VideoCapture(vid_path)
        return cap

    def cap_info(self, cap):  # Top Level
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return frame_count, frame_width, frame_height

    def load_video_arr(self, cap, frame_count, frame_width, frame_height, max_frames):
        num_frames = min(frame_count, max_frames)
        arr = np.empty((num_frames, frame_height, frame_width, 3), np.dtype('uint8'))
        inserted_frames = 0
        while inserted_frames < num_frames:
            ret, arr[inserted_frames] = cap.read()
            inserted_frames += 1
        # cap.release()
        return arr

    def load_video_arr_generator(self, cap, frame_count, frame_width, frame_height):
        pass

    def run(self, vid_path, generator=False, max_frames=100):  # Run
        cap = self.load_cap(vid_path)
        frame_count, frame_width, frame_height = self.cap_info(cap)
        arr = self.load_video_arr(cap, frame_count, frame_width, frame_height, max_frames)
        return arr


class Load_Training_Data:
    def __init__(self):
        pass

    def load_ball_labels(self):  # Top Level
        """
        loads the
        """
        pass

    def load_event_labels(self):
        pass

    def run_ball_detection(self, video_path, labels_path):  # Run
        """
        """
        pass

    def run_event_detection(self, video_path, labels_path):  # Run
        """
        """
        pass


if __name__ == '__main__':
    x = Load_Data()
    self = x
    vid_path = ROOT_PATH + "/Data/Test/Game1/gameplay.mp4"
    arr = x.run(vid_path)
