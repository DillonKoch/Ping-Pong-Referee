# ==============================================================================
# File: load_data.py
# Project: src
# File Created: Tuesday, 19th January 2021 8:16:12 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 31st January 2021 10:12:54 am
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

    def run(self, vid_path, max_frames=100):  # Run
        cap = self.load_cap(vid_path)
        frame_count, frame_width, frame_height = self.cap_info(cap)
        arr = self.load_video_arr(cap, frame_count, frame_width, frame_height, max_frames)
        return arr


if __name__ == '__main__':
    x = Load_Data()
    self = x
    vid_path = ROOT_PATH + "/Data/Game1/gameplay.mp4"
    arr = x.run(vid_path)
