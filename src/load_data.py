# ==============================================================================
# File: load_data.py
# Project: src
# File Created: Tuesday, 19th January 2021 8:16:12 pm
# Author: Dillon Koch
# -----
# Last Modified: Sunday, 7th March 2021 10:33:00 am
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


class Load_Video:
    """
    - loads 9 frames at a time from the video (0-8, 1-9, 2-10, ...)
    - can call next() on this
    """

    def __init__(self, vid_path):
        self.vid_path = vid_path
        self.frame_count = 0

    def load_cap(self):  # Top Level
        assert os.path.isfile(self.vid_path)
        cap = cv2.VideoCapture(self.vid_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'({width}x{height}) at {fps} fps, {num_frames} frames')
        return cap, num_frames

    def vid_generator(self, cap, num_frames, resize, sequence_length, rollaxis):  # Top Level
        count = 0
        frames = []
        while count < num_frames:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (320, 128)) if resize else frame
            frame = np.rollaxis(frame, 2, 0) if rollaxis else frame
            frames.append(frame)
            if len(frames) == sequence_length:
                yield np.array(frames)
                frames = frames[1:]

    def run(self, resize=False, sequence_length=9, rollaxis=True):
        cap, num_frames = self.load_cap()
        vid_generator = self.vid_generator(cap, num_frames, resize, sequence_length, rollaxis)
        return vid_generator


if __name__ == '__main__':
    vid_path = ROOT_PATH + "/Data/Test/Game1/gameplay.mp4"
    x = Load_Video(vid_path)
    self = x
    vid_generator = x.run()
