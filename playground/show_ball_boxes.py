# ==============================================================================
# File: show_ball_boxes.py
# Project: src
# File Created: Saturday, 30th January 2021 8:31:31 pm
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 2nd February 2021 11:19:01 am
# Modified By: Dillon Koch
# -----
#
# -----
# showing bounding boxes of the ball in a video
# ==============================================================================


import json
import sys
from os.path import abspath, dirname

import numpy as np
from tqdm import tqdm
import cv2

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.load_data import Load_Data


class Show_Ball:
    def __init__(self):
        self.load_data = Load_Data()

    def load_ball_dict(self, data_subfolder):  # Top Level
        """
        loads the ball_markup.json file in a /Data/{Game} folder to a dict
        - data_subfolder: name of the subfolder in /Data/ with json files and .mp4 video
        """
        json_path = ROOT_PATH + f"/Data/{data_subfolder}/ball_markup.json"
        with open(json_path) as f:
            ball_dict = json.load(f)
        return ball_dict

    def load_event_dict(self, data_subfolder):  # Top Level
        """
        loads the events_markup.json file in a /Data/{Game} folder to a dict
        - data_subfolder: name of the subfolder in /Data/ with json files and .mp4 video
        """
        json_path = ROOT_PATH + f"/Data/{data_subfolder}/events_markup.json"
        with open(json_path) as f:
            event_dict = json.load(f)
        return event_dict

    def add_ball_box(self, arr, i, ball_dict):  # Top Level
        frame_index = str(i)
        if frame_index in list(ball_dict.keys()):
            xy_dict = ball_dict[frame_index]
            x = xy_dict['x']
            y = xy_dict['y']
            x_min = max(0, x - 9)
            x_max = min(1920, x + 9)
            y_min = max(0, y - 9)
            y_max = min(1080, y + 9)
            ball_arr = cv2.rectangle(arr, (x_max, y_max), (x_min, y_min), (0, 255, 0), 2)
            # assert not np.array_equal(ball_arr, arr)
            return ball_arr
        return arr

    def _detect_near_event(self, frame_index, event_dict):  # Specific Helper  add_event_box
        """
        detects if the frame is near an event in the events_markup.json file
        - frame_buffer: allowable amount of frames away from the event
        """
        event_frames = [int(key) for key in list(event_dict.keys())]
        closest_event = event_frames[min(range(len(event_frames)), key=lambda i: abs(event_frames[i] - frame_index))]
        if abs(closest_event - frame_index) < 5:
            return event_dict[str(closest_event)]
        return None

    def add_event_box(self, arr, i, event_dict):  # Top Level
        empty_color = (255, 255, 255)
        bounce_color = (0, 255, 0)
        net_color = (255, 0, 0)
        frame_index = i
        event = self._detect_near_event(frame_index, event_dict)
        if event is not None:
            color = empty_color if event == 'empty' else net_color if event == 'net' else bounce_color
            event_arr = cv2.rectangle(arr, (100, 100), (1820, 980), color, 2)
            return event_arr
        return arr

    def write_video(self, out, arr):  # Top Level
        out.write(arr)
        cv2.imshow('frame', arr)
        c = cv2.waitKey(1)

    def run(self, data_subfolder, max_frames=np.inf):  # Run
        """
        - data_subfolder: name of the subfolder in /Data/ with json files and .mp4 video
        """
        cap = self.load_data.load_cap(ROOT_PATH + f"/Data/{data_subfolder}/gameplay.mp4")
        ball_dict = self.load_ball_dict(data_subfolder)
        event_dict = self.load_event_dict(data_subfolder)
        frame_count, frame_width, frame_height = self.load_data.cap_info(cap)
        frames_to_read = min(frame_count, max_frames)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output.mp4', fourcc, 40.0, (1920, 1080))

        for i in tqdm(range(frames_to_read)):
            arr = self.load_data.load_video_arr(cap, frame_count, frame_width, frame_height, 1)[0]
            arr = self.add_ball_box(arr, i, ball_dict)
            arr = self.add_event_box(arr, i, event_dict)
            self.write_video(out, arr)
        out.release()


if __name__ == '__main__':
    x = Show_Ball()
    self = x
    data_subfolder = "Game1"
    x.run(data_subfolder, 1000000)
