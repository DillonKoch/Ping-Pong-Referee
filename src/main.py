# ==============================================================================
# File: main.py
# Project: src
# File Created: Friday, 19th March 2021 11:04:27 pm
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 20th March 2021 12:53:24 am
# Modified By: Dillon Koch
# -----
#
# -----
# Using all trained models to analyze a new video
# ==============================================================================


import sys
from os.path import abspath, dirname

import torch

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from ball_tracking import Detect_Ball
from event_detection import EventDetection
from load_data import Load_Video
from shot_chart_video import Shot_Chart_Video
from table_segmentation import Segmentation


class Main:
    def __init__(self, gameplay_path):
        self.gameplay_path = gameplay_path
        self.lv = Load_Video(self.gameplay_path)
        self.scv = Shot_Chart_Video()

    def run_ball_model(self, vid):
        model = Detect_Ball()
        model = torch.load(ROOT_PATH + "/src/ball_detection_local.pth")
        model.eval()
        ball_dict = {frame: model(frame) for frame in range(len(vid))}
        ball_dict = {key: val for key, val in ball_dict.items() if ((val[0] > 0) and (val[1] > 0))}
        return ball_dict

    def run_event_model(self, vid):
        model = EventDetection()
        model = torch.load(ROOT_PATH + "/src/event_detection.pth")
        model.eval()
        event_dict = {frame: model(frame) for frame in range(len(vid))}
        event_dict = {key: val for key, val in event_dict.items() if ((val[0] > 0) and (val[1] > 0))}
        return event_dict

    def get_corners(self, vid):
        model = EventDetection()
        model = torch.load(ROOT_PATH + "/src/segmentation.pth")
        model.eval()
        vid_corners = {frame: model(frame) for frame in range(0, len(vid), 250)}
        vid_corners = {key: val for key, val in vid_corners.items() if ((val[0] > 0) and (val[1] > 0))}
        return vid_corners

    def run(self, chart_only=False, coffin_corner=False):  # Run
        vid = self.lv.run(self.gameplay_path)
        ball_dict = self.run_ball_model(vid)
        bounce_dict = self.run_event_model(vid)
        vid_corners = self.get_corners(vid)
        chart = self.scv.run(ball_dict, bounce_dict, vid_corners, self.gameplay_path, chart_only, coffin_corner)
        return chart


if __name__ == '__main__':
    x = Main()
    self = x
    x.run()
