# ==============================================================================
# File: shot_chart_video.py
# Project: src
# File Created: Sunday, 7th March 2021 9:22:31 am
# Author: Dillon Koch
# -----
# Last Modified: Saturday, 20th March 2021 12:41:08 am
# Modified By: Dillon Koch
# -----
#
# -----
# creates a video of the shot chart populating in real time next to gameplay
# also an option to not show the gameplay and just make a video of the chart
# ==============================================================================


import copy
import json
import sys
from os.path import abspath, dirname

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.load_data import Load_Video
from src.table import Table


class Shot_Chart_Video:
    def __init__(self):
        pass

    def dimensions_to_corners(self, dimensions):  # Top Level
        """
        extracts the shot chart table dimensions into the four corners
        """
        top_left = (dimensions['table x1'], dimensions['table y1'])
        bottom_left = (dimensions['table x1'], dimensions['table y2'])
        top_right = (dimensions['table x2'], dimensions['table y1'])
        bottom_right = (dimensions['table x2'], dimensions['table y2'])
        return top_left, bottom_left, top_right, bottom_right

    def compute_homography(self, vid_corners, chart_corners):  # Specific Helper video_xy_to_chart
        """
        computes the 3x3 homogenous matrix H used to map points from video to shot chart
        """
        # arranging the corner locations into the P matrix (Ph = 0)
        P = np.zeros((8, 9))
        for i, (pp, p) in enumerate(zip(chart_corners, vid_corners)):
            pp = np.append(pp, 1)
            up, vp, _ = pp
            p = np.append(p, 1)
            new_P = np.zeros((2, 9))
            new_P[0, :3] = p.T
            new_P[0, -3:] = -up * p.T
            new_P[1, 3:6] = p.T
            new_P[1, -3:] = -vp * p.T
            P[(i * 2):(i * 2) + 2] = new_P

        # solving for H using SVD on the Ph = 0 equation
        u, s, v = np.linalg.svd(P)
        h = v.T[:, -1]
        h1 = h[:3]
        h2 = h[3:6]
        h3 = h[6:]
        H = np.array([h1, h2, h3])
        H /= H[2, 2]
        return H

    def video_xy_to_chart(self, vid_corners, chart_corners, ball_x, ball_y):  # Top Level
        """
        uses corner correspondences to map an (x, y) position from video to the chart
        """
        H = self.compute_homography(vid_corners, chart_corners)
        x, y, z = H.dot(np.array([ball_x, ball_y, 1]))
        chart_x = x / z
        chart_y = y / z
        return int(chart_x), int(chart_y)

    def _load_gameplay(self, gameplay_path):  # Specific Helper save_chart_to_vid
        """
        loads a generator of gameplay video
        """
        load_video = Load_Video(gameplay_path)
        vid_generator = load_video.run(sequence_length=1, rollaxis=False)
        return vid_generator

    def _out_writer(self):  # Specific Helper save_chart_to_vid
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('shot_chart.mp4', fourcc, 120.0, (1920, 1080))
        return out

    def _combine_gameplay_chart(self, vid_generator, current_chart, chart_width_pct=0.4):   # Specific Helper save_chart_to_vid
        """
        combines a gameplay frame and shot chart into one frame
        """
        # creating resized chart
        chart_width = int(1920 * chart_width_pct)
        chart_height = int(1080 / (1920 / chart_width))
        current_chart = cv2.resize(current_chart, (chart_width, chart_height))

        # creating resized gameplay frame
        gameplay_width = int(1920 - chart_width)
        gameplay_height = int(1080 / (1920 / gameplay_width))
        gameplay_frame = next(vid_generator)[0]
        # cv2.imwrite('game_frame.png', gameplay_frame)
        # print('here')
        gameplay_frame = cv2.resize(gameplay_frame, (gameplay_width, gameplay_height))

        # combining chart and gameplay
        final_frame = np.zeros((1080, 1920, 3))
        gameplay_border = int((1080 - gameplay_height) / 2)
        chart_border = int((1080 - chart_height) / 2)
        final_frame[gameplay_border:-gameplay_border, :gameplay_width, :] = gameplay_frame
        final_frame[chart_border:-chart_border, gameplay_width:, :] = current_chart
        return final_frame

    def save_chart_to_vid(self, bounce_dict, chart_history, gameplay_path, chart_only):  # Top Level
        """
        uses the shot chart's history to visualize the bounces appearing as the game is played
        """
        vid_generator = self._load_gameplay(gameplay_path) if not chart_only else None
        out_writer = self._out_writer()
        bounce_frames = [int(key) for key in list(bounce_dict.keys())]
        num_frames = max(bounce_frames) + 200

        # looping over gameplay frames and chart frames to create video
        chart_num = 0
        for i in tqdm(range(num_frames)):
            if i in bounce_frames:
                chart_num += 1
            current_chart = chart_history[chart_num]
            out_frame = self._combine_gameplay_chart(vid_generator, current_chart) if vid_generator is not None else current_chart
            out_writer.write(out_frame.astype('uint8'))
        out_writer.release()

    def run(self, ball_dict, bounce_dict, vid_corners, gameplay_path, chart_only=False, coffin_corner=False):  # Run
        table = Table()
        chart, dimensions = table.run(coffin_corner=coffin_corner)
        chart_corners = self.dimensions_to_corners(dimensions)
        chart_history = [copy.deepcopy(chart)]
        score1 = 0
        score2 = 0
        for frame in list(bounce_dict.keys()):
            ball_x = ball_dict[frame]['x']
            ball_y = ball_dict[frame]['y']
            chart_x, chart_y = self.video_xy_to_chart(vid_corners, chart_corners, ball_x, ball_y)
            chart, score1, score2 = table.add_point(chart, chart_x, chart_y, dimensions, score1=score1, score2=score2)
            print(f'Added point ({chart_x}, {chart_y}) to shot chart')
            chart_history.append(copy.deepcopy(chart))
        self.save_chart_to_vid(bounce_dict, chart_history, gameplay_path, chart_only=chart_only)
        return chart_history


if __name__ == '__main__':
    x = Shot_Chart_Video()
    self = x

    game_path = ROOT_PATH + "/Data/Test/Game2/"
    # chart = x.run(ball_dict, bounce_dict, vid_corners, gameplay_path, chart_only, coffin_corner)
