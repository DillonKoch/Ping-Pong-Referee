# ==============================================================================
# File: main.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:19:56 am
# Author: Dillon Koch
# -----
# Last Modified: Friday, 5th February 2021 8:47:33 am
# Modified By: Dillon Koch
# -----
#
# -----
# Combines detect_event, detect_ball, and detect_table to process a video of
# ping pong gameplay and record the pixel locations where the ball bounces on the table,
# then uses those points to create a ping-pong shaped shot chart
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from src.detect_ball import Detect_Ball
from src.detect_event import Detect_Event
from src.detect_table import Detect_Table
from src.load_data import Load_Data


class Main:
    def __init__(self):
        pass

    def load_video(self, video_path):  # Top Level
        """
        just load the video as needed
        """
        pass

    def detect_events(self, vid):  # Top Level
        """
        runs the detect_events model to find bounces in the video
        """
        pass

    def find_bounce_pixels(self, bounce_frame):  # Top Level
        """
        given a frame where a bounce was predicted,
        this finds the pixel where ball meets table
        """
        pass

    def find_table_dimensions(self, bounce_frames):  # Top Level
        """
        finds the four corners of the table in the given frame
        """
        pass

    def compute_chart_bounce_location(self, video_bounce_pixels, table_dimensions):  # Top Level
        """
        uses the bounce location and table dimensions to compute the location on the
        final bird's eye shot chart where the ball bounced
        """
        pass

    def create_shot_chart(self, chart_bounce_locations):  # Top Level
        """
        creates an image of a ping pong table with dots where bounces occurred
        """
        # using the pixel locations of bounces, add them all to a trapezoid shot chart,
        # then transform to a rectangle chart
        # --------- OR --------------
        # using the four corners of the table, solve for the transformation matrix
        # use that transformation matrix to convert pixel locations in the trapezoid to
        # locations in the rectangle shot chart
        # then put a dot on those locations for the first time in the rectangle!
        pass

    def run(self, video_path):  # Run
        vid = self.load_video(video_path)
        bounce_frames, net_hits = self.detect_events(vid)
        video_bounce_pixels = self.find_bounce_pixels(bounce_frames)
        table_dimensions = self.find_table_dimensions(bounce_frames)
        chart_bounce_locations = self.compute_chart_bounce_locations(bounce_frames, video_bounce_pixels, table_dimensions)
        shot_chart = self.create_shot_chart(chart_bounce_locations)
        return shot_chart


if __name__ == '__main__':
    x = Main()
    self = x
    shot_chart = x.run()
