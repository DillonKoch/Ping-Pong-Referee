# ==============================================================================
# File: main.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:19:56 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 2nd February 2021 1:12:53 pm
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
        # just load the video as needed
        pass

    def analyze_video(self, vid):  # Top Level
        # loop through 9 frame increments to look for the table, ball, events
        # when an event is detected, run record_bounce.py to get the pixel location
        table = None
        bounce_stacks = None
        bounce_locations = None
        bounce_frames = None
        for bounce_frame in bounce_frames:
            table_dims = self.get_table_dims(bounce_frame)
            new_item = (bounce_frame)
        video_data = [(bounce_frame, table_dims, bounce_location)]
        return video_data

    def analyze_video(self, vid):
        bounce_stacks = self.bounce_stacks_from_video(vid)

    def create_shot_chart(self, table, bounce_frames, bounce_locations):  # Top Level
        # using the pixel locations of bounces, add them all to a trapezoid shot chart,
        # then transform to a rectangle chart
        trap_chart = self.bounce_locations_to_chart(bounce_locations, table)

        # --------- OR --------------
        # using the four corners of the table, solve for the transformation matrix
        # use that transformation matrix to convert pixel locations in the trapezoid to
        # locations in the rectangle shot chart
        # then put a dot on those locations for the first time in the rectangle!

        pass

    def run(self, video_path):  # Run
        vid = self.load_video(video_path)
        video_data = self.analyze_video(vid)
        shot_chart = self.create_shot_chart(video_data)
        return shot_chart


if __name__ == '__main__':
    x = Main()
    self = x
    shot_chart = x.run()
