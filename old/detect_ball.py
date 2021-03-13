# ==============================================================================
# File: detect_ball.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:08:33 am
# Author: Dillon Koch
# -----
# Last Modified: Friday, 5th February 2021 2:10:54 pm
# Modified By: Dillon Koch
# -----
#
# -----
# Uses the model trained from detect_ball_model.py to find the ball location
# in a stack of 9 frames
# ==============================================================================

from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Detect_Ball:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Detect_Ball()
    self = x
    x.run()
