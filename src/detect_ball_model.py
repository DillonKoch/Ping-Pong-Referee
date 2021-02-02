# ==============================================================================
# File: detect_ball_model.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:08:57 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 2nd February 2021 11:18:42 am
# Modified By: Dillon Koch
# -----
#
# -----
# Trains a model to detect the ball in a stack of 9 frames of ping pong video
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Detect_Ball_Model:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Detect_Ball_Model()
    self = x
    x.run()
