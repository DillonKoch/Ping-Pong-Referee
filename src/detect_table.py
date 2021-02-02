# ==============================================================================
# File: detect_table.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:17:44 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 2nd February 2021 11:18:39 am
# Modified By: Dillon Koch
# -----
#
# -----
# file for detecting the corners of the ping pong table in a video
# using either semantic segmentation (from TTNet paper or otherwise)
# or using edge detection, like in the document scanner posts
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Detect_Table:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Detect_Table()
    self = x
    x.run()
