# ==============================================================================
# File: detect_event.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:12:22 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 2nd February 2021 11:18:51 am
# Modified By: Dillon Koch
# -----
#
# -----
# Uses the model trained in detect_event_model.py to predict the probability
# of an event in a stack of 9 frames
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Detect_Event:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Detect_Event()
    self = x
    x.run()
