# ==============================================================================
# File: detect_event_model.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:11:32 am
# Author: Dillon Koch
# -----
# Last Modified: Friday, 5th February 2021 1:53:04 pm
# Modified By: Dillon Koch
# -----
#
# -----
# Trains a model to detect an event (bounce, net hit, non-event) from a stack
# of 9 frames of ping pong video
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Detect_Event_Model:
    def __init__(self):
        pass

    def create_model(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Detect_Event_Model()
    self = x
    x.run()
