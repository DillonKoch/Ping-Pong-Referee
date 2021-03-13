# ==============================================================================
# File: record_bounce.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:26:56 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 2nd February 2021 11:27:55 am
# Modified By: Dillon Koch
# -----
#
# -----
# When bounces are detected, this will use the location of the ball to record
# where it lands on the table
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Record_Bounce:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Record_Bounce()
    self = x
    x.run()
