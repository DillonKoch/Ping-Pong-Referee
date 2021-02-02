# ==============================================================================
# File: transform_table.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:28:10 am
# Author: Dillon Koch
# -----
# Last Modified: Tuesday, 2nd February 2021 11:29:36 am
# Modified By: Dillon Koch
# -----
#
# -----
# Transforms the table from trapezoid-shaped in the video to bird's eye view
# the dots can be put on the table before transformation, or added after (if the
# transformation matrix is solved)
# ==============================================================================


from os.path import abspath, dirname
import sys

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Transform_Table:
    def __init__(self):
        pass

    def run(self):  # Run
        pass


if __name__ == '__main__':
    x = Transform_Table()
    self = x
    x.run()
