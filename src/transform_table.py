# ==============================================================================
# File: transform_table.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:28:10 am
# Author: Dillon Koch
# -----
# Last Modified: Wednesday, 24th February 2021 9:45:17 am
# Modified By: Dillon Koch
# -----
#
# -----
# Transforms the table from trapezoid-shaped in the video to bird's eye view
# the dots can be put on the table before transformation, or added after (if the
# transformation matrix is solved)
# ==============================================================================


import sys
from os.path import abspath, dirname

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Table:
    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width


class Transform_Table:
    def __init__(self):
        pass

    def _add_net_line(self, table, net_x, p1y, p2y):  # Helping Helper _add_table_lines
        """
        adds a grey dashed line to the middle of the table to represent the net
        """
        i = p1y
        while i < p2y:
            table = cv2.line(table, (net_x, i), (net_x, i + 9), (127, 127, 127), 2)
            i += 18
        return table

    def _add_table_lines(self, table):  # Specific Helper create_table
        """
        adds the border lines and middle line of the ping pong table to the image
        """
        length_width_ratio = 9 / 5
        table_width = 1500
        table_height = table_width / length_width_ratio
        p1x = int((1920 - table_width) / 2)
        p1y = int((1080 - table_height) / 2)
        p2x = int(1920 - p1x)
        p2y = int(1080 - p1y)
        table = cv2.rectangle(table, (p1x, p1y), (p2x, p2y), 0, 4)

        # adding middle line
        midline_y = int(p1y + (table_height / 2))
        table = cv2.rectangle(table, (p1x, midline_y), (p2x, midline_y), 0, 4)

        # adding dashed line for net
        net_x = int(p1x + (table_width / 2))
        table = self._add_net_line(table, net_x, p1y, p2y)
        return table

    def create_table(self):  # Top Level
        """
        """
        table = np.zeros((1080, 1920, 3)).astype('float64')
        table.fill(255)
        table = self._add_table_lines(table)
        return table

    def add_coffin_corners(self, table):  # Top Level
        """
        adds the shaeded coffin corner regions to the table if playing the coffin corner game
        """
        return table

    def add_point(self, table, x, y):  # Run
        """
        adds a blue (x, y) point to the table image
        """
        table = cv2.circle(table, (x, y), 1, (255, 0, 0), 6)
        return table

    def add_random_points(self, table):  # QA Testing
        """
        just plots a bunch of random points on the table to make sure add_point works
        """
        pass

    def run(self, coffin_corner=True):  # Run
        table = self.create_table()
        table = self.add_coffin_corners(table) if coffin_corner else table
        table = self.add_point(table, 1000, 300)
        cv2.imwrite('temp.png', table)
        return table


if __name__ == '__main__':
    x = Transform_Table()
    self = x
    table = x.run()
