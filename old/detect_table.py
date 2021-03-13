# ==============================================================================
# File: detect_table.py
# Project: src
# File Created: Tuesday, 2nd February 2021 11:17:44 am
# Author: Dillon Koch
# -----
# Last Modified: Friday, 5th February 2021 9:07:12 pm
# Modified By: Dillon Koch
# -----
#
# -----
# file for detecting the corners of the ping pong table in a video
# using either semantic segmentation (from TTNet paper or otherwise)
# or using edge detection, like in the document scanner posts
# ==============================================================================


import sys
from os.path import abspath, dirname

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Detect_Table:
    def __init__(self):
        pass

    def run(self):  # Run
        # step 1
        img = cv2.imread('test.png')
        original_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.Canny(img, 100, 250)
        # step 2
        # temp
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(img, kernel, iterations=1)
        # end temp
        contours = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # contours = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        print(len(contours))
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            new_img = cv2.drawContours(original_img, [approx], -1, (0, 255, 0), 2)
            # if len(approx) == 3:
            #     screenCnt = approx
            #     new_img = cv2.drawContours(original_img, [screenCnt], -1, (0, 255, 0), 2)

        # lines = cv2.HoughLines(img, 1, np.pi / 180, 200)
        # for line in lines:
        #     r = line[0][0]
        #     theta = line[0][1]
        #     a = np.cos(theta)
        #     b = np.sin(theta)
        #     x0 = a * r
        #     y0 = b * r
        #     x1 = int(x0 + 1000 * (-b))
        #     y1 = int(y0 + 1000 * (a))
        #     x2 = int(x0 - 1000 * (-b))
        #     y2 = int(y0 - 1000 * (a))
        #     cv2.line(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return img

    def temp(self):
        img = cv2.imread('test.png')
        original_img = img.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        # edges = cv2.Canny(img, 100, 250)
        edges = self.run()
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, maxLineGap=85)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            a_sq = (y2 - y1)**2
            b_sq = (x2 - x1)**2
            line_len = np.sqrt(a_sq + b_sq)
            if line_len > 200:
                cv2.line(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return original_img


if __name__ == '__main__':
    x = Detect_Table()
    self = x
    img = x.run()
    # img = x.temp()
    cv2.imwrite('temp.png', img)
    plt.imshow(img)
