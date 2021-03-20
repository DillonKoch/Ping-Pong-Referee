# ==============================================================================
# File: extract_imgs.py
# Project: src
# File Created: Saturday, 6th March 2021 2:04:36 pm
# Author: Dillon Koch
# -----
# Last Modified: Friday, 19th March 2021 11:35:46 pm
# Modified By: Dillon Koch
# -----
#
# -----
# saves off all the frames from the videos used for training
# ==============================================================================

import concurrent.futures
import json
import os
import sys
from os.path import abspath, dirname

import cv2
from tqdm import tqdm

ROOT_PATH = dirname(dirname(abspath(__file__)))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)


class Extract_Imgs:
    def __init__(self):
        self.train_path = ROOT_PATH + "/Data/Train/"
        self.test_path = ROOT_PATH + "/Data/Test/"

    def create_folders(self):  # Top Level
        """
        creates an "images" folder in each game folder inside Train/Test
        """
        test_games = [self.test_path + game for game in os.listdir(self.test_path)]
        train_games = [self.train_path + game for game in os.listdir(self.train_path)]
        game_dirs = test_games + train_games
        for game in game_dirs:
            new_dir = game + "/images"
            if not os.path.isdir(new_dir):
                os.mkdir(new_dir)
        return game_dirs

    def _load_json(self, json_path):  # Specific Helper save_game_imgs
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

    def _smooth_frames(self, frames):  # Specific Helper save_game_imgs
        smooth_frames = []
        frames = [int(frame) for frame in frames]
        for frame in frames:
            start = frame - 9
            end = frame + 9
            for i in range(start, end, 1):
                smooth_frames.append(i)
        return list(set(smooth_frames))

    def _save_frames(self, game_dir, save_frames):  # Specific Helper save_game_imgs
        vid_path = game_dir + "/gameplay.mp4"
        cap = cv2.VideoCapture(vid_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(num_frames)):
            ret, frame = cap.read()
            if i in save_frames:
                path = game_dir + f"/images/frame_{i}.png"
                if not os.path.exists(path):
                    assert cv2.imwrite(path, frame)

    def save_game_imgs(self, game_dir):  # Top Level
        ball_markup = self._load_json(game_dir + "/ball_markup.json")
        ball_frames = self._smooth_frames(list(ball_markup.keys()))
        event_markup = self._load_json(game_dir + "/events_markup.json")
        event_frames = self._smooth_frames(list(event_markup.keys()))
        save_frames = list(set(ball_frames + event_frames))
        self._save_frames(game_dir, save_frames)

    def run(self):  # Run
        game_dirs = self.create_folders()
        game_dir = game_dirs[0]

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     result = executor.map(self.save_game_imgs, game_dirs)

        for game_dir in game_dirs:
            self.save_game_imgs(game_dir)

        # for each video in testing and training
        # load the markup jsons
        # create list of all frames needed to be extracted
        # go through the video with cv2.cap and save them off


def func(game_dir):
    x = Extract_Imgs()
    x.save_game_imgs(game_dir)


if __name__ == '__main__':
    x = Extract_Imgs()
    self = x
    game_dirs = x.create_folders()
    x.run()
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     for game_dir in game_dirs:
    #         executor.submit(func, game_dir=game_dir)
