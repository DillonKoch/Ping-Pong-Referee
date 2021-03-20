# Ping Pong Shot Charts and Training Games


### Steps to run locally
1. Clone the repository
```
$ git clone https://github.com/DillonKoch/Ping-Pong-Shot-Chart.git
```
2. Run the following code to download the OpenTTGames dataset and extract the necessary images from the videos to train models. This may take a while.
```
$ cd Ping-Pong-Shot-Chart/src
$ python download_data.py
$ python extract_imgs.py
```
3. Each of the three models used in this repo can be trained by running these files:
```
$ python ball_tracking.py
$ python event_detection.py
$ python table_segmentation.py
```

4. Once those models are trained, run the main file to create a new video that displays the game and shot chart side by side. Specify the path to the video you'd like to run the models on with the --path argument.
```
$ python main.py --path=/path/to/new/video
```
