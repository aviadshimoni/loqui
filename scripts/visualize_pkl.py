# encoding: utf-8
import cv2
import torch
import argparse
import numpy as np


def view_video_from_pkl(pkl_file: str):
    """
    Function to display a pickle file as mp4.
    Gets a path to a pickle file and displays it to the screen as a video
    """

    with open(pkl_file, 'rb') as f:
        result = torch.load(f)
        video = result['video']

    for i, frame_bytes in enumerate(video):
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Frame', frame)
        cv2.waitKey(30)  # Delay between frames (in milliseconds)


def main():
    view_video_from_pkl(pkl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_path', help="Path to pickle file to be displayed", default=None, required=True)
    args = parser.parse_args()
    pkl = vars(args).get("pickle_path")
    main()
