# encoding: utf-8
import os
import cv2
import numpy as np
import utils.system_globals as sg


def ensure_dir(directory: str) -> None:
    """
    gets a path to directory, if it doesn't exist - create it
    :param directory: path to directory
    :return: None
    """

    if not os.path.exists(directory):
        os.makedirs(directory)


def extract_opencv(file_name: str) -> list:
    """
    gets a path to video file, try to extract the ROI from it
    :param file_name: path to video file
    :return: ROI of given video file
    """

    video = []
    cap = cv2.VideoCapture(file_name)

    while cap.isOpened():
        ret, frame = cap.read()  # BGR

        if ret:
            frame = frame[115:211, 79:175]
            frame = sg.jpeg.encode(frame)
            video.append(frame)
        else:
            break

    cap.release()

    return video


def load_duration(file: str) -> np.array:
    """
    gets a path to an video example file and return its duration calculated by numpy
    :param file: path for video file
    :return: duration of the file
    """

    with open(file, "r") as f:
        lines = f.readlines()

        for line in lines:
            if line.find("Duration") != -1:
                duration = float(line.split(' ')[1])

    tensor = np.zeros(29)
    mid = 29 / 2
    start = int(mid - duration / 2 * 25)
    end = int(mid + duration / 2 * 25)
    tensor[start:end] = 1.0

    return tensor
