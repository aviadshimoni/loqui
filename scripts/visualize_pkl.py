import cv2
import torch
import numpy as np


def view_video_from_pkl(pkl_file):
    with open(pkl_file, 'rb') as f:
        result = torch.load(f)
        video = result['video']

    for i, frame_bytes in enumerate(video):
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_GRAYSCALE)
        cv2.imshow('Frame', frame)
        cv2.waitKey(30)  # Delay between frames (in milliseconds)


pkl_path = "/Users/aviads/Desktop/colman/loqui/loqui/dataset/lrw_roi_npy_gray_pkl_jpeg/ABOUT/test/ABOUT_00001.pkl"
view_video_from_pkl(pkl_path)