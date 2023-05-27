# encoding: utf-8
import os
import cv2
import glob
import time
import torch
import numpy as np
from turbojpeg import TurboJPEG
from torch.utils.data import Dataset, DataLoader
from moviepy.editor import VideoFileClip

jpeg = TurboJPEG()


def get_video_duration(file_path):
    video = VideoFileClip(file_path)
    duration = video.duration
    video.close()
    return duration


def ensure_dir(directory: str) -> None:
    """
    gets a path to directory, if it doesn't exist - create it
    :param directory: path to directory
    :return: None
    """

    if not os.path.exists(directory):
        os.makedirs(directory)


def load_duration(file: str) -> np.array:
    """
    gets a path to a video example file and returns its duration calculated by numpy
    :param file: path for the video file
    :return: duration of the file
    """

    with open(file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.find('Duration') != -1:
                duration = float(line.split(' ')[1])

    tensor = np.zeros(29)
    mid = 29 / 2
    start = int(mid - duration / 2 * 25)
    end = int(mid + duration / 2 * 25)
    tensor[start:end] = 1.0

    return tensor.astype(np.bool_)


def extract_opencv(file_name: str) -> list:
    """
     Gets a path to a video file, tries to extract the ROI from it.
     :param file_name: Path to the video file.
     :return: ROI frames of the given video file.
     """

    video = []
    cap = cv2.VideoCapture(file_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame_count = 29
    start_frame = max(0, (frame_count - target_frame_count) // 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while cap.isOpened() and len(video) < target_frame_count:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (256, 256))
            frame = frame[115:211, 79:175]
            print(frame.shape)
            _, jpeg_frame = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg_frame.tobytes()
            video.append(frame_bytes)
        else:
            break

    cap.release()

    # Pad the video with duplicate frames if it has less than 29 frames
    if len(video) < target_frame_count:
        padding_frames = video[-1:] * (target_frame_count - len(video))
        video.extend(padding_frames)

    return video


# target_dir = 'lrw_roi_80_116_175_211_npy_gray_pkl_jpeg'
target_dir = '/tf/Daniel/custom_lipread_pkls'
ensure_dir(target_dir)


class LRWDataset(Dataset):
    """
    Object that represents the preparation process of LRW dataset.
    Inherits LRWDatasetInterface which implements Dataset of torch module
    and generates training samples of LRW
    """

    def __init__(self, mp4_path='lrw_mp4'):
        self.mp4_path = mp4_path

        with open('/tf/loqui/label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()

        self.list = []

        for i, label in enumerate(self.labels):
            files = glob.glob(os.path.join(self.mp4_path, label, '*', '*.mp4'))

            for file in files:
                file_to_save = file.replace(self.mp4_path, target_dir).replace('.mp4', '.pkl')
                save_path = os.path.split(file_to_save)[0]
                ensure_dir(save_path)

            files = sorted(files)
            self.list += [(file, i) for file in files]

    def __getitem__(self, idx: int) -> dict:
        """
        implements the operator []
        :param idx: index to return of the dataset object
        :return: by given index, return the respectively data on that index
        """

        inputs = extract_opencv(self.list[idx][0])

        duration = self.list[idx][0]
        metadata_file = duration.replace('.mp4', '.txt')
        labels = self.list[idx][1]

        if os.path.isfile(metadata_file):
            duration = load_duration(metadata_file)
        else:
            duration = get_video_duration(self.list[idx][0])

        result = {'video': inputs,
                  'label': int(labels),
                  'duration': duration}

        output = self.list[idx][0].replace(self.mp4_path, target_dir).replace('.mp4', '.pkl')
        torch.save(result, output)

        return result

    def __len__(self) -> int:
        """
        implements the len operator
        :return: len of self.list
        """

        return len(self.list)


def main():
    loader = DataLoader(LRWDataset("/tf/Daniel/custom_lipread_mp4"),
                        batch_size=1,
                        num_workers=16,
                        shuffle=False,
                        drop_last=False)

    tic = time.time()

    for i, batch in enumerate(loader):
        toc = time.time()
        eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
        print(f'eta:{eta:.5f}')


if __name__ == '__main__':
    main()
