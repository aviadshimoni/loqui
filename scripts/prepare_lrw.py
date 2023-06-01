import os
import cv2
import glob
import time
import torch
import numpy as np
from turbojpeg import TurboJPEG
from torch.utils.data import Dataset, DataLoader
from moviepy.editor import VideoFileClip
import codecs

jpeg = TurboJPEG()

def get_original_video_duration(file_path):
    video = VideoFileClip(file_path)
    duration = video.duration
    video.close()
    return duration

def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

import codecs

def load_duration(file: str) -> np.array:
    with codecs.open(file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

        duration = None

        for line in lines:
            if line.find('Duration') != -1:
                duration = float(line.split(' ')[1])
                break

        if duration is None:
            raise ValueError("Duration not found in the file.")

    tensor = np.zeros(29)
    mid = 29 / 2
    start = int(mid - duration / 2 * 25)
    end = int(mid + duration / 2 * 25)
    tensor[start:end] = 1.0

    return tensor.astype(np.bool_)


def extract_opencv(file_name: str, roi_coordinates: tuple) -> tuple:
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
            frame = frame[roi_coordinates[0]:roi_coordinates[1], roi_coordinates[2]:roi_coordinates[3]]
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

    return video, frame_count

target_dir = '/tf/loqui/custom_lipread_pkls'
ensure_dir(target_dir)

class LRWDataset(Dataset):
    def __init__(self, mp4_path='lrw_mp4', roi_coordinates=(115, 211, 79, 175)):
        self.mp4_path = mp4_path

        with open('/tf/loqui/label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()

        self.data = []

        for i, label in enumerate(self.labels):
            files = glob.glob(os.path.join(self.mp4_path, label, '*', '*.mp4'))

            for file in files:
                file_to_save = file.replace(self.mp4_path, target_dir).replace('.mp4', '.pkl')
                save_path = os.path.split(file_to_save)[0]
                ensure_dir(save_path)

            files = sorted(files)
            self.data += [(file, i) for file in files]

        self.roi_coordinates = roi_coordinates

    def __getitem__(self, idx: int) -> dict:
        video_path, label = self.data[idx]
        inputs, frame_count = extract_opencv(video_path, roi_coordinates=self.roi_coordinates)

        metadata_file = video_path.replace('.mp4', '.txt')

        if os.path.isfile(metadata_file):
            duration = load_duration(metadata_file)
            print(f"duration lrw mp4: {duration}")
        else:
            original_duration = get_original_video_duration(video_path)
            duration = original_duration * 29 / frame_count
            print(f"duration loqui-custom mp4: {duration}")

        result = {
            'video': inputs,
            'label': int(label),
            'duration': duration
        }

        output = video_path.replace(self.mp4_path, target_dir).replace('.mp4', '.pkl')
        torch.save(result, output)

        return result

    def __len__(self) -> int:
        return len(self.data)


def main():
    loader = DataLoader(
        LRWDataset("/tf/Daniel/custom_lipread_mp4"),
        batch_size=1,
        num_workers=16,
        shuffle=False,
        drop_last=False
    )

    tic = time.time()

    for i, batch in enumerate(loader):
        toc = time.time()
        eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
        print(f'eta:{eta:.5f}')

if __name__ == '__main__':
    main()
