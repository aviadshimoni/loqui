import cv2
from turbojpeg import TurboJPEG
import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset, DataLoader

jpeg = TurboJPEG()


def extract_opencv(filename):
    video = []
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        ret, frame = cap.read()  # BGR
        if ret:
            frame = frame[115:211, 79:175]
            frame = jpeg.encode(frame)
            video.append(frame)
        else:
            break
    cap.release()

    # Pad the video with duplicate frames if it has less than 29 frames
    target_frame_count = 29
    if len(video) < target_frame_count:
        padding_frames = video[-1:] * (target_frame_count - len(video))
        video.extend(padding_frames)

    return video


def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


target_dir = '/tf/loqui/custom_lipread_pkls'
ensure_dir(target_dir)


class LRWDataset(Dataset):
    def __init__(self, mp4_path='lrw_mp4'):
        self.mp4_path = mp4_path
        with open('/tf/loqui/label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()

        self.list = []

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join(self.mp4_path, label, '*', '*.mp4'))
            for file in files:
                savefile = file.replace(self.mp4_path, target_dir).replace('.mp4', '.pkl')
                savepath = os.path.split(savefile)[0]
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

            files = sorted(files)

            self.list += [(file, i) for file in files]

    def __getitem__(self, idx):
        inputs = extract_opencv(self.list[idx][0])
        result = {}

        name = self.list[idx][0]
        labels = self.list[idx][1]

        result['video'] = inputs
        result['label'] = int(labels)
        result['duration'] = self.generate_duration(inputs)
        result['video'] = self.standardize_video_length(result['video'])

        savename = self.list[idx][0].replace('lrw_mp4', target_dir).replace('.mp4', '.pkl')
        torch.save(result, savename)

        return result

    def __len__(self):
        return len(self.list)

    def generate_duration(self, video):
        num_frames = len(video)
        duration = np.zeros(29, dtype=bool)

        if num_frames > 0:
            frame_indices = np.linspace(0, num_frames - 1, 29, dtype=int)
            duration[frame_indices] = True

        return duration

    def standardize_video_length(self, video, desired_length=29):
        num_frames = len(video)

        if num_frames < desired_length:
            # Temporal interpolation
            interpolated_video = []
            frame_indices = np.linspace(0, num_frames - 1, desired_length, dtype=int)

            for index in frame_indices:
                interpolated_video.append(video[index])

            return interpolated_video

        elif num_frames > desired_length:
            # Temporal subsampling
            subsampled_video = []
            frame_indices = np.linspace(0, num_frames - 1, desired_length, dtype=int)

            for index in frame_indices:
                subsampled_video.append(video[index])

            return subsampled_video

        return video


if __name__ == '__main__':
    loader = DataLoader(LRWDataset("/tf/Daniel/custom_lipread_mp4"),
                        batch_size=96,
                        num_workers=2,
                        shuffle=False,
                        drop_last=False)

    import time

    tic = time.time()
    for i, batch in enumerate(loader):
        toc = time.time()
        eta = ((toc - tic) / (i + 1) * (len(loader) - i)) / 3600.0
        print(f'eta:{eta:.5f}')
