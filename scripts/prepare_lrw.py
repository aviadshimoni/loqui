import os
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import cv2
import numpy as np
from turbojpeg import TurboJPEG
from utils.helpers import extract_opencv
jpeg = TurboJPEG()


target_dir = '/tf/Daniel/pkls_test' # REPLACE ME

if not os.path.exists(target_dir):
    os.makedirs(target_dir)


class LRWDataset(Dataset):
    def __init__(self, mp4_path='lrw_mp4'):
        self.mp4_path = mp4_path
        with open('label_sorted.txt') as myfile:
            self.labels = myfile.read().splitlines()

        self.data = []

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join(self.mp4_path, label, '*', '*.mp4'))
            for file in files:
                savefile = file.replace(self.mp4_path, target_dir).replace('.mp4', '.pkl')
                savepath = os.path.split(savefile)[0]
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

            files = sorted(files)

            self.data += [(file, i) for file in files]

    def __getitem__(self, idx: int) -> dict:
        video_path, label = self.data[idx]
        inputs = extract_opencv(video_path)

        metadata_file = video_path.replace('.mp4', '.txt')

        if os.path.isfile(metadata_file):
            duration = self.load_duration(metadata_file)
            print(f"duration lrw mp4: {duration}")
        else:
            duration = np.array([0.0] * 3 + [1.0] * 23 + [0.0] * 3)  # Update duration values

        result = {
            'video': inputs,
            'label': int(label),
            'duration': duration
        }

        output = video_path.replace(self.mp4_path, target_dir).replace('.mp4', '.pkl')
        torch.save(result, output)

        return result

    def __len__(self):
        return len(self.data)

    def load_duration(self, file):
        with open(file, 'r') as f:
            for line in f:
                if 'Duration' in line:
                    duration = float(line.split(' ')[1])
                    break

        tensor = np.zeros(29)
        mid = 29 / 2
        start = int(mid - duration / 2 * 25)
        end = int(mid + duration / 2 * 25)
        tensor[start:end] = 1.0
        return tensor


if __name__ == '__main__':
    loader = DataLoader(LRWDataset("/tf/Daniel/lipread_mp4"),
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