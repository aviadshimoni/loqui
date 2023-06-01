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

        try:
            result['video'] = inputs
            result['label'] = int(labels)
            result['duration'] = self.load_duration(self.list[idx][0]).astype(np.bool_)
            result['video'] = self.standardize_video_length(result['video'])

        except Exception as e:
            print(f"Error processing video: {name}")
            print(f"Error message: {str(e)}")
            return None

        savename = self.list[idx][0].replace('lrw_mp4', target_dir).replace('.mp4', '.pkl')
        torch.save(result, savename)

        return result

    def __len__(self):
        return len(self.list)

    def load_duration(self, file):
        txt_file = file.replace('.mp4', '.txt')
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if line.find('Duration') != -1:
                        duration = float(line.split(' ')[1])
                        return np.full(29, True, dtype=bool)

        return self.generate_duration(extract_opencv(file))

    def generate_duration(self, video):
        num_frames = len(video)
        duration = np.zeros(29, dtype=bool)

        if num_frames > 0:
            frame_indices = np.linspace(0, num_frames - 1, min(29, num_frames), dtype=int)
            duration[frame_indices] = True

        return duration

    def standardize_video_length(self, video, desired_length=29):
        num_frames = len(video)

        if num_frames < desired_length:
            # Temporal interpolation
            interpolated_video = []
            frame_indices = np.linspace(0, num_frames - 1, desired_length, endpoint=True)

            for index in frame_indices:
                interpolated_video.append(video[int(index)])

            return interpolated_video

        elif num_frames > desired_length:
            # Temporal subsampling
            subsampled_video = []
            frame_indices = np.linspace(0, num_frames - 1, desired_length, endpoint=True)

            for index in frame_indices:
                subsampled_video.append(video[int(index)])

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
