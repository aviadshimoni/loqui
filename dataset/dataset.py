import numpy as np
import glob
from cv_transforms import center_crop, random_crop, horizontal_flip
import os
from keras.utils import Sequence
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE

jpeg = TurboJPEG()


class LRWDataset(Sequence):
    def __init__(self, phase, args, batch_size=1):

        with open('label_sorted.txt') as labels_file:
            self.labels = labels_file.read().splitlines()

        self.list = []
        self.phase = phase
        self.args = args
        self.batch_size = batch_size

        if not hasattr(self.args, 'is_aug'):
            setattr(self.args, 'is_aug', True)

        for (i, label) in enumerate(self.labels):
            files = glob.glob(os.path.join('lrw_roi_npy_gray_pkl_jpeg', label, phase, '*.pkl'))
            files = sorted(files)

            self.list += [file for file in files]

    def __getitem__(self, idx):

        batch_x = []
        batch_y = []
        batch_duration = []

        for i in range(self.batch_size):
            tensor = np.load(self.list[idx * self.batch_size + i])
            inputs = tensor.get('video')
            inputs = [jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
            inputs = np.stack(inputs, 0) / 255.0
            inputs = inputs[:, :, :, 0]

            if self.phase == 'train':
                inputs = random_crop(inputs, (88, 88))
                inputs = horizontal_flip(inputs)
            elif self.phase == 'val' or self.phase == 'test':
                inputs = center_crop(inputs, (88, 88))

            batch_x.append(inputs[np.newaxis, ...])
            batch_y.append(tensor.get('label'))
            batch_duration.append(1.0 * tensor.get('duration'))

        return np.concatenate(batch_x, axis=0), np.array(batch_y), np.array(batch_duration)

    def __len__(self):
        return int(np.ceil(len(self.list) / self.batch_size))
