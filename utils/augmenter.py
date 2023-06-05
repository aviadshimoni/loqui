import random
import numpy as np
from spicy import ndimage


def center_crop(batch_img, size):
    w, h = batch_img.shape[2], batch_img.shape[1]
    th, tw = size
    x1 = int(round((w - tw)) / 2.)
    y1 = int(round((h - th)) / 2.)
    img = batch_img[:, y1:y1 + th, x1:x1 + tw]

    return img


def random_crop(batch_img, size):
    th, tw = size
    x1 = random.randint(0, 8)
    y1 = random.randint(0, 8)
    img = batch_img[:, y1:y1 + th, x1:x1 + tw]

    return img


def add_gaussian_noise(batch_img, mean, std):
    """
    Add random Gaussian noise to the input batch of images.
    :param batch_img: Input batch of images (numpy array).
    :param mean: Mean of the Gaussian distribution.
    :param std: Standard deviation of the Gaussian distribution.
    :return: Augmented batch of images.
    """
    noise = np.random.normal(mean, std, size=batch_img.shape)
    noisy_batch_img = batch_img + noise
    noisy_batch_img = np.clip(noisy_batch_img, 0.0, 1.0)  # Ensure values are within [0, 1] range

    return noisy_batch_img


def random_scale(batch_img, scale_range):
    scale_factor = random.uniform(scale_range[0], scale_range[1])
    scaled_img = ndimage.zoom(batch_img, scale_factor, mode='nearest')
    return scaled_img


def horizontal_flip(batch_img):
    if random.random() > 0.5:
        batch_img = np.ascontiguousarray(batch_img[:, :, ::-1])

    return batch_img
