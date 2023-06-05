import random
import numpy as np


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


def random_brightness(batch_img, brightness_range):
    """
    Adjust the brightness of the input frames randomly within the given range.
    :param batch_img: Input batch of images (numpy array).
    :param brightness_range: Tuple (min, max) specifying the range of brightness adjustment.
    :return: Augmented batch of images.
    """
    min_brightness, max_brightness = brightness_range
    brightness_factor = random.uniform(min_brightness, max_brightness)
    batch_img = batch_img * brightness_factor
    batch_img = np.clip(batch_img, 0.0, 1.0)  # Ensure values are within [0, 1] range

    return batch_img


def horizontal_flip(batch_img):
    if random.random() > 0.5:
        batch_img = np.ascontiguousarray(batch_img[:, :, ::-1])

    return batch_img
