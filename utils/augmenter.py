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
    w, h = batch_img.shape[2], batch_img.shape[1]
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    img = batch_img[:, y1:y1 + th, x1:x1 + tw]

    return img


def horizontal_flip(batch_img):
    if random.random() > 0.5:
        batch_img = np.ascontiguousarray(batch_img[:, :, ::-1])

    return batch_img
