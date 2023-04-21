import random
import numpy as np
import torch

def tensor_random_flip(tensor):
    # (b, c, t, h, w)
    if random.random() > 0.5:
        return torch.flip(tensor, dims=[4])
    return tensor


def tensor_random_crop(tensor, size):
    h, w = tensor.size(-2), tensor.size(-1)
    tw, th = size
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return tensor[..., y1:y1+th, x1:x1+tw]


def center_crop(batch_img, size):
    _, h, w = batch_img.shape
    th, tw = size
    x1 = (w - tw) // 2
    y1 = (h - th) // 2
    return batch_img[:, y1:y1+th, x1:x1+tw]


def random_crop(batch_img, size):
    _, h, w = batch_img.shape
    th, tw = size
    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    return batch_img[:, y1:y1+th, x1:x1+tw]


def horizontal_flip(batch_img):
    if random.random() > 0.5:
        batch_img = np.fliplr(batch_img)
    return batch_img
