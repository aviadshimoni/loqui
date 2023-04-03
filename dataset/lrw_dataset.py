# encoding: utf-8
import os
import glob
import torch
import numpy as np
import utils.system_globals as sg
import utils.augmenter as data_augmenter
from turbojpeg import TJPF_GRAY
from lrw_dataset_interface import LRWDatasetInterface


class LRWDataset(LRWDatasetInterface):
    """
    Object that represents the LRW dataset.
    Inherits LRWDatasetInterface which implements Dataset of torch module
    and being used when training or testing the model
    """

    def __init__(self, phase, args, labels_path: str = "label_sorted.txt") -> None:
        self.phase = phase  # train/val/test
        self.labels_path = labels_path
        self.labels = self.set_labels()
        self.list = self.append_files()
        self.args = args  # TODO: find out wtf is this

        # TODO: find out wtf is this
        if not hasattr(self.args, "is_aug"):
            setattr(self.args, "is_aug", True)

    def set_labels(self) -> list:
        """
        reads the labels file
        :return: a list of labels
        """

        with open(self.labels_path) as f:
            return f.read().splitlines()

    def append_files(self) -> list:
        """
        for each label, collect the pickle files under the corresponding phase
        :return: a list of ROIS
        """

        lst = []

        for i, label in enumerate(self.labels):
            files = glob.glob(os.path.join("lrw_roi_npy_gray_pkl_jpeg", label, self.phase, "*.pkl"))
            files = sorted(files)

            lst += [file for file in files]

        return lst

    def __getitem__(self, idx: int) -> dict:
        """
        implements the operator []
        :param idx: index to return of the dataset object
        :return: by given index, return the respectively data on that index
        """

        tensor = torch.load(self.list[idx])
        inputs = tensor.get("video")
        inputs = [sg.jpeg.decode(img, pixel_format=TJPF_GRAY) for img in inputs]
        inputs = np.stack(inputs, 0) / 255.0
        inputs = inputs[:, :, :, 0]

        # TODO check what type inputs is and update the augmentation functions documentations
        if self.phase == "train":
            batch_img = data_augmenter.random_crop(inputs, (88, 88))
            batch_img = data_augmenter.horizontal_flip(batch_img)
        else:  # phase in ["val", "test"]
            batch_img = data_augmenter.center_crop(inputs, (88, 88))

        result = {"video": torch.FloatTensor(batch_img[:, np.newaxis, ...]),
                  "label": tensor.get("label"),
                  "duration": 1.0 * tensor.get("duration")}
        # print(result["video"].size())

        return result

    def __len__(self) -> int:
        """
        implements the len operator
        :return: len of self.list
        """

        return len(self.list)
