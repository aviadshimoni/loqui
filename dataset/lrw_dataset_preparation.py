# encoding: utf-8
import os
import glob
import torch
import numpy as np
import utils.utils as utils
from lrw_dataset_interface import LRWDatasetInterface


class LRWDatasetPreparation(LRWDatasetInterface):
    """
    Object that represents the preparation process of LRW dataset.
    Inherits LRWDatasetInterface which implements Dataset of torch module
    and generates training samples of LRW
    """

    def __init__(self, labels_path: str = "label_sorted.txt",
                 target_dir: str = "lrw_roi__npy_gray_pkl_jpeg",
                 ensure_paths: bool = True) -> None:
        self.labels_path = labels_path
        self.target_dir = target_dir
        self.labels = self.set_labels()
        self.list = self.append_files()
        self.ensure_paths = ensure_paths

        if self.ensure_paths:
            utils.ensure_dir(target_dir)

    def set_labels(self) -> list:
        """
        reads the labels file
        :return: a list of labels
        """

        with open(self.labels_path) as f:
            return f.read().splitlines()

    def append_files(self) -> list:
        """
        for each label, iterate over all of its examples and save a pickle for each example.
        :return: a list which each element is a tuple made of examples and their number of label
        """

        lst = []

        for i, label in enumerate(self.labels):
            files = glob.glob(os.path.join("lrw_mp4", label, '*', "*.mp4"))
            for file in files:
                file_to_save = file.replace("lrw_mp4", self.target_dir).replace(".mp4", ".pkl")
                save_path = os.path.split(file_to_save)[0]
                if self.ensure_paths:
                    utils.ensure_dir(save_path)

            files = sorted(files)

            lst += [(file, i) for file in files]

        return lst

    def __getitem__(self, idx: int) -> dict:
        """
        implements the operator []
        :param idx: index to return of the dataset object
        :return: by given index, return the respectively data on that index
        """

        inputs = utils.extract_opencv(self.list[idx][0])

        duration = self.list[idx][0]
        labels = self.list[idx][1]
        result = {"video": inputs,
                  "label": int(labels),
                  "duration": utils.load_duration(duration.replace(".mp4", ".txt")).astype(np.bool)}

        output = self.list[idx][0].replace("lrw_mp4", self.target_dir).replace(".mp4", ".pkl")
        torch.save(result, output)

        return result

    def __len__(self) -> int:
        """
        implements the len operator
        :return: len of self.list
        """

        return len(self.list)
