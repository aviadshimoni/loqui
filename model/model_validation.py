# encoding: utf-8
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.lrw_dataset import LRWDataset
from model.video_model import VideoModel


def run_validation_set(video_model: VideoModel, batch_size: int, num_workers: int):
    """
    Evaluate the model by validation set
    """

    with torch.no_grad():
        dataset = LRWDataset("val", dataset_prefix="/tf/rois/")
        print(f"dataset object of validation set: {dataset}")
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True)
        print(f"loader length: {len(loader)}")
        validation_accuracy = []

        for i, inp in enumerate(loader):
            start_time = time.time()
            video_model.eval()
            video = inp.get("video").cuda(non_blocking=True)
            label = inp.get("label").cuda(non_blocking=True)
            print(f"current label: {label}")
            total = total + video.size(0)
            y_v = video_model(video)
            y_v_acc = (y_v.argmax(-1) == label).cpu().numpy().tolist()
            print(f"y_v_acc that we extend to validation accuracy: {y_v_acc}")
            validation_accuracy.extend(y_v_acc)
            end_time = time.time()

            if i % 10 == 0:
                summary = f"Validation accuracy: {np.array(validation_accuracy).reshape(-1).mean()}, " \
                          f"ETA: {(end_time - start_time) * (len(loader) - i) / 3600.0}"
                print(summary)

        accuracy = np.array(validation_accuracy).reshape(-1).mean()
        summary = f"Validation accuracy: {np.array(validation_accuracy).reshape(-1).mean()}"
        print(summary)

        return accuracy
