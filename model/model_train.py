# encoding: utf-8
import time
import utils.utils
import torch
import torch.nn as nn
from os.path import join, dirname, realpath
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from model.lrw_dataset import LRWDataset
from model.video_model import VideoModel
from model.model_validation import run_validation_set


def parallel_model(model: VideoModel) -> nn.DataParallel:
    """
    Parallelize the application of the given module
    """

    model = nn.DataParallel(model)
    return model


def train(batch_size: int, num_workers: int, learning_rate: float, n_classes: int, epochs: int,
          save_weights_prefix: str = dirname(realpath(__file__))):
    """
    Handle the model training
    """

    dataset = LRWDataset("train", dataset_prefix="/tf/rois/")
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False,
                        pin_memory=True)
    iteration = 0
    best_accuracy = 0
    grad_scaler = GradScaler()
    video_model = VideoModel(n_classes, training=True).cuda()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = batch_size / 32.0 / torch.cuda.device_count() * learning_rate
    optimizer = torch.optim.Adam(video_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
    video_model = parallel_model(video_model)

    for epoch in range(epochs):
        for i, inp in enumerate(loader):
            start_time = time.time()
            video_model.train()
            video = inp.get("video").cuda(non_blocking=True)
            label = inp.get("label").cuda(non_blocking=True)
            y_v = video_model(video)
            loss_bp = loss_fn(y_v, label)

            loss = {"CE V": loss_bp}
            optimizer.zero_grad()
            grad_scaler.scale(loss_bp).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            end_time = time.time()

            summary = f"Epoch: {epoch}, Train iteration: {iteration}, ETA: " \
                      f"{(end_time - start_time) * (len(loader) - i) / 3600.0}, " \
                      f"Best accuracy: {best_accuracy}"

            for k, v in loss.items():
                summary += f", {k}: {v}"

            print(summary)

            # if we are at the end of an epoch or at the very first start run validation set
            if i == len(loader) - 1 or epoch == 0 and i == 0:
                accuracy = run_validation_set(video_model, batch_size, num_workers)

                if accuracy > best_accuracy:
                    save_name = join(save_weights_prefix, f"train_iter_{iteration}_epoch_{epoch}_acc_{accuracy}.pt")
                    utils.utils.ensure_dir(save_weights_prefix)
                    torch.save({"video_model": video_model.module.state_dict()}, save_name)
                    best_accuracy = accuracy

            iteration += 1

        scheduler.step()
