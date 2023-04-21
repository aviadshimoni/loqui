# encoding: utf-8
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from model.lrw_dataset import LRWDataset
from model.video_model import VideoModel


def train(batch_size: int, learning_rate: float, n_classes: int, epochs: int):
    """
    Handle the model training
    """

    dataset = LRWDataset("train")
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=1,
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

    for epoch in range(epochs):
        for i, inp in enumerate(loader):
            start_time = time.time()
            video_model.train()
            video = inp.get("video").cuda(non_blocking=True)
            label = inp.get("label").cuda(non_blocking=True)
            y_v = video_model(video)
            loss_bp = loss_fn(y_v, label)

            loss = {'CE V': loss_bp}
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

            #TODO get test info (accuracy, time etc) and update best accuracy if needed. if updated save weights
            iteration += 1

        scheduler.step()
