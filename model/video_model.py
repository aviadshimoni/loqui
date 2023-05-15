# encoding: utf-8
import torch
import torch.nn as nn
from model.video_cnn import VideoCNN
from torch.cuda.amp import autocast


class VideoModel(nn.Module):
    """
    The VideoModel class is a PyTorch nn.Module that consists of a video CNN and a GRU network.
    The VideoModel class takes as input a batch of videos,
    and outputs a tensor of shape (batch_size, n_class) containing the logits for each video in the batch.
    """

    def __init__(self, n_class: int, dropout: float = 0.5, training: bool = False) -> None:
        super(VideoModel, self).__init__()
        self.gru = nn.GRU(512, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.video_cnn = VideoCNN()
        self.v_cls = nn.Linear(1024 * 2, n_class)
        self.dropout = nn.Dropout(p=dropout)
        self.training = training

    def forward(self, v: torch.Size) -> torch.Size:
        """
        The output of the CNN is fed into a bidirectional GRU network with 1024 hidden units per direction, and 3 layers.
        The GRU network outputs a tensor of shape (batch_size, seq_len, 2048).
        The output of the GRU network is then fed into a linear layer with n_class output units,
         which produces a tensor of shape (batch_size, n_class) containing the logits for each video in the batch.
        """

        self.gru.flatten_parameters()

        if self.training:
            with autocast():
                f_v = self.video_cnn(v)
                f_v = self.dropout(f_v)
            f_v = f_v.float()
        else:
            f_v = self.video_cnn(v)
            f_v = self.dropout(f_v)

        h, _ = self.gru(f_v)
        y_v = self.v_cls(self.dropout(h)).mean(1)

        #return y_v
        return y_v

