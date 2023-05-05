# encoding: utf-8
import torch.nn as nn
from model.video_cnn import VideoCNN
from torch.cuda.amp import autocast


class VideoModel(nn.Module):
    """
    TODO add documentation
    """

    def __init__(self, n_class: int, dropout: float = 0.5, training: bool = False) -> None:
        super(VideoModel, self).__init__()
        self.gru = nn.GRU(512, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.video_cnn = VideoCNN()
        self.v_cls = nn.Linear(1024 * 2, n_class)
        self.dropout = nn.Dropout(p=dropout)
        self.training = training

    def forward(self, v):
        """
        TODO add documentation
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
        print(f"type of v inside video_model is: {v}")
        print(f"type of y_v inside video_model is: {y_v}")
        return y_v
