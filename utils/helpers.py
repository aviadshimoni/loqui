import logging

import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.cuda.amp import autocast
import torch
import matplotlib.pyplot as plt


def parallel_model(model):
    return nn.DataParallel(model)


def load_missing(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict.keys() and v.size() == model_dict[k].size()}
    missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]

    print('loaded params/tot params:{}/{}'.format(len(pretrained_dict), len(model_dict)))
    print('miss matched params:', missed_params)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def show_lr(optimizer):
    return ','.join(['{:.6f}'.format(param_group['lr']) for param_group in optimizer.param_groups])


def collate_fn(batch):
    videos = [sample['video'] for sample in batch]
    labels = [sample['label'] for sample in batch]
    durations = [sample['duration'] for sample in batch]

    # Resize video frames to the same dimensions
    max_frame_count = max([len(video) for video in videos])
    resized_videos = []
    for video in videos:
        padding_frames = video[-1:] * (max_frame_count - len(video))
        resized_video = video + padding_frames
        resized_videos.append(resized_video)

    # Convert the resized videos to a tensor
    videos_tensor = torch.tensor(resized_videos)

    # Convert other lists to tensors
    labels_tensor = torch.tensor(labels)
    durations_tensor = torch.tensor(durations)

    return {'video': videos_tensor, 'label': labels_tensor, 'duration': durations_tensor}

def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=False,
                        collate_fn=collate_fn,
                        pin_memory=True)
    return loader




def get_prediction(video_model, video, border, is_border=False):
    """
    gets a model, a video and border. run the model over the given video and return the prediction.
    uses the border if is_border is true
    """

    if is_border:
        return video_model(video, border)
    return video_model(video)


def calculate_loss(mixup, alpha, video_model, video, label, border, is_border=False):
    loss = {}
    loss_fn = nn.CrossEntropyLoss()

    with autocast():
        if mixup:
            mixup_coef = np.random.beta(alpha, alpha)
            shuffled_indices = torch.randperm(video.size(0)).cuda(non_blocking=True)
            mixed_video = mixup_coef * video + (1 - mixup_coef) * video[shuffled_indices, :]
            mix_border = mixup_coef * border + (1 - mixup_coef) * border[shuffled_indices, :]
            mixed_label_a, mixed_label_b = label, label[shuffled_indices]
            predicted_label = get_prediction(video_model, mixed_video, mix_border, is_border=is_border)
            loss_bp = mixup_coef * loss_fn(predicted_label, mixed_label_a) + (1 - mixup_coef) * loss_fn(predicted_label,
                                                                                                        mixed_label_b)
        else:
            predicted_label = get_prediction(video_model, video, border, is_border=is_border)
            loss_bp = loss_fn(predicted_label, label)
    loss['CE V'] = loss_bp

    return loss


def prepare_data(sample: {}):
    """
    extract the relevant data from a given sample
    """

    video = sample['video'].cuda(non_blocking=True)
    label = sample['label'].cuda(non_blocking=True).long()
    border = sample['duration'].cuda(non_blocking=True).float()

    return video, label, border


def plot_train_metrics(train_losses: [], train_accuracies: [], epoch: int) -> None:
    """
    Plot the metrics of train
    :param train_losses: list of losses
    :param train_accuracies: list of accuracies
    :param epoch: number of epoch
    :return: None
    """

    print_interval = 5

    if epoch > 0 and epoch % print_interval == 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

        ax1.plot(train_losses, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss vs. Epoch')
        ax1.legend()

        ax2.plot(train_accuracies, label='Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy vs. Epoch')
        ax2.legend()

        plt.tight_layout()
        plt.show()


def add_msg(msg, k, v):
    if msg:
        msg += ','
    msg += k.format(v)

    return msg


def get_logger(name):
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(name)
    return logger
