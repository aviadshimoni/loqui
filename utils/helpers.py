import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.cuda.amp import autocast
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import logging


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


def dataset2dataloader(dataset, batch_size, num_workers, shuffle=True):
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle,
                        drop_last=False,
                        pin_memory=True)
    return loader


def calculate_loss(mixup, alpha, video_model, video, label):
    loss = {}
    loss_fn = nn.CrossEntropyLoss()
    with autocast():
        if mixup:
            mixup_coef = np.random.beta(alpha, alpha)
            shuffled_indices = torch.randperm(video.size(0)).cuda(non_blocking=True)
            mixed_video = mixup_coef * video + (1 - mixup_coef) * video[shuffled_indices, :]
            mixed_label_a, mixed_label_b = label, label[shuffled_indices]
            predicted_label = video_model(mixed_video)
            loss_bp = mixup_coef * loss_fn(predicted_label, mixed_label_a) + (1 - mixup_coef) * loss_fn(predicted_label,
                                                                                                        mixed_label_b)
        else:
            predicted_label = video_model(video)
            loss_bp = loss_fn(predicted_label, label)
    loss['CE V'] = loss_bp

    return loss


def prepare_data(sample: {}):
    video = sample['video'].cuda(non_blocking=True)
    label = sample['label'].cuda(non_blocking=True).long()

    return video, label


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


def show_confusion_matrix(true_labels, predicted_labels, class_labels):
    """
    Calculate and plot the confusion matrix.
    :param true_labels: List of true labels.
    :param predicted_labels: List of predicted labels.
    :param class_labels: List of class labels.
    """

    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Create a list of label indexes
    label_indexes = np.arange(len(class_labels))

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.xticks(label_indexes, class_labels, rotation=45, ha="right")
    plt.yticks(label_indexes, class_labels)
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