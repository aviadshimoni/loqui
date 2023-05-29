import numpy as np
import torch
import time
from torch.cuda.amp import autocast
from utils import helpers
from model.lrw_dataset import LRWDataset


@torch.no_grad()
def validation(video_model, batch_size: int, num_workers: int = 1, is_border: bool = False):
    """
    Evaluate the model by validation set
    :param video_model: TODO
    :param batch_size: TODO
    :param num_workers: TODO
    :param is_border: TODO
    :return:
    """

    dataset = LRWDataset("val", dataset_prefix="/tf/loqui")
    print(f"Dataset object of Validation set: {dataset}, len is: {len(dataset)}")
    loader = helpers.dataset2dataloader(dataset, batch_size, num_workers, shuffle=False)

    print('start testing validation set')
    validation_accuracy = []

    for i_iter, sample in enumerate(loader):
        video_model.eval()
        video, label, border = helpers.prepare_data(sample)

        with autocast():
            predicted_label = helpers.get_prediction(video_model, video, border, is_border=is_border)

        validation_accuracy.extend((predicted_label.argmax(-1) == label).cpu().numpy().tolist())

    accuracy = float(np.array(validation_accuracy).mean())
    accuracy_msg = f'v_acc_{accuracy:.5f}_'

    return accuracy, accuracy_msg