import numpy as np
import torch
import time
from torch.cuda.amp import autocast
from utils import helpers
from model.lrw_dataset import LRWDataset
import warnings

# def show_confusion_matrix(true_labels, predicted_labels, class_labels):
#     """
#     Calculate and plot the confusion matrix.
#     :param true_labels: List of true labels.
#     :param predicted_labels: List of predicted labels.
#     :param class_labels: List of class labels.
#     """
#
#     # Calculate the confusion matrix
#     cm = confusion_matrix(true_labels, predicted_labels)
#
#     # Create a list of label indexes
#     label_indexes = np.arange(len(class_labels))
#
#     # Plotting the confusion matrix
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#                 xticklabels=class_labels, yticklabels=class_labels)
#     plt.xlabel("Predicted Labels")
#     plt.ylabel("True Labels")
#     plt.title("Confusion Matrix")
#     plt.xticks(label_indexes, class_labels, rotation=45, ha="right")
#     plt.yticks(label_indexes, class_labels)
#     plt.show()

@torch.no_grad()
def validation(video_model, batch_size: int, num_workers: int = 1):
    """
    Evaluate the model by validation set
    :param video_model: TODO
    :param batch_size: TODO
    :param num_workers: TODO
    :return:
    """

    dataset = LRWDataset("val", dataset_prefix="/tf/Daniel")
    print(f"Dataset object of Validation set: {dataset}, len is: {len(dataset)}")
    loader = helpers.dataset2dataloader(dataset, batch_size, num_workers, shuffle=False)

    print('start testing validation set')
    predicted_labels = []
    true_labels = []
    validation_accuracy = []

    for i_iter, sample in enumerate(loader):
        video_model.eval()

        start_time = time.time()
        video, label = helpers.prepare_data(sample)

        with autocast():
            y_v = video_model(video)

        validation_accuracy.extend((y_v.argmax(-1) == label).cpu().numpy().tolist())
        predicted_labels.extend(y_v.argmax(-1).cpu().numpy().tolist())
        true_labels.extend(label.cpu().numpy().tolist())
        end_time = time.time()

        if i_iter % 10 == 0:
            msg = helpers.add_msg('', 'v_acc={:.5f}', np.array(validation_accuracy).mean())
            msg = helpers.add_msg(msg, 'eta={:.5f}', (end_time - start_time) * (len(loader) - i_iter) / 3600.0)

            print(msg)

    accuracy = float(np.array(validation_accuracy).mean())
    accuracy_msg = f'v_acc_{accuracy:.5f}_'
    helpers.show_confusion_matrix(true_labels, predicted_labels, dataset.get_labels())

    return accuracy, accuracy_msg
