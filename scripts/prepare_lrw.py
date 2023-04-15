# encoding: utf-8
import time
from torch.utils.data import DataLoader
from dataset.lrw_dataset_preparation import LRWDatasetPreparation


def main():
    dataset = LRWDatasetPreparation()
    loader = DataLoader(dataset,
                        batch_size=1,
                        num_workers=8,
                        shuffle=False)

    start = time.time()

    for i, batch in enumerate(loader):
        end = time.time()
        eta = ((end - start) / (i + 1) * (len(loader) - i)) / 3600.0
        print(f"eta:{eta:.5f}")


if __name__ == '__main__':
    main()
