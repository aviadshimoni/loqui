# encoding: utf-8
import argparse
from model.model_validation import run_validation_set
from model.model_train import train


def main():
    if is_test:
        run_validation_set()
    else:
        train(batch_size, num_workers, learning_rate, n_classes, epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="Test mode. If it is true - test once and exit", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--batch_size', help="Batch size", default=None, required=True, action=argparse.BooleanOptionalAction) # fix to get number instead of bool
    parser.add_argument('--num_workers', help="Number of workers", default=None, required=True, action=argparse.BooleanOptionalAction) # fix to get number instead of bool
    parser.add_argument('--learning_rate', help="Learning rate", default=None, required=True, action=argparse.BooleanOptionalAction) # fix to get number instead of bool
    parser.add_argument('--n_classes', help="Number of total classes", default=None, required=True, action=argparse.BooleanOptionalAction) # fix to get number instead of bool
    parser.add_argument('--epochs', help="Number of maximum epochs in training", default=None, required=True, action=argparse.BooleanOptionalAction) # fix to get number instead of bool
    args = parser.parse_args()
    is_test = vars(args).get("test")
    batch_size = vars(args).get("batch_size")
    num_workers = vars(args).get("num_workers")
    learning_rate = vars(args).get("learning_rate")
    n_classes = vars(args).get("n_classes")
    epochs = vars(args).get("epochs")
    main()
