import argparse
import torch
import os
from model.model_train import train

torch.backends.cudnn.benchmark = True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    def str2bool(v: str) -> bool:
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--n_class', type=int, required=True)
    parser.add_argument('--max_epoch', type=int, required=True)
    parser.add_argument('--test', type=str2bool, required=True)
    parser.add_argument('--num_workers', type=int, required=False, default=1)
    parser.add_argument('--gpus', type=str, required=False, default='0')
    parser.add_argument('--weights', type=str, required=False, default=None)
    parser.add_argument('--save_prefix', type=str, required=True)
    parser.add_argument('--mixup', type=str2bool, required=False, default=False)
    parser.add_argument('--border', type=str2bool, required=True)
    parser.add_argument('--dataset', type=str, required=False, default='lrw')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    return args


if __name__ == '__main__':
    args = parse_arguments()

    if args.test:
        # TODO trigger test-single-video notebook and run a prediction,
        #  perhaps we can remove test completely because we run single prediction from ui
        exit()

    train(args.lr, args.batch_size, args.n_class, args.max_epoch, args.num_workers, args.gpus, weights=args.weights,
          save_prefix=args.save_prefix, mixup=args.mixup, is_border=args.border)
