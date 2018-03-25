"""main.py"""
import argparse

import torch
import numpy as np

from solver import BEGAN
from utils import str2bool


def main(args):

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    net = BEGAN(args)
    net.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BEGAN')

    # Optimization
    parser.add_argument('--epoch', default=20, type=int, help='epoch size')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--D_lr', default=1e-4, type=float, help='learning rate for the Discriminator')
    parser.add_argument('--G_lr', default=1e-4, type=float, help='learning rate for the Generator')
    parser.add_argument('--gamma', default=0.5, type=float, help='equilibrium balance ratio')
    parser.add_argument('--lambda_k', default=0.001, type=float, help='the proportional gain of k')

    # Network
    parser.add_argument('--model_type', default='skip_repeat', type=str, help='three types of models : simple, skip, skip_repeat')
    parser.add_argument('--n_filter', default=64, type=int, help='scaling unit of the number of filters')
    parser.add_argument('--n_repeat', default=2, type=int, help='repetition number of network layers')
    parser.add_argument('--hidden_dim', default=64, type=int, help='dimension of the hidden state')
    parser.add_argument('--load_ckpt', default=True, type=str2bool, help='load previous checkpoint')
    parser.add_argument('--ckpt_dir', default='checkpoint', type=str, help='weight directory')
    parser.add_argument('--image_size', default=32, type=int, help='image size')

    # Dataset
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='CIFAR10, CelebA')
    parser.add_argument('--num_workers', default=1, type=int, help='num_workers for data loader')

    # Visualization
    parser.add_argument('--env_name', default='main', type=str, help='experiment name')
    parser.add_argument('--visdom', default=True, type=str2bool, help='enable visdom')
    parser.add_argument('--port', default=8097, type=int, help='visdom port number')
    parser.add_argument('--timestep', default=50, type=int, help='visdom curve time step')
    parser.add_argument('--output_dir', default='output', type=str, help='image output directory')

    # Misc
    parser.add_argument('--seed', default=1, type=int, help='random seed number')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')

    args = parser.parse_args()

    main(args)
