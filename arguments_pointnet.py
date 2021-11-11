import argparse
import socket
import os
from dex_envs.configs import *

"""
Here are the param for the training

"""


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=128, help='number of points to sample in mujoco')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--n_epoch', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--expID', type=int, default=0)
    parser.add_argument('--seed', type=int, default=125)
    parser.add_argument('--train_names', nargs='*', default=[],
                        type=str, help='the environment name')
    parser.add_argument('--test_names', nargs='*', default=[],
                        type=str, help='the environment name')
    parser.add_argument('--save_dir', type=str,
                        default='dex_logs/', help='the path to save the models')
    parser.add_argument('--std_data_aug', type=float,
                        default=0.00384, help="data augmentation noise std")
    parser.add_argument('--alpha', default=0.459, type=float,
                        help="coefficient for classification and relative rotation estimation")
    parser.add_argument('--no_save', action="store_true",
                        help="save model at each iter or not")
    parser.add_argument('--lr', type=float, default=0.0004086)
    parser.add_argument('--output_dim', type=int, default=512,
                        help='bottleneck dim of Pointnet; claimed to be important by the paper')

    args = parser.parse_args()

    # default to use train/test split specified in dex_envs/configs
    args.train_names = args.train_names if args.train_names else ALL_TRAIN
    args.test_names = args.test_names if args.test_names else ALL_TEST
    assert len(list(set(args.train_names) & set(args.test_names))
               ) == 0, 'cannot have overlapping train/test envs'

    return args
