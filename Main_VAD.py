import torch
import torch.nn as nn

import sys
import os
import glob
import numpy as np
from sklearn.cluster import KMeans
import scipy.io as scio
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.dataset import ImageFolder
from datasets.eval_dataset import sliding_whole_dataset

from Solver_Pred import Solver
from Model import MM_Model

import config

RANDOM_SEED = config.seed
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


def train(args):
    config.log_path = os.path.join(args.log_path, args.dataset_name)

    config.dataset_path = args.dataset_path
    config.dataset_name = args.dataset_name
    config.folder_path = args.folder_path
    #
    config.compact = args.compact

    # Initialize solver with resume flag
    solver = Solver(config, Model=MM_Model)

    train_set = ImageFolder(dataset_root_path=config.dataset_path, folder_path=config.folder_path,
                            video_num=config.clips_length, frame_interval=config.frame_interval,
                            img_size=config.img_size, rgb_tags=config.rgb_tags)
    training_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True)
    training_iter = iter(training_loader)
    training_nums = len(training_loader)

    # images = next(training_iter)  # training_iter.next()

    config.pretrain_batches = training_nums * 200
    ts_idx = 0
    # If resuming, start from the saved iteration
    start_iter = 0
    if args.resume:
        solver.load_model(eval_only=False)
        start_iter = solver.start_epoch
        print(f"Resuming from iteration {start_iter}")

    if args.pretrain_tag:
        for iter_idx in range(start_iter, config.pretrain_batches):
            if (ts_idx + 2) >= training_nums:
                training_iter = iter(training_loader)
                ts_idx = 0
            train_batch = next(training_iter)
            ts_idx += 1
            solver.train_batch_AE(train_batch)
            if (iter_idx > (config.pretrain_batches // 4)) and config.compact:
                solver.train_batch_cmp(train_batch)

            if (iter_idx + 1) % 500 == 0:
                solver.training_info('Epoches: idx - {} '.format(iter_idx + 1))
            if ((iter_idx + 1) % 50000 == 0) or ((iter_idx + 1) == config.pretrain_batches):
                solver.save_model(iter_idx + 1)
    return


def eval_model(dataset_path, dataset_name, log_dir, arg):
    config.log_path = os.path.join(log_dir, dataset_name)

    config.dataset_path = dataset_path
    config.dataset_name = dataset_name

    solver = Solver(config, Model=MM_Model)
    solver.load_model(eval_only=True)  # Modified to use eval_only parameter
    eval_loader, eval_labels = sliding_whole_dataset(config).generate_video_sequence()
    solver.para_tag = False
    solver.eval_datasets(eval_loader, eval_labels, 0)

    return


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    current_dir = Path.cwd()
    print(f"You are currently in: {current_dir}")

    data_name = 'avenue'  # 'SUT'
    # folder_path = "SUT_data\\Train\\" # 'UCSDped2\\Train\\'
    # test_folder = "SUT_data\\Test\\"# "UCSDped2\\Test"
    folder_path = "Avenue\\Avenue_Dataset\\Train"
    test_folder = "Avenue\\Avenue_Dataset\\Test"
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=int, default=0, help="selected gpu idx")
    parser.add_argument('--dataset_name', type=str, default=data_name, choices=['ped2', 'avenue'],
                        help="selected datasets")
    parser.add_argument('--dataset_path', type=str, default=current_dir, help="datasets root path ")
    parser.add_argument('--folder_path', type=str, default=folder_path, help="data frames folder path")
    parser.add_argument('--Test_Folder', type=str, default=test_folder, help="data frames test folder path")
    parser.add_argument('--log_path', type=str, default='./log', help="log dir ")

    parser.add_argument('--pretrain_tag', type=int, default=1, help="pre train ae model")

    parser.add_argument('--eval', type=int, default=0, help="evaluation")

    parser.add_argument('--compact', type=int, default=1, help="compact mode")

    # Add resume argument
    parser.add_argument('--resume', type=int, default=0,
                        help="resume training from checkpoint (0: start new, 1: resume)")

    args = parser.parse_args()

    # Set GPU device if available
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    if args.eval:
        eval_model(args.dataset_path, args.dataset_name, args.log_path, args)
    else:
        # Print training mode - convert int to bool
        print(f"{'Resuming' if args.resume == 1 else 'Starting new'} training...")
        # Pass bool to Solver
        train(args)
