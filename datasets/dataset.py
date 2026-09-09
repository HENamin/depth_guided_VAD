
import os
import numpy as np
from PIL import Image
from skimage.io import imread
import cv2

import torch

from torch.utils.data import Dataset

from .transform import train_transform


def read_img_list(img_path_list, img_size=(256, 256), rgb_tags=False):
    frame_concate = []
    for img_path_iter in img_path_list:
        if rgb_tags:
            cur_frame = cv2.imread(img_path_iter)
        else:
            cur_frame = cv2.cvtColor(cv2.imread(img_path_iter), cv2.COLOR_BGR2GRAY)

        if not img_size == None:
            cur_frame = cv2.resize(cur_frame, (img_size[0], img_size[1]))

        cur_frame_np = np.array(cur_frame, dtype=np.float) / np.float(255.0)
        if len(cur_frame_np.shape) == 2:
            cur_frame_np = cur_frame_np[np.newaxis, :, :, np.newaxis]
        if len(cur_frame_np.shape) == 3:
            cur_frame_np = cur_frame_np[np.newaxis, :, :, :]
        frame_concate.append(cur_frame_np)
    frame_concate = np.concatenate(frame_concate, axis=0)
    return frame_concate


def read_original_img_list(img_path_list, img_size=(256, 256), rgb_tags=False):
    frame_concate = []
    for img_path_iter in img_path_list:
        frame_concate.append(Image.open(img_path_iter))
    return frame_concate


def video_path_list(dataset_path):
    video_list = os.listdir(dataset_path)
    video_list.sort()
    video_path_list = []
    idx = 0
    for video_path_iter in video_list:
        img_list = os.listdir(os.path.join(dataset_path, video_path_iter))
        img_list = [os.path.join(dataset_path, video_path_iter, var) for var in img_list]
        print(video_path_iter)
        print(idx, len(img_list))
        idx = idx + 1
        img_list.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        video_path_list.append(img_list)

    return video_path_list


def generate_trainfile(dataset_root_path, video_path, video_num=5, frame_interval=1):
    # path = '/mnt/data/DataSet/datasets/' + 'ped1/training/frames/'
    path = os.path.join(dataset_root_path, video_path)
    frame_path_list = video_path_list(path)
    batch_path_list = []
    for video_iter in frame_path_list:
        for frame_idx in range(0, len(video_iter) - video_num * frame_interval):
            single_batch_list = []
            for inputs_idx in range(0, video_num):
                single_batch_list.append(video_iter[frame_idx + (inputs_idx * frame_interval)])
            batch_path_list.append(single_batch_list)

    return batch_path_list


class ImageFolder(Dataset):
    def __init__(self, dataset_root_path, folder_path, video_num=5, frame_interval=1, img_size=256, rgb_tags=False):
        self.files = generate_trainfile(dataset_root_path, folder_path, video_num=video_num)
        self.img_size = img_size
        self.transform = train_transform(self.img_size, rgb_rags=rgb_tags)
        self.video_num = video_num

    def __getitem__(self, index):
        img_path = self.files[index]
        batch = torch.cat([self.transform(imread(name)) for name in img_path], 0)
        return batch

    def __len__(self):
        return len(self.files)
