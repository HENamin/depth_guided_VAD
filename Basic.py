import torch
import torch.nn as nn
import numpy as np


def video_frames(frames, img_channel, frames_num):
    """0-th frame for appearance feature 1-th to last frames for motion feature"""

    motion_in = frames[:, 0 * img_channel:(frames_num - 1) * img_channel, :, :]  # frames 0 to 3 for motion
    static_in = frames[:, 0 * img_channel:1 * img_channel, :, :]  # first frame for spatial features
    motion_target = frames[:, (frames_num - 1) * img_channel:, :, :]  # - static_in   # last frame - first frame
    static_target = static_in

    return static_in, motion_in, static_target, motion_target
