import pandas as pd
from glob import glob
import cv2
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from piq import ssim

# Define the paths directly
gt_frames_folder = '/home/host/pegah/evl/ER-NeRF/frames/Shaheen/GT'
pd_frames_folder = '/home/host/pegah/evl/ER-NeRF/frames/Shaheen/GHu'

# Function to read images from folder
def read_images_from_folder(folder_path, to_rgb=False, to_gray=False, to_nchw=False):
    image_paths = sorted(glob(osp.join(folder_path, '*.png')))  # Assumes frames are PNG, adjust if needed
    frames = []
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if to_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    frames = np.stack(frames)
    if to_nchw:
        frames = np.transpose(frames, (0, 3, 1, 2))  # NCHW format for PyTorch
    return frames

# Run the code in Jupyter
ssim_values = []

# Read frames from both folders
gt_frames = read_images_from_folder(gt_frames_folder, True, False, True)
pd_frames = read_images_from_folder(pd_frames_folder, True, False, True)

# Convert frames to PyTorch tensors and normalize to [0, 1]
gt_frames = torch.from_numpy(gt_frames).float() / 255.
pd_frames = torch.from_numpy(pd_frames).float() / 255.

# Ensure both sets have the same number of frames
assert gt_frames.shape == pd_frames.shape, "The number of frames in both folders should be the same."

# Calculate SSIM for each frame pair
for i in tqdm(range(gt_frames.shape[0])):
    ssim_value = ssim(pd_frames[i].unsqueeze(0), gt_frames[i].unsqueeze(0), data_range=1.)
    ssim_values.append(ssim_value.item())

# Print mean SSIM value for all frames
print('Mean SSIM:', np.mean(ssim_values))
