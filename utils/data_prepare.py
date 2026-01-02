import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import cv2
import torchvision
import random
import tifffile
from skimage.transform import resize
import kornia.geometry.transform as transform

class get_data(Dataset):
    def __init__(
        self,
        input_size,
        output_size,
        save_dir,
        split,
        dataset,
        downsize_coeff
    ):
        self.input_size=input_size
        self.output_size=output_size
        self.save_dir = save_dir
        self.split=split
        self.dataset = dataset
        self.downsize_coeff = downsize_coeff

        self.use_processed = True
        self.homography_matrix = torch.load("/home/ponoma/workspace/Lensless_Image_Reconstruction/data/GT2RML_homography_x4_color_detached.npy", weights_only=True)

        if self.split == 'train':
            self.patterns=np.load(self.save_dir+'train_patterns.npy')
            self.targets = np.load(self.save_dir + 'train_targets.npy')
        else:
            self.patterns = np.load(self.save_dir + 'val_patterns.npy')
            self.targets = np.load(self.save_dir + 'val_targets.npy')

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        if self.dataset == 'mirflickr':
            pattern_ = np.load(self.patterns[idx])[..., ::-1]   # using numpy images that were saved in BGR format instead of RGB
            target_ = np.load(self.targets[idx])[..., ::-1]  

            pattern = np.clip(np.flipud(pattern_)/0.9, 0,1)     # max of measurements is 0.9. Normalizing to range [0, 1]                                 
            target = np.clip(np.flipud(target_), 0,1) 
        else:
            pattern = tifffile.imread(self.patterns[idx])
            target = tifffile.imread(self.targets[idx])
            height, width, _ = pattern.shape

            if np.max(pattern) > 1.0:
                pattern = (pattern/255).astype(np.float32)     # max of measurements is 255. Normalizing to range [0, 1]
            if np.max(target) > 1.0:
                target = (target/255).astype(np.float32)

            pattern = resize(pattern, (height // self.downsize_coeff, width // self.downsize_coeff), anti_aliasing=True).astype(np.float32) 

            if not self.use_processed:
                target = resize(target, (height // self.downsize_coeff, width // self.downsize_coeff), anti_aliasing=True).astype(np.float32)
                target = apply_homography(target)

            pattern = np.clip(pattern, 0,1)
            target = np.clip(target, 0,1)

        # Randomly pics one channel to learn from
        c = random.randint(0, 2)
        pattern=pattern[:,:,c]
        target = target[:, :, c]

        return np.reshape(pattern, (1,self.input_size[0], self.input_size[1])), np.reshape(target, (1,self.output_size[0], self.output_size[1]))        # puts it channels first
    

def apply_homography(img, dataset="rml", downsize=4):
    if dataset == "rml":
        if downsize is None:
            M = torch.load("../data/efov_rml_homography_x8_FINAL_detached.npy").to(torch.float32)
            M = torch.inverse(M)
        else:
            if downsize == 4:
                M = torch.load("/home/ponoma/workspace/Lensless_Image_Reconstruction/data/GT2RML_homography_x4_color_detached.npy").to(torch.float32)
    elif dataset == "diffusercam":
        if downsize == 4:
            M = torch.load("../data/DC2GT_homography_x4_color_detached.npy")
            M = torch.inverse(M)        # because we actually want GT -> DC space
    elif dataset == 'efov':
        M = torch.load("../data/efov_rml_homography_x8_FINAL_detached.npy", weights_only=True).to(torch.float32)
        M = torch.inverse(M) 

    # Convert to tensor and ensure array is stored contiguously for faster operations
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(torch.float32)
    img = img.contiguous()

    if len(img.shape) == 3:  # If image has channels and they are in the last dimension, permute
            if img.shape[-1] == 3:
                img = img.permute(2, 0, 1)  # Change from (H,W,C) to (C,H,W)
                img = img.contiguous() # Ensure contiguous after permute
            img = img[None, ...]  # Add batch dimension
    else:
        img = img[None, None, ...]  # Add batch and channel dimensions

    warped_img = transform.warp_perspective(img.float(), M.float(), 
                                              dsize=(img.shape[2], img.shape[3])).squeeze().detach().cpu()          # last two dimensions should be height and width
    
    
    warped_img = warped_img.permute(1, 2, 0)  # switch to (H, W, C)
    warped_img = warped_img.contiguous() # Ensure contiguous after permute

    # Normalize to 0-1
    warped_img = warped_img / torch.max(warped_img)

    # Convert to numpy array and ensure C-contiguous before saving
    warped_img_np = np.ascontiguousarray(warped_img.detach().numpy())

    return warped_img_np
