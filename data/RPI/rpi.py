import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class RPI(Dataset):
    """
    Dataset class for RPI (AAPM) data.
    Loads data from two separate folders (A/B structure).
    - hq_dir: Folder with clean images (Ground Truth)
    - lq_dir: Folder with artifact images (Metal Artifact)
    
    This dataset implements UNPAIRED learning strategy for ADN.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.opt = kwargs
        
        # Default image size for the model (can be adjusted via config if needed)
        self.img_size = 256 
        
        # Load separate paths from configuration
        self.hq_root = self.opt['hq_dir']
        self.lq_root = self.opt['lq_dir']
        
        # Recursively find all .png images in both folders
        self.hq_paths = sorted(glob.glob(os.path.join(self.hq_root, '**', '*.png'), recursive=True))
        self.lq_paths = sorted(glob.glob(os.path.join(self.lq_root, '**', '*.png'), recursive=True))
        
        self.hq_size = len(self.hq_paths)
        self.lq_size = len(self.lq_paths)

        print(f"Found {self.hq_size} HQ (Clean) images in: {self.hq_root}")
        print(f"Found {self.lq_size} LQ (Artifact) images in: {self.lq_root}")

        if self.hq_size == 0 or self.lq_size == 0:
            raise FileNotFoundError(f"No .png images found in {self.hq_root} or {self.lq_root}. "
                                    f"Check 'hq_dir' and 'lq_dir' paths.")

        # Transform: 16-bit [0, 65535] -> float [0, 1] -> tensor [-1, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # Normalize [0, 1] to [-1, 1]
        ])

    def _load_image_16bit(self, path):
        """Loads a 16-bit PNG image and normalizes it to [0, 1]."""
        # Load as-is (16-bit)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot load image: {path}")
            
        # Resize to target size (e.g., 256x256) if dimensions differ
        # AAPM raw data is usually 512x512, but models often expect 256x256
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), 
                             interpolation=cv2.INTER_LINEAR)
            
        # Convert to float32 and normalize from [0, 65535] to [0, 1]
        img = img.astype(np.float32) / 65535.0
        
        # Add channel dimension: (H, W) -> (H, W, C)
        img = np.expand_dims(img, axis=-1)
        return img
    
    def __getitem__(self, index):
        # ADN is trained on UNPAIRED data.
        # We fetch one HQ image deterministically and one LQ image randomly.
        
        # --- Get High-Quality (Clean) Image ---
        # Use modulo to loop over the dataset if index exceeds size
        hq_path = self.hq_paths[index % self.hq_size]
        hq_img = self._load_image_16bit(hq_path)
        hq_img_tensor = self.transform(hq_img)

        # --- Get Low-Quality (Artifact) Image ---
        # Pick a RANDOM LQ image to ensure no pairing (disentanglement)
        lq_index = random.randint(0, self.lq_size - 1)
        lq_path = self.lq_paths[lq_index]
        lq_img = self._load_image_16bit(lq_path)
        lq_img_tensor = self.transform(lq_img)

        # Return the dictionary structure expected by train.py
        data = {
            'lq_image': lq_img_tensor,
            'hq_image': hq_img_tensor,
            'mask': torch.empty(0),  # Empty tensor as masks are not used here
            'lq_path': lq_path,
            'hq_path': hq_path
        }
        
        return data

    def __len__(self):
        """Returns the size of the larger subset to ensure a full epoch coverage."""
        return max(self.hq_size, self.lq_size)