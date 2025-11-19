import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DeepLesion(Dataset):
    """
    PyTorch Dataset for loading unpaired high-quality (HQ) and low-quality (LQ)
    16-bit PNG medical images.

    - `hq_dir`: directory containing clean (artifact-free) images
    - `lq_dir`: directory containing degraded images with artifacts
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.opt = kwargs
        self.img_size = 256

        # Directories for HQ and LQ datasets
        self.hq_root = self.opt['hq_dir']
        self.lq_root = self.opt['lq_dir']

        # Locate PNG images recursively
        self.hq_paths = sorted(glob.glob(os.path.join(self.hq_root, '**', '*.png'), recursive=True))
        self.lq_paths = sorted(glob.glob(os.path.join(self.lq_root, '**', '*.png'), recursive=True))

        self.hq_size = len(self.hq_paths)
        self.lq_size = len(self.lq_paths)

        print(f"Found {self.hq_size} HQ images in: {self.hq_root}")
        print(f"Found {self.lq_size} LQ images in: {self.lq_root}")

        if self.hq_size == 0 or self.lq_size == 0:
            raise FileNotFoundError(
                f"No PNG images found in '{self.hq_root}' or '{self.lq_root}'. "
                f"Please verify 'hq_dir' and 'lq_dir' paths in the configuration file."
            )

        # Transform: 16-bit [0,65535] → float [0,1] → normalized tensor [-1,1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _load_image_16bit(self, path):
        """
        Loads a 16-bit PNG image, resizes it to the target resolution,
        normalizes it into [0,1] range, and converts to single-channel format.
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Failed to read image: {path}")

        # Ensure the desired spatial resolution
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Convert to float32 and normalize
        img = img.astype(np.float32) / 65535.0

        # Convert from (H, W) to (H, W, 1)
        img = np.expand_dims(img, axis=-1)

        return img

    def __getitem__(self, index):
        """
        Returns an unpaired sample:
        - HQ image (deterministic, indexed)
        - LQ image (random)
        """
        # Load HQ image
        hq_path = self.hq_paths[index % self.hq_size]
        hq_img = self._load_image_16bit(hq_path)
        hq_img_tensor = self.transform(hq_img)

        # Load LQ image (random to ensure unpaired sampling)
        lq_index = random.randint(0, self.lq_size - 1)
        lq_path = self.lq_paths[lq_index]
        lq_img = self._load_image_16bit(lq_path)
        lq_img_tensor = self.transform(lq_img)

        return {
            'lq_image': lq_img_tensor,
            'hq_image': hq_img_tensor,
            'mask': torch.empty(0),  # mask unused; returned for compatibility
            'lq_path': lq_path,
            'hq_path': hq_path
        }

    def __len__(self):
        """Ensures full epoch coverage by using the larger dataset size."""
        return max(self.hq_size, self.lq_size)
