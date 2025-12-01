import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from scipy import linalg
from torch.nn import functional as F

class FIDCalculator:
    """
    Calculates Frechet Inception Distance (FID) between two distributions of images.
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.dims = 2048 # InceptionV3 pool3 output size
        
        # Load InceptionV3 pre-trained on ImageNet
        # Note: We need to remove the classification head
        block_idx = models.inception.Inception3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.model.to(device)
        self.model.eval()

    def get_activations(self, images_tensor):
        """
        Computes activations for a batch of images.
        Input: Tensor (B, C, H, W) normalized usually [-1, 1] or [0, 1]
        """
        # Resize to 299x299 (Inception requirement)
        if images_tensor.shape[-1] != 299:
            images_tensor = F.interpolate(images_tensor, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Inception expects 3 channels. If grayscale, repeat.
        if images_tensor.shape[1] == 1:
            images_tensor = images_tensor.repeat(1, 3, 1, 1)

        # Normalize to ImageNet mean/std if not already handled, 
        # typically FID implementations expect inputs in range [0, 1] or normalized.
        # Here assuming simple range adaptation.
        
        with torch.no_grad():
            # Inception v3 logic to get features before FC
            x = images_tensor
            # Trigger standard forward hooks or use intermediate layers?
            # Easiest clean way in torch:
            # (Standard pytorch inception_v3 returns logits, we need feature vector)
            # We will use a hook or modified forward if possible, 
            # but simplest standard way:
            pred = self.model(x) # Since we replaced fc with Identity, this gives 2048 vec
            
        return pred.cpu().numpy()

    def calculate_statistics(self, activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """
        Numpy implementation of the Frechet Distance.
        """
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        
        # Product of covariances
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Numerical stability check
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean

    def compute_metric(self, real_loader, fake_loader):
        """
        Main entry point. Iterates over two dataloaders to compute FID.
        """
        act_real = []
        act_fake = []

        print("Extracting features from Real images...")
        for batch in real_loader:
            if isinstance(batch, (list, tuple)): batch = batch[0]
            batch = batch.to(self.device)
            act_real.append(self.get_activations(batch))

        print("Extracting features from Generated images...")
        for batch in fake_loader:
            if isinstance(batch, (list, tuple)): batch = batch[0]
            batch = batch.to(self.device)
            act_fake.append(self.get_activations(batch))

        act_real = np.concatenate(act_real, axis=0)
        act_fake = np.concatenate(act_fake, axis=0)

        mu1, sigma1 = self.calculate_statistics(act_real)
        mu2, sigma2 = self.calculate_statistics(act_fake)

        fid_value = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid_value