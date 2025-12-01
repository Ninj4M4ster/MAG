import numpy as np
import torch

def calculate_histogram(img, bins=256, range_min=0, range_max=255):
    """
    Computes normalized histogram for an image.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    # Flatten the image channel
    hist, _ = np.histogram(img.ravel(), bins=bins, range=(range_min, range_max))
    
    # Normalize histogram to sum to 1 (probability distribution)
    hist = hist.astype(np.float32)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist /= hist_sum
    return hist

def calculate_hism(img1, img2, bins=256):
    """
    Calculates Histogram Intersection Similarity Method (HISM).
    Returns a value between 0 (no overlap) and 1 (identical distributions).
    """
    h1 = calculate_histogram(img1, bins=bins)
    h2 = calculate_histogram(img2, bins=bins)
    
    # Intersection calculation: sum of min values at each bin
    intersection = np.minimum(h1, h2).sum()
    return intersection