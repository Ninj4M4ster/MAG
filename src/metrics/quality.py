import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

def prepare_image_for_metric(img):
    """
    Converts Tensor or Array to standard numpy uint8 or float format [0, 255] or [0, 1].
    Assumes input is potentially [-1, 1] from GANs.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    # Squeeze batch/channel dims if individual image processing
    img = img.squeeze()
    
    # Denormalize from [-1, 1] to [0, 255] if necessary
    if img.min() < 0:
        img = (img * 0.5 + 0.5) * 255.0
        
    return img.astype(np.uint8)

def calculate_psnr(img_true, img_test):
    """
    Wrapper for PSNR calculation.
    """
    img1 = prepare_image_for_metric(img_true)
    img2 = prepare_image_for_metric(img_test)
    return psnr_func(img1, img2, data_range=255)

def calculate_ssim(img_true, img_test):
    """
    Wrapper for SSIM calculation.
    """
    img1 = prepare_image_for_metric(img_true)
    img2 = prepare_image_for_metric(img_test)
    
    # SSIM requires specifying the data range and win_size usually handled by defaults,
    # but for small images, win_size might need adjustment.
    return ssim_func(img1, img2, data_range=255)