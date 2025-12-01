import os
import argparse
import torch
import numpy as np

import glob
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tqdm import tqdm
from src.metrics import calculate_hism, calculate_psnr, calculate_ssim, FIDCalculator


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(root_dir, '*.*')))
        # Filter for images
        self.files = [f for f in self.files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('L') # Grayscale for CT
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(path)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Image Generation Quality")
    
    # Paths
    parser.add_argument('--real_dir', type=str, required=True, help='Path to Real (Ground Truth) images')
    parser.add_argument('--fake_dir', type=str, required=True, help='Path to Generated images')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the evaluation report (e.g., results.txt)')
    
    # Execution
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for FID calculation')
    
    # Metric Selection
    parser.add_argument('--metrics', nargs='+', default=['psnr', 'ssim', 'hism', 'fid'],
                        choices=['psnr', 'ssim', 'hism', 'fid'],
                        help='Space-separated list of metrics to calculate (default: all)')

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    requested_metrics = set(args.metrics)

    print(f"Starting evaluation...")
    print(f"    Real Dir: {args.real_dir}")
    print(f"    Fake Dir: {args.fake_dir}")
    print(f"    Metrics:  {', '.join(requested_metrics)}")
    print(f"    Device:   {device}")

    # 1. Setup Transforms
    # Standard transform for [0, 1] tensor range
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()          
    ])

    real_dataset = ImageFolderDataset(args.real_dir, transform=transform)
    fake_dataset = ImageFolderDataset(args.fake_dir, transform=transform)
    
    # 2. Check for Paired Data (by filename)
    real_names = {os.path.basename(f) for f in real_dataset.files}
    fake_names = {os.path.basename(f) for f in fake_dataset.files}
    common_names = sorted(list(real_names.intersection(fake_names)))
    
    is_paired = len(common_names) > 0
    print(f"Found {len(common_names)} paired images based on filenames.")

    # --- METRICS STORAGE ---
    results = {
        'psnr': [],
        'ssim': [],
        'hism': []
    }
    
    final_scores = {}

    # 3. Calculate Paired/Element-wise Metrics (PSNR, SSIM, HISM)
    # These only make sense if we have matching filenames
    paired_metrics_requested = any(m in requested_metrics for m in ['psnr', 'ssim', 'hism'])
    
    if is_paired and paired_metrics_requested:
        print("Calculating Paired Metrics...")
        
        real_map = {os.path.basename(f): f for f in real_dataset.files}
        fake_map = {os.path.basename(f): f for f in fake_dataset.files}

        for name in tqdm(common_names, desc="Processing Pairs"):
            # Load images
            r_img = Image.open(real_map[name]).convert('L').resize((256, 256))
            f_img = Image.open(fake_map[name]).convert('L').resize((256, 256))
            
            # Convert to numpy arrays
            r_arr = np.array(r_img)
            f_arr = np.array(f_img)

            # Calculate only requested metrics
            if 'psnr' in requested_metrics:
                results['psnr'].append(calculate_psnr(r_arr, f_arr))
            if 'ssim' in requested_metrics:
                results['ssim'].append(calculate_ssim(r_arr, f_arr))
            if 'hism' in requested_metrics:
                results['hism'].append(calculate_hism(r_arr, f_arr))

        # Aggregate results
        if 'psnr' in requested_metrics:
            final_scores['psnr'] = (np.mean(results['psnr']), np.std(results['psnr']))
        if 'ssim' in requested_metrics:
            final_scores['ssim'] = (np.mean(results['ssim']), np.std(results['ssim']))
        if 'hism' in requested_metrics:
            final_scores['hism'] = (np.mean(results['hism']), np.std(results['hism']))

    elif not is_paired and paired_metrics_requested:
        print("Warning: PSNR/SSIM/HISM requested but no paired filenames found. Skipping.")

    # 4. Calculate Distribution Metrics (FID)
    if 'fid' in requested_metrics:
        print("Calculating FID (Distribution Metric)...")
        
        real_loader = DataLoader(real_dataset, batch_size=args.batch_size, num_workers=2)
        fake_loader = DataLoader(fake_dataset, batch_size=args.batch_size, num_workers=2)

        fid_runner = FIDCalculator(device=device)
        fid_val = fid_runner.compute_metric(real_loader, fake_loader)
        final_scores['fid'] = fid_val
        print(f"    FID Calculated: {fid_val:.4f}")

    # 5. Generate Report
    report_lines = []
    report_lines.append("=" * 40)
    report_lines.append(f"EVALUATION REPORT")
    report_lines.append("=" * 40)
    report_lines.append(f"Real Dir: {args.real_dir}")
    report_lines.append(f"Fake Dir: {args.fake_dir}")
    report_lines.append("-" * 40)
    
    if 'psnr' in final_scores:
        mean, std = final_scores['psnr']
        report_lines.append(f"PSNR (Higher is better): {mean:.4f} +/- {std:.4f}")
    
    if 'ssim' in final_scores:
        mean, std = final_scores['ssim']
        report_lines.append(f"SSIM (Higher is better): {mean:.4f} +/- {std:.4f}")
        
    if 'hism' in final_scores:
        mean, std = final_scores['hism']
        report_lines.append(f"HISM (Higher is better): {mean:.4f} +/- {std:.4f}")
        
    if 'fid' in final_scores:
        report_lines.append(f"FID  (Lower is better):  {final_scores['fid']:.4f}")
        
    report_lines.append("=" * 40)
    
    report_text = "\n".join(report_lines)
    print(report_text)

    # 6. Save to file
    if args.save_path:
        try:
            # Ensure directory exists
            folder = os.path.dirname(args.save_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
                
            with open(args.save_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved successfully to: {args.save_path}")
        except Exception as e:
            print(f"\nError saving report: {e}")

if __name__ == "__main__":
    main()