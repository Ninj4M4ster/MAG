import os
import argparse
import random
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# Attempt to import network architectures
# Adjust imports based on your specific project structure
try:
    from src.networks.adn import ADN, ADN_VAE
except ImportError:
    raise ImportError("Could not import ADN or ADN_VAE from networks.adn. Please check your python path.")

def parse_args():
    """
    Parses command line arguments for artifact generation.
    Architecture settings are now loaded from the config file.
    """
    parser = argparse.ArgumentParser(description="ADN Artifact Generation Script (Clean -> Corrupt)")
    
    # --- Configuration ---
    parser.add_argument('--config', type=str, default='configs/adn.yaml', 
                        help='Path to the YAML training configuration file (e.g., configs/adn.yaml)')
    
    # --- Model Selection ---
    parser.add_argument('--model_type', type=str, default='vae', choices=['adn', 'vae'],
                        help='Choose class wrapper: "adn" (Standard) or "vae" (Variational). '
                             'Note: This selects the Python class, but dimensions come from the YAML.')

    # --- Paths ---
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth checkpoint file')
    parser.add_argument('--clean_dir', type=str, required=True, help='Directory containing clean images (Content source)')
    parser.add_argument('--artifact_dir', type=str, required=True, help='Directory containing artifact images (Style/Artifact source)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save generated images')
    
    # --- Execution Options ---
    parser.add_argument('--img_size', type=int, default=256, help='Resize images to this size (0 to disable, uses original size)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (-1 for CPU)')
    
    return parser.parse_args()

def load_config(config_path):
    """
    Loads the YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Configuration loaded from: {config_path}")
    return config

def load_model(args, config, device):
    """
    Initializes the model using parameters from the YAML config and loads weights.
    """
    print(f"Initializing model architecture: {args.model_type.upper()}...")
    
    if 'model' not in config or 'adn' not in config['model']:
        raise ValueError("Invalid config file: Missing 'model' or 'model.adn' section.")

    model_cfg = config['model']['adn']
    
    input_ch = model_cfg.get('input_ch', 1)
    base_ch = model_cfg.get('base_ch', 16)
    num_down = model_cfg.get('num_down', 2)
    num_residual = model_cfg.get('num_residual', 2)
    num_sides = model_cfg.get('num_sides', 3)
    
    # Handle "all" string for num_sides or integer
    if num_sides != 'all':
        num_sides = int(num_sides)

    # Other architecture params
    res_norm = model_cfg.get('res_norm', 'instance')
    down_norm = model_cfg.get('down_norm', 'instance')
    up_norm = model_cfg.get('up_norm', 'layer')
    fuse = model_cfg.get('fuse', True)
    shared_decoder = model_cfg.get('shared_decoder', False)

    print(f"    Parameters: Base CH={base_ch}, Down={num_down}, Res={num_residual}, Sides={num_sides}")

    # Select the Class
    if args.model_type == 'vae':
        ModelClass = ADN_VAE
    else:
        ModelClass = ADN

    model = ModelClass(
        input_ch=input_ch,
        base_ch=base_ch,
        num_down=num_down,
        num_residual=num_residual,
        num_sides=num_sides,
        res_norm=res_norm,
        down_norm=down_norm,
        up_norm=up_norm,
        fuse=fuse,
        shared_decoder=shared_decoder
    )
    
    # Load weights
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")
        
    print(f"Loading weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle different checkpoint structures (Trainer wrapper vs direct state_dict)
    if 'model_g' in checkpoint:
        state_dict = checkpoint['model_g']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Clean up state dictionary keys (remove 'module.' if trained with DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    # Load state dict
    try:
        model.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed. Ensure --model_type and Config match the checkpoint.\n    Details: {e}")

    model.to(device)
    model.eval() # Set to evaluation mode
    
    return model

def get_transforms(img_size):
    """
    Returns transformation pipeline: Resize (Optional) -> Grayscale -> Tensor -> Normalize
    """
    transform_list = []
    if img_size > 0:
        transform_list.append(transforms.Resize((img_size, img_size)))
    
    transform_list += [
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] range
    ]
    return transforms.Compose(transform_list)

def tensor_to_img(tensor):
    """
    Converts a normalized tensor (-1, 1) back to a PIL Image (0, 255).
    """
    tensor = tensor.detach().cpu().squeeze()
    tensor = tensor * 0.5 + 0.5  # Denormalize to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    array = (tensor.numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)

def main():
    args = parse_args()
    
    # Setup Device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"Running on device: {device}")

    # Load Configuration
    config = load_config(args.config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model (Architecture defined by Config + Type arg)
    model = load_model(args, config, device)

    # Prepare Data Loaders
    clean_files = sorted([f for f in os.listdir(args.clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])
    artifact_files = sorted([f for f in os.listdir(args.artifact_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))])

    if not clean_files:
        raise ValueError("No images found in clean directory.")
    if not artifact_files:
        raise ValueError("No images found in artifact directory.")

    transform = get_transforms(args.img_size)

    print(f"Starting generation on {len(clean_files)} images...")
    
    # Inference Loop
    for clean_name in tqdm(clean_files, desc="Generating"):
        
        # 1. Load Clean Image (Content)
        clean_path = os.path.join(args.clean_dir, clean_name)
        img_clean = Image.open(clean_path).convert('L')
        img_clean_t = transform(img_clean).unsqueeze(0).to(device)

        # 2. Select Artifact Image (Style Reference)
        # Randomly select a 'donor' image for artifact style
        ref_name = random.choice(artifact_files)
        ref_path = os.path.join(args.artifact_dir, ref_name)
        img_ref = Image.open(ref_path).convert('L')
        img_ref_t = transform(img_ref).unsqueeze(0).to(device)

        # 3. Forward Pass (Generate Artifact)
        with torch.no_grad():
            # forward_hl: High Quality Content (clean) -> Low Quality Output
            # x_low=img_ref_t  (Source of artifact style/sides)
            # x_high=img_clean_t (Source of anatomical content)
            
            pred_img = model.forward_hl(x_low=img_ref_t, x_high=img_clean_t)

        # 4. Save Output
        result_pil = tensor_to_img(pred_img)
        
        # Construct output filename (e.g., gen_patient123.png)
        save_name = f"gen_{os.path.splitext(clean_name)[0]}.png"
        save_path = os.path.join(args.output_dir, save_name)
        result_pil.save(save_path)

    print(f"Process completed. Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()