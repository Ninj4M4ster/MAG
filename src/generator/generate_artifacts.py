import os
import yaml
import numpy as np
import cv2  # For loading and resizing images
import astra as astra
import scipy.io as sio  # For loading .mat files
from tqdm import tqdm
import argparse

def simulate_metal_artifact(clean_image_hu, metal_mask, vol_geom, proj_geom,
                            projector_id, sino_backend, recon_alg_name, gpu_index):
    """
    Simulates a Metal Artifact (MAR) using the ASTRA toolbox.

    This function is now "stateless" regarding the projector setup;
    it receives all ASTRA configuration as parameters.
    
    Logic:
    1. Forward project the clean image -> sinogram_clean
    2. Forward project the metal mask -> sinogram_metal
    3. Create a 'metal_trace' (where metal is) from sinogram_metal.
    4. Replace values in sinogram_clean where the trace is 
       (e.g., via Linear Interpolation - LI). -> sinogram_artifacted
    5. Reconstruct (backward project) from sinogram_artifacted -> artifacted_image
    """
    
    # Ensure input data is float32, as required by ASTRA
    clean_image_hu = clean_image_hu.astype(np.float32)
    metal_mask = metal_mask.astype(np.float32)

    # --- Step 1 & 2: Forward Projection ---
    # Pass 'gpuIndex' - if it's None (CPU mode), it will be ignored by astra.create_sino
    sinogram_clean_id, sinogram_clean = sino_backend(clean_image_hu, projector_id, gpuIndex=gpu_index)
    sinogram_metal_id, sinogram_metal = sino_backend(metal_mask, projector_id, gpuIndex=gpu_index)

    # --- Step 3: Create Metal Trace ---
    metal_trace = (sinogram_metal > 0.01)

    # --- Step 4: Artifact Simulation (Linear Interpolation) ---
    sinogram_artifacted = np.copy(sinogram_clean)
    
    num_angles = sinogram_clean.shape[0]
    num_detectors = sinogram_clean.shape[1]
    
    x_indices = np.arange(num_detectors) # Detector indices

    for i in range(num_angles): # Loop over each projection angle
        angle_slice = sinogram_clean[i, :]
        trace_slice = metal_trace[i, :]
        
        if not np.any(trace_slice):
            continue

        good_indices = x_indices[~trace_slice]
        bad_indices = x_indices[trace_slice]
        
        good_values = angle_slice[good_indices]
        
        if len(good_values) < 2:
            continue

        interp_values = np.interp(bad_indices, good_indices, good_values)
        sinogram_artifacted[i, bad_indices] = interp_values

    # --- Step 5: Reconstruction (Backward) ---
    
    sinogram_artifacted_id = astra.data2d.create('-sino', proj_geom, sinogram_artifacted)
    recon_image_id = astra.data2d.create('-vol', vol_geom)

    recon_cfg = astra.astra_dict(recon_alg_name)
    recon_cfg['ProjectorId'] = projector_id
    recon_cfg['ProjectionDataId'] = sinogram_artifacted_id
    recon_cfg['ReconstructionDataId'] = recon_image_id
    
    if gpu_index is not None:
        recon_cfg['option'] = {'GPUindex': gpu_index}
    elif recon_alg_name == 'SIRT':
        recon_cfg['IterationCount'] = 20 
    
    alg_id = astra.algorithm.create(recon_cfg)
    astra.algorithm.run(alg_id)
    
    recon_image = astra.data2d.get(recon_image_id)

    # --- ASTRA Memory Cleanup ---
    astra.algorithm.delete(alg_id)
    astra.data2d.delete([sinogram_clean_id, sinogram_metal_id, sinogram_artifacted_id, recon_image_id])
    
    return recon_image, sinogram_artifacted

# --- Setup Functions ---

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Python MAR Generator using ASTRA")
    parser.add_argument('-c', '--config', type=str, default='config/dataset.yaml',
                        required=True,
                        help='Path to the dataset.yaml config file (REQUIRED)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help="Override 'raw_dir' from config (Folder with clean PNGs)")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Override 'dataset_dir' from config (Output folder for PNGs)")
    parser.add_argument('--mask_dir', type=str, default=None,
                        help="Override 'mar_dir' from config (Folder with 'SampleMasks.mat')")
    parser.add_argument('--data_list', type=str, default=None,
                        help="Override 'data_list' from config (Path to image list .txt file)")
    return parser.parse_args()

def load_config(config_path):
    """Loads and parses the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config_yaml = yaml.safe_load(f)
            return config_yaml['deep_lesion']
    except Exception as e:
        print(f"Error: Could not read config file: {e}")
        return None

def setup_paths(args, config):
    """Establishes and validates all required directory paths."""
    print("Loading paths...")
    raw_dir = args.input_dir if args.input_dir else config.get('raw_dir')
    dataset_dir = args.output_dir if args.output_dir else config.get('dataset_dir')
    mar_dir = args.mask_dir if args.mask_dir else config.get('mar_dir')
    data_list_path = args.data_list if args.data_list else config.get('data_list')

    if not all([raw_dir, dataset_dir, mar_dir, data_list_path]):
        print("Error: A key path (input, output, mask, or data_list) is missing.")
        print("Check your .yaml file or command-line arguments.")
        return None
    
    return raw_dir, dataset_dir, mar_dir, data_list_path

def setup_astra_geometries(config):
    """Parses CT parameters and creates ASTRA geometries."""
    try:
        CTpara = config['CTpara']
        IM_PIX_NUM = int(CTpara.get('imPixNum', 512))
        NUM_DETECTORS = int(CTpara.get('sinogram_size_x', 320))
        NUM_ANGLES = int(CTpara.get('angNum', 320))
        SOD = float(CTpara.get('SOD', 1075.0))
        DOD = float(CTpara.get('DOD', 500.0))
        DETECTOR_SPACING = float(CTpara.get('angSize', 0.1))
    except Exception as e:
        print(f"Error parsing 'CTpara' from config: {e}")
        return None

    print("--- ASTRA Configuration ---")
    print(f"Image Size: {IM_PIX_NUM}x{IM_PIX_NUM}")
    print(f"Sinogram Geometry: {NUM_ANGLES} angles x {NUM_DETECTORS} detectors")
    print(f"SOD: {SOD}, DOD: {DOD}, Spacing: {DETECTOR_SPACING}")
    print("--------------------------")

    vol_geom = astra.create_vol_geom(IM_PIX_NUM, IM_PIX_NUM)
    angles = np.linspace(0, 2 * np.pi, NUM_ANGLES, endpoint=False)
    proj_geom = astra.create_proj_geom('fanflat', DETECTOR_SPACING, NUM_DETECTORS,
                                     angles, SOD, DOD)
    
    return vol_geom, proj_geom, IM_PIX_NUM

def load_metadata(mar_dir, data_list_path):
    """Loads the metal masks and the image list."""
    try:
        print(f"Loading masks from: {mar_dir}")
        metal_masks_data = sio.loadmat(os.path.join(mar_dir, 'SampleMasks.mat'))
        metal_masks = metal_masks_data['CT_samples_bwMetal']
    
        print(f"Loading data list from: {data_list_path}")
        with open(data_list_path, 'r') as f:
            data_list = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading metadata (masks or data list): {e}")
        return None
    
    return metal_masks, data_list

def translate_matlab_indices(config):
    """Translates MATLAB 1-based index logic to 0-based Python lists."""
    print("Generating index lists (from MATLAB logic)...")
    CTpara = config['CTpara']
    
    # Process test mask indices
    test_mask_indices_list_matlab = CTpara.get('test_mask_indices', [])
    test_mask_indices_py = [int(i) - 1 for i in test_mask_indices_list_matlab]
    
    # Process train mask indices (set difference)
    all_mask_indices_py = set(range(100)) 
    train_mask_indices_py = list(all_mask_indices_py - set(test_mask_indices_py))
    
    # Process train image indices (MATLAB: (0:3999) * 10 + 1)
    train_indices_matlab = (np.arange(0, 4000) * 10 + 1).tolist()
    train_image_indices_py = [int(i) - 1 for i in train_indices_matlab]
    
    # Process test image indices (MATLAB: (0:199) * 10 + 45000)
    test_indices_matlab = (np.arange(0, 200) * 10 + 45000).tolist()
    test_image_indices_py = [int(i) - 1 for i in test_indices_matlab]
    
    # Create maps for the main loop
    phase_indices_map = {'train': train_image_indices_py, 'test': test_image_indices_py}
    mask_indices_map = {'train': train_mask_indices_py, 'test': test_mask_indices_py}
    
    print("Index lists generated.")
    return phase_indices_map, mask_indices_map

# --- Main Execution ---

def main():
    # --- 1. Setup ---
    args = parse_arguments()
    config = load_config(args.config)
    if config is None:
        return

    paths = setup_paths(args, config)
    if paths is None:
        return
    raw_dir, dataset_dir, mar_dir, data_list_path = paths

    geometries = setup_astra_geometries(config)
    if geometries is None:
        return
    vol_geom, proj_geom, IM_PIX_NUM = geometries

    metadata = load_metadata(mar_dir, data_list_path)
    if metadata is None:
        return
    metal_masks, data_list = metadata

    indices = translate_matlab_indices(config)
    phase_indices_map, mask_indices_map = indices

    # --- 2. Main Generation Loop ---
    splits = ['train', 'test']
    for phase in splits:
        print(f"\nProcessing phase: {phase}")
        phase_dir = os.path.join(dataset_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)

        # --- Setup ASTRA Projector (once per phase) ---
        sino_backend = astra.create_sino
        try:
            projector_id = astra.create_projector('cuda', proj_geom, vol_geom)
            recon_alg_name = 'FBP_CUDA'
            gpu_index = 0  # Use default GPU
            print(f"INFO: [{phase}] Using CUDA (GPU) projector.")
        except Exception as e:
            print(f"WARNING: [{phase}] Could not init CUDA projector ({e}). Switching to CPU (slower)...")
            projector_id = astra.create_projector('line_fanflat', proj_geom, vol_geom) 
            recon_alg_name = 'SIRT'
            gpu_index = None

        image_indices_py = phase_indices_map.get(phase, [])
        mask_indices_py = mask_indices_map.get(phase, [])
        
        if not image_indices_py or not mask_indices_py:
            print(f"Warning: No indices found for phase '{phase}'. Skipping.")
            continue
            
        selected_metal = metal_masks[:, :, mask_indices_py]
        
        # Loop over all images in this phase
        for i, img_idx in enumerate(tqdm(image_indices_py, desc=f"Generating {phase}")):
            
            try:
                image_name = data_list[img_idx]
            except IndexError:
                print(f"[ERROR] Index {img_idx} out of bounds for data_list (length {len(data_list)}). Skipping.")
                continue
            except Exception as e:
                print(f"[ERROR] Could not get image name: {e}")
                continue
            
            image_name_norm = image_name.replace("\\", "/")
            
            output_dir_patient = os.path.join(phase_dir, os.path.splitext(image_name_norm)[0])
            os.makedirs(output_dir_patient, exist_ok=True)
            
            gt_file_png = os.path.join(output_dir_patient, 'gt.png')
            
            if os.path.exists(gt_file_png):
                continue
                
            try:
                raw_image_path = os.path.join(raw_dir, image_name_norm)
                raw_image = cv2.imread(raw_image_path, cv2.IMREAD_UNCHANGED)

                if raw_image is None:
                    print(f"Warning: Could not read image: {raw_image_path}")
                    continue
                
                image_hu = raw_image.astype(np.float32) - 32768.0
                
                image_hu_resized = cv2.resize(image_hu, (IM_PIX_NUM, IM_PIX_NUM), 
                                              interpolation=cv2.INTER_LINEAR)
                
                image_hu_resized = np.clip(image_hu_resized, -1000.0, None)
                
                gt_image_resized = cv2.resize(raw_image.astype(np.float32), 
                                             (IM_PIX_NUM, IM_PIX_NUM), 
                                             interpolation=cv2.INTER_LINEAR).astype(raw_image.dtype)
                
                cv2.imwrite(gt_file_png, gt_image_resized)
                
                # Loop over all selected metal masks for this phase
                for j in range(selected_metal.shape[2]):
                    metal_mask = selected_metal[:, :, j]
                    
                    metal_mask_resized = cv2.resize(metal_mask.astype(np.float32),
                                                    (IM_PIX_NUM, IM_PIX_NUM),
                                                    interpolation=cv2.INTER_LINEAR)
                    metal_mask_resized = (metal_mask_resized > 0.5).astype(np.float32)

                    # --- SIMULATE THE ARTIFACT ---
                    ma_CT, _ = simulate_metal_artifact(
                        image_hu_resized,
                        metal_mask_resized,
                        vol_geom,
                        proj_geom,
                        projector_id,
                        sino_backend,
                        recon_alg_name,
                        gpu_index
                    )
                    
                    ma_CT_16bit = ma_CT + 32768.0
                    ma_CT_16bit = np.clip(ma_CT_16bit, 0, 65535).astype(raw_image.dtype)
                    
                    ct_file_png = os.path.join(output_dir_patient, f'{j+1}.png')
                    cv2.imwrite(ct_file_png, ma_CT_16bit)

            except Exception as e:
                print(f"\nError processing {image_name_norm}: {e}")
                import traceback
                traceback.print_exc()

        # --- Clean up projector after the phase is done ---
        astra.projector.delete(projector_id)

    print("MAR generation finished.")

if __name__ == "__main__":
    main()