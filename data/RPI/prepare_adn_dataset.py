import os
import shutil
import argparse
from tqdm import tqdm
from glob import glob

def prepare_adn_data(source_dir, output_dir):
    """
    Processes the raw AAPM/RPI dataset into a standardized format for ADN training.
    Filters image files and organizes them into paired 'clean' (ground truth)
    and 'artifact' directories.
    """
    
    # Define output structure
    # dataset_adn/
    #   ├── artifact/  (Input images with artifacts)
    #   └── clean/     (Ground Truth images)
    
    img_artifact_dir = os.path.join(output_dir, "artifact")
    img_clean_dir = os.path.join(output_dir, "clean")
    
    os.makedirs(img_artifact_dir, exist_ok=True)
    os.makedirs(img_clean_dir, exist_ok=True)

    print(f"Searching for files in: {source_dir} ...")
    
    # Recursively search for all files in the source directory
    all_files = glob(os.path.join(source_dir, "**", "*"), recursive=True)
    
    # Filter for supported image extensions
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    image_files = [f for f in all_files if f.lower().endswith(valid_exts)]

    if not image_files:
        print("ERROR: No images found. Please check if the data is decompressed.")
        
        # Check for .npy files (common in scientific datasets) to provide a helpful hint
        npy_files = [f for f in all_files if f.lower().endswith('.npy')]
        if npy_files:
            print(f"Found {len(npy_files)} .npy files. This script supports standard images only. "
                  "Conversion is required for .npy files.")
        return

    print(f"Found {len(image_files)} images. Analyzing pairs...")

    pairs_found = 0
    
    # Dictionary to pair images by ID
    # key: ID (filename without suffix), value: { 'clean': path, 'artifact': path }
    data_map = {}

    for file_path in tqdm(image_files, desc="Indexing"):
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # Filter out auxiliary files like sinograms or masks
        if "sinogram" in filename.lower() or "mask" in filename.lower():
            continue

        # DETECT CLEAN IMAGE (Ground Truth)
        # Keywords: GT, clean, label
        is_clean = False
        if "_GT" in name_no_ext or "clean" in name_no_ext.lower() or "label" in name_no_ext.lower():
            is_clean = True
            # Strip suffix to obtain a common ID for pairing
            img_id = name_no_ext.replace("_GT", "").replace("_clean", "").replace("_label", "")
        
        # DETECT ARTIFACT IMAGE (Input)
        # Keywords: MA (Metal Artifact), metal, FBP
        elif "_MA" in name_no_ext or "metal" in name_no_ext.lower() or "_FBP" in name_no_ext:
            is_clean = False
            img_id = name_no_ext.replace("_MA", "").replace("_metal", "").replace("_FBP", "")
            
        else:
            # Fallback: Check parent directory for classification
            parent = os.path.basename(os.path.dirname(file_path)).lower()
            img_id = name_no_ext
            
            if "gt" in parent or "clean" in parent or "label" in parent:
                is_clean = True
            elif "ma" in parent or "input" in parent or "metal" in parent:
                is_clean = False
            else:
                # Unknown classification; skip file
                continue

        # Add to map
        if img_id not in data_map:
            data_map[img_id] = {}
        
        if is_clean:
            data_map[img_id]['clean'] = file_path
        else:
            data_map[img_id]['artifact'] = file_path

    # Process identified pairs
    print("Copying and renaming files...")
    
    for img_id, paths in tqdm(data_map.items(), desc="Processing"):
        if 'clean' in paths and 'artifact' in paths:
            # Complete pair found
            clean_src = paths['clean']
            artifact_src = paths['artifact']
            
            # Unify filename for the dataset (e.g., ID.png)
            ext = os.path.splitext(clean_src)[1]
            new_name = f"{img_id}{ext}"
            
            # Copy Clean/GT
            shutil.copy2(clean_src, os.path.join(img_clean_dir, new_name))
            
            # Copy Artifact/Input
            shutil.copy2(artifact_src, os.path.join(img_artifact_dir, new_name))
            
            pairs_found += 1

    print(f"\n--- PROCESSING COMPLETE ---")
    print(f"Pairs created: {pairs_found}")
    print(f"Data saved to: {output_dir}")
    print(f"  -> {img_clean_dir}")
    print(f"  -> {img_artifact_dir}")
    
    if pairs_found == 0:
        print("\nWARNING: No pairs found. Verify filename patterns in the source directory.")
        print("Sample files found in source:")
        print(image_files[:5] if image_files else "None")

def main():
    parser = argparse.ArgumentParser(description="Prepare AAPM data for ADN training pipeline.")
    parser.add_argument("-s", "--source", required=True, help="Source directory containing raw extracted data (e.g., data_rpi)")
    parser.add_argument("-o", "--output", default="dataset_adn", help="Output directory for the processed dataset")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' does not exist.")
        return
        
    prepare_adn_data(args.source, args.output)

if __name__ == "__main__":
    main()