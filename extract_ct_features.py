# extract_ct_features.py
# Parameterize extract_ct_features.py

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse

# --- Import your project modules ---
import config
from model import MultiModalModel # We need this to correctly instantiate the model architecture
from dataset import get_val_transforms_for_npy # Use validation transforms (no augmentation)

# =============================================================================
# A simplified, CT-only Dataset for feature extraction
# =============================================================================
class CTOnlyDataset(Dataset):
    """
    A lightweight dataset that finds and loads all CT .npy files from a directory.
    """
    def __init__(self, ct_data_dir, transform):
        self.ct_dir = ct_data_dir
        self.transform = transform
        
        # Scan the directory for all .npy files
        self.file_list = [f for f in os.listdir(ct_data_dir) if f.endswith('.npy')]
        if not self.file_list:
            raise FileNotFoundError(f"No .npy files found in the specified CT directory: {ct_data_dir}")
            
        # Extract sample_id from filename (e.g., 'sample123_ct.npy' -> 'sample123')
        self.sample_ids = [os.path.splitext(f)[0].replace('_ct', '') for f in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.ct_dir, self.file_list[idx])
        sample_id = self.sample_ids[idx]
        
        try:
            ct_array = np.load(file_path).astype(np.float32)
            
            # Ensure the array has a channel dimension for MONAI (C, D, H, W)
            if ct_array.ndim == 3:
                ct_array = np.expand_dims(ct_array, axis=0)
            
            # Apply MONAI transformations
            processed_ct = self.transform({"image": ct_array})
            return {'image': processed_ct['image'], 'sample_id': sample_id}
            
        except Exception as e:
            print(f"\nWarning: Could not load or process file {file_path}. Error: {e}")
            return None # Return None to be filtered out by the collate function

def collate_fn_filter_none(batch):
    """A collate function that filters out None items from a batch."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Default collate behavior for the remaining items
    images = torch.stack([item['image'] for item in batch])
    sample_ids = [item['sample_id'] for item in batch]
    return {'image': images, 'sample_id': sample_ids}


# =============================================================================
# Main execution function
# =============================================================================
def main(heavy_model_path, output_path): # ★★★ 修正点 1: 确保这里的参数名是 output_path
    """
    Loads a trained heavy model, extracts CT features for all available CT scans,
    and saves them to a specified CSV file.

    Args:
        heavy_model_path (str): Path to the saved .pth file of the trained heavy model.
        output_path (str): Path where the output .csv file with features will be saved.
    """
    print("\n" + "="*80)
    print("      WORKFLOW: Pre-computing CT Feature Vectors")
    print("="*80)
    print(f"Loading heavy model from: {heavy_model_path}")
    print(f"Output features will be saved to: {output_path}") # ★★★ 修正点 2: 保持变量名一致

    if not os.path.exists(heavy_model_path):
        raise FileNotFoundError(f"ERROR: Heavy model file not found at {heavy_model_path}")

    # --- 1. Load the trained heavy model and extract the CT feature extractor ---
    
    # IMPORTANT: To load the heavy model correctly, we must temporarily set the
    # config to "heavy mode" so the MultiModalModel class builds the right architecture
    # (with Swin-UNETR).
    print("Temporarily setting config to 'heavy mode' to load model architecture...")
    config.ACTIVE_MODALITIES = ["ct", "urine", "blood", "microbiome", "kegg", "clinical"]
    config.setup_experiment_params()
    
    # Instantiate the full model architecture, then load the saved state dictionary
    full_model = MultiModalModel().to(config.DEVICE)
    checkpoint = torch.load(heavy_model_path, map_location=config.DEVICE)
    full_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract the CT feature extractor part of the model
    ct_feature_extractor = full_model.feature_extractors['ct']
    if ct_feature_extractor is None:
        raise KeyError("ERROR: The loaded model does not contain a 'ct' feature extractor. Check model architecture.")
    
    ct_feature_extractor.eval() # Set to evaluation mode
    print("CT Feature Extractor (SwinUNETR) successfully extracted from the heavy model.")

    # --- 2. Create the dataset and dataloader for all CT scans ---
    ct_data_dir = config.DATA_PATHS.get('ct')
    if not ct_data_dir or not os.path.isdir(ct_data_dir):
        raise FileNotFoundError(f"ERROR: CT data directory not found or not valid in config: {ct_data_dir}")

    ct_transforms = get_val_transforms_for_npy()
    dataset = CTOnlyDataset(ct_data_dir, ct_transforms)
    
    # Use a larger batch size for faster inference if GPU memory allows
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=4, collate_fn=collate_fn_filter_none)
    
    # --- 3. Iterate through data and extract features ---
    all_features = []
    all_sample_ids = []
    
    print(f"\nStarting feature extraction from {len(dataset)} CT images...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting CT Features"):
            if batch is None: # Skip batches that had processing errors
                continue
            images = batch['image'].to(config.DEVICE)
            features = ct_feature_extractor(images)
            
            all_features.append(features.cpu().numpy())
            all_sample_ids.extend(batch['sample_id'])
            
    # --- 4. Format and save features to a CSV file ---
    if not all_features:
        print("\nWARNING: No features were extracted. The output CSV file will be empty.")
        df = pd.DataFrame(columns=['sample_id'] + [f'ct_feat_{i}' for i in range(config.D_MODEL)])
    else:
        concatenated_features = np.concatenate(all_features, axis=0)
        df = pd.DataFrame(concatenated_features, columns=[f'ct_feat_{i}' for i in range(concatenated_features.shape[1])])
        df.insert(0, 'sample_id', all_sample_ids)
    
    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # ★★★ 修正点 3: 保持变量名一致
    df.to_csv(output_path, index=False) # ★★★ 修正点 4: 保持变量名一致
    
    print("\n" + "="*80)
    print(f"SUCCESS! Pre-computed CT features have been saved to:")
    print(os.path.abspath(output_path)) # ★★★ 修正点 5: 保持变量名一致
    print("="*80)


if __name__ == "__main__":
    # Setup argparse to accept command-line arguments, making the script flexible
    parser = argparse.ArgumentParser(description="Extract CT features using a trained heavy model.")
    parser.add_argument('--heavy_model_path', type=str, required=True, 
                        help='Full path to the saved .pth file of the trained heavy model.')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Full path where the output .csv file with features will be saved.')
    
    args = parser.parse_args()
    
    # ★★★ 修正点 6: 确保这里的调用和上面的定义完全匹配
    main(heavy_model_path=args.heavy_model_path, output_path=args.output_path)