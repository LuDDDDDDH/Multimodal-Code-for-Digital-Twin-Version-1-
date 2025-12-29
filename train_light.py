# train_light.py

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import argparse

# ==================== NEWLY ADDED IMPORTS ====================
# These were missing, causing the NameError for the validate function.
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
# =============================================================

# --- Import your project modules ---
import config
from dataset import MultiModalDataset, custom_collate_fn
from model import MultiModalModel
from utils import (
    EarlyStopping, 
    save_checkpoint, 
    setup_optimizers_schedulers,
    NegativeLogLikelihood,
    calculate_c_index,
    calculate_time_dependent_auc # This one is also needed by validate
)

# =============================================================================
# Core Training & Validation Logic
# =============================================================================

def train_one_epoch(model, dataloader, criterions, optimizer, device):
    model.train()
    task_losses = {task: 0.0 for task in criterions.keys()}
    
    for batch in dataloader:
        efficacy_labels = batch.get('efficacy_label').to(device).view(-1)
        toxicity_labels = batch.get('toxicity_label').to(device).view(-1)
        survival_times = batch.get('survival_time').to(device).view(-1)
        survival_status = batch.get('survival_status').to(device).view(-1)
        
        optimizer.zero_grad()
        outputs = model(batch)
        
        loss_efficacy = criterions['efficacy'](outputs['efficacy'], efficacy_labels)
        loss_toxicity = criterions['toxicity'](outputs['toxicity'], toxicity_labels)
        loss_survival = criterions['survival'](outputs['survival'], survival_times, survival_status)
        
        total_loss = loss_efficacy + loss_toxicity + loss_survival
        
        total_loss.backward()
        optimizer.step()

        task_losses['efficacy'] += loss_efficacy.item()
        task_losses['toxicity'] += loss_toxicity.item()
        task_losses['survival'] += loss_survival.item()

    num_batches = len(dataloader)
    avg_losses = {task: total_loss / num_batches for task, total_loss in task_losses.items()}
    avg_losses['total_for_logging'] = sum(avg_losses.values())
    
    return avg_losses


def validate(model, dataloader, criterions, device):
    model.eval()
    
    total_loss_sum = 0
    efficacy_loss_sum, toxicity_loss_sum, survival_loss_sum = 0, 0, 0
    
    all_outputs = defaultdict(list)
    all_labels = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            # Ensure labels are available before trying to extend
            if batch.get('efficacy_label') is not None:
                all_labels['efficacy'].extend(batch.get('efficacy_label').view(-1).numpy())
            if batch.get('toxicity_label') is not None:
                all_labels['toxicity'].extend(batch.get('toxicity_label').view(-1).numpy())
            if batch.get('survival_time') is not None:
                all_labels['survival_time'].extend(batch.get('survival_time').view(-1).numpy())
            if batch.get('survival_status') is not None:
                all_labels['survival_status'].extend(batch.get('survival_status').view(-1).numpy())
            
            efficacy_labels = batch.get('efficacy_label').to(device).view(-1)
            toxicity_labels = batch.get('toxicity_label').to(device).view(-1)
            survival_times = batch.get('survival_time').to(device).view(-1)
            survival_status = batch.get('survival_status').to(device).view(-1)

            outputs = model(batch)
            
            all_outputs['efficacy'].extend(torch.sigmoid(outputs['efficacy']).cpu().numpy())
            all_outputs['toxicity'].extend(torch.sigmoid(outputs['toxicity']).cpu().numpy())
            all_outputs['survival'].extend(outputs['survival'].cpu().numpy())
            
            loss_efficacy = criterions['efficacy'](outputs['efficacy'], efficacy_labels)
            loss_toxicity = criterions['toxicity'](outputs['toxicity'], toxicity_labels)
            loss_survival = criterions['survival'](outputs['survival'], survival_times, survival_status)
            total_loss = loss_efficacy + loss_toxicity + loss_survival

            total_loss_sum += total_loss.item()
            efficacy_loss_sum += loss_efficacy.item()
            toxicity_loss_sum += loss_toxicity.item()
            survival_loss_sum += loss_survival.item()

    num_batches = len(dataloader)
    metrics = {
        'loss_total': total_loss_sum / num_batches,
        'loss_efficacy': efficacy_loss_sum / num_batches,
        'loss_toxicity': toxicity_loss_sum / num_batches,
        'loss_survival': survival_loss_sum / num_batches,
    }

    for key in all_labels: all_labels[key] = np.array(all_labels[key])
    for key in all_outputs: all_outputs[key] = np.array(all_outputs[key])

    # Check if there are labels to calculate metrics on
    if len(all_labels['efficacy']) > 0:
        metrics['efficacy_auc'] = roc_auc_score(all_labels['efficacy'], all_outputs['efficacy'])
        binary_preds_eff = (all_outputs['efficacy'] > 0.5).astype(int)
        metrics['efficacy_acc'] = accuracy_score(all_labels['efficacy'], binary_preds_eff)
        metrics['efficacy_f1'] = f1_score(all_labels['efficacy'], binary_preds_eff)

    if len(all_labels['toxicity']) > 0:
        metrics['toxicity_auc'] = roc_auc_score(all_labels['toxicity'], all_outputs['toxicity'])
        binary_preds_tox = (all_outputs['toxicity'] > 0.5).astype(int)
        metrics['toxicity_acc'] = accuracy_score(all_labels['toxicity'], binary_preds_tox)
        metrics['toxicity_f1'] = f1_score(all_labels['toxicity'], binary_preds_tox)

    if len(all_labels['survival_status']) > 0:
        metrics['survival_c_index'] = calculate_c_index(
            all_outputs['survival'], all_labels['survival_time'], all_labels['survival_status']
        )
        # The following time-dependent AUCs might be computationally intensive
        # You can comment them out during debugging if needed
        metrics['survival_auc_1yr'] = calculate_time_dependent_auc(
            all_outputs['survival'], all_labels['survival_time'], all_labels['survival_status'], time_point=365
        )
        metrics['survival_auc_2yr'] = calculate_time_dependent_auc(
            all_outputs['survival'], all_labels['survival_time'], all_labels['survival_status'], time_point=730
        )

    return metrics

# =============================================================================
# Main Workflow for LIGHT Model Training
# =============================================================================

def run_light_model_training(seed):
    """
    Main function to run a full K-Fold cross-validation for the LIGHT model.
    """
    print("\n" + "="*80)
    print(f"      WORKFLOW: LIGHT MODEL TRAINING (Seed: {seed})")
    print("="*80 + "\n")

    print("--- Overriding config for Light Model Training ---")
    
    repeat_label = f'repeat_seed_{seed}'
    repeat_dir = os.path.join(config.BASE_DIR, 'outputs', 'full_experiment', repeat_label)
    config.OUTPUT_DIR = repeat_dir 
    
    if not os.path.exists(repeat_dir):
        os.makedirs(repeat_dir)
        print(f"Created light model directory: {repeat_dir}")
        
    config.ACTIVE_MODALITIES = ["ct_vector", "urine", "blood", "microbiome", "kegg", "clinical"]
    if 'ct' in config.ACTIVE_MODALITIES:
        config.ACTIVE_MODALITIES.remove('ct')
    config.TRAINING_STRATEGY = 'direct'
    
    config.setup_experiment_params()
    print("--- Config setup complete for Light Model ---\n")

    split_file_path = os.path.join(repeat_dir, 'kfold_splits.json')
    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"CRITICAL ERROR: kfold_splits.json not found in {repeat_dir}. Please run train_heavy.py for this seed first.")
    
    with open(split_file_path, 'r') as f:
        fold_splits = json.load(f)
        
    manifest_df = pd.read_csv(config.MANIFEST_PATH)
    
    for fold_num_str, id_splits in fold_splits.items():
        fold_num = int(fold_num_str.split('_')[-1])
        print(f"\n==================== [ Light Model Fold {fold_num}/{len(fold_splits)} ] ====================\n")
        
        train_ids = id_splits['train_ids']
        val_ids = id_splits['val_ids']

        ct_vector_path = os.path.join(repeat_dir, f"precomputed_ct_features_fold_{fold_num}.csv")
        if not os.path.exists(ct_vector_path):
            raise FileNotFoundError(f"CRITICAL ERROR: CT vector file not found for fold {fold_num}: {ct_vector_path}")
        
        original_ct_vector_path = config.DATA_PATHS.get('ct_vector')
        config.DATA_PATHS['ct_vector'] = ct_vector_path
        print(f"Loading fold-specific CT features from: {os.path.basename(ct_vector_path)}")
        
        train_ds = MultiModalDataset(train_ids, manifest_df, config.DATA_PATHS, config.MODALITY_NAMES, mode='train')
        val_ds = MultiModalDataset(val_ids, manifest_df, config.DATA_PATHS, config.MODALITY_NAMES, mode='val')
        
        if original_ct_vector_path:
            config.DATA_PATHS['ct_vector'] = original_ct_vector_path
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
        
        model = MultiModalModel().to(config.DEVICE)
        
        efficacy_pos_weight = torch.tensor([config.EFFICACY_POS_WEIGHT], device=config.DEVICE)
        toxicity_pos_weight = torch.tensor([config.TOXICITY_POS_WEIGHT], device=config.DEVICE)
        criterions = {
            'efficacy': nn.BCEWithLogitsLoss(pos_weight=efficacy_pos_weight),
            'toxicity': nn.BCEWithLogitsLoss(pos_weight=toxicity_pos_weight),
            'survival': NegativeLogLikelihood()
        }
        
        model_save_path = os.path.join(repeat_dir, f"best_light_model_fold_{fold_num}.pth")
        early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True, path=model_save_path)

        optimizer, scheduler = setup_optimizers_schedulers(model, is_staged=False, stage='direct')
        total_epochs = config.DIRECT_TRAINING_EPOCHS

        for epoch in range(total_epochs):
            train_losses = train_one_epoch(model, train_loader, criterions, optimizer, config.DEVICE)
            val_metrics = validate(model, val_loader, criterions, config.DEVICE)
            
            print(f"Epoch {epoch+1}/{total_epochs} | "
                  f"Train Loss: {train_losses['total_for_logging']:.4f} | "
                  f"Val C-idx: {val_metrics.get('survival_c_index', 0):.4f}, Eff-AUC: {val_metrics.get('efficacy_auc', 0):.4f}, Tox-AUC: {val_metrics.get('toxicity_auc', 0):.4f}")
            
            early_stopping_score = (val_metrics.get('efficacy_auc', 0) + val_metrics.get('toxicity_auc', 0) + val_metrics.get('survival_c_index', 0)) / 3.0
            print(f"          EarlyStopping Score (avg): {early_stopping_score:.4f}")
            
            if scheduler:
                scheduler.step()

            early_stopping(early_stopping_score, model, optimizer, epoch)
            
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        
        print(f"\nFold {fold_num} light model training complete. Best model saved to: {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LIGHT multi-modal models using pre-computed features for one full K-Fold run.")
    parser.add_argument('--seed', type=int, required=True, help='The random seed for the experiment repeat, used to locate the correct data folder.')
    
    args = parser.parse_args()
    
    run_light_model_training(seed=args.seed)