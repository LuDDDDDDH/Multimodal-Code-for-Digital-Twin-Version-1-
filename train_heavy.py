# train_heavy.py

import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import argparse

# --- 导入您项目中的模块 ---
# 确保这些模块可以被正确找到
import config
from dataset import MultiModalDataset, custom_collate_fn
from model import MultiModalModel
from utils import (
    EarlyStopping, 
    setup_optimizers_schedulers,
    NegativeLogLikelihood
)

# --- 导入核心训练/验证函数 ---
# 假设您已将原始train.py重命名为train_light.py
# 并且它包含了无需修改的 `validate` 函数
from train_light import validate

def run_heavy_model_training(seed):
    """
    执行一次完整的、针对重型模型的K折交叉验证。
    该模型使用原始3D CT扫描和Swin-UNETR。
    
    Args:
        seed (int): 用于K折数据划分的随机种子。
    """
    print("\n" + "="*80)
    print(f"      WORKFLOW: HEAVY MODEL TRAINING (Seed: {seed})")
    print("="*80 + "\n")

    # --- 1. 强制为重型模型模式设置config ---
    print("--- 强制设置config为重型模型训练模式 ---")
    config.ACTIVE_MODALITIES = ["ct", "urine", "blood", "microbiome", "kegg", "clinical"]
    config.CT_EXTRACTOR = 'swin_unetr'
    # config.TRAINING_STRATEGY = 'staged' # 对重型骨干网络推荐使用分阶段训练
    
    # 为本次重复定义一个唯一的输出目录
    repeat_label = f'repeat_seed_{seed}'
    output_dir = os.path.join(config.BASE_DIR, 'outputs', 'full_experiment', repeat_label)
    
    # 使用正确的输出路径更新config
    config.OUTPUT_DIR = output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建重型模型输出目录: {output_dir}")

    # 重新运行setup以应用被覆盖的设置
    config.setup_experiment_params()
    print("--- 重型模型Config设置完成 ---\n")

    # --- 2. 数据准备与K-Fold划分 ---
    manifest_df = pd.read_csv(config.MANIFEST_PATH)
    all_sample_ids = manifest_df['sample_id'].unique()
    
    # 使用传入的seed以保证每次重复的划分都不同
    kf = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=seed)
    
    fold_splits_info = {}

    # --- 3. K-折交叉验证主循环 ---
    for fold, (train_index, val_index) in enumerate(kf.split(all_sample_ids)):
        fold_num = fold + 1
        print(f"\n==================== [ Heavy Model Fold {fold_num}/{config.K_FOLDS} ] ====================\n")
        
        train_ids = all_sample_ids[train_index]
        val_ids = all_sample_ids[val_index]

        fold_splits_info[f"fold_{fold_num}"] = {'train_ids': train_ids.tolist(), 'val_ids': val_ids.tolist()}

        train_ds = MultiModalDataset(train_ids, manifest_df, config.DATA_PATHS, config.MODALITY_NAMES, mode='train')
        val_ds = MultiModalDataset(val_ids, manifest_df, config.DATA_PATHS, config.MODALITY_NAMES, mode='val')
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=4, pin_memory=True)
        
        # --- 4. 模型初始化与训练 ---
        model = MultiModalModel().to(config.DEVICE)
        
        efficacy_pos_weight = torch.tensor([config.EFFICACY_POS_WEIGHT], device=config.DEVICE)
        toxicity_pos_weight = torch.tensor([config.TOXICITY_POS_WEIGHT], device=config.DEVICE)
        criterions = {
            'efficacy': nn.BCEWithLogitsLoss(pos_weight=efficacy_pos_weight),
            'toxicity': nn.BCEWithLogitsLoss(pos_weight=toxicity_pos_weight),
            'survival': NegativeLogLikelihood()
        }
        
        model_save_path = os.path.join(output_dir, f"best_heavy_model_fold_{fold_num}.pth")
        early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, verbose=True, path=model_save_path)

        # --- 设置分阶段训练或直接训练 ---
        is_staged = (config.TRAINING_STRATEGY == 'staged')
        if is_staged:
            # --- Stage 1: 冻结阶段 (预热) ---
            print("\n--- [Stage 1/2] 训练融合模块与任务头 (骨干网络已冻结) ---\n")
            optimizer, scheduler = setup_optimizers_schedulers(model, is_staged=True, stage='frozen')
            for epoch in range(config.FROZEN_EPOCHS):
                # 沿用之前的train_one_epoch，因为它适用于任何优化器设置
                # 注意: train_one_epoch需要从train_light导入
                from train_light import train_one_epoch 
                train_losses = train_one_epoch(model, train_loader, criterions, optimizer, config.DEVICE)
                val_metrics = validate(model, val_loader, criterions, config.DEVICE)
                print(f"Frozen Epoch {epoch+1}/{config.FROZEN_EPOCHS} | Train Loss: {train_losses['total_for_logging']:.4f} | Val C-idx: {val_metrics.get('survival_c_index', 0):.4f}")
            
            # --- Stage 2: 微调阶段 ---
            print("\n--- [Stage 2/2] 全模型微调 ---\n")
            optimizer, scheduler = setup_optimizers_schedulers(model, is_staged=True, stage='finetune')
            total_epochs = config.FINETUNE_EPOCHS
        else: #直接训练
            print("\n--- [Direct Training] 全模型直接训练 ---\n")
            optimizer, scheduler = setup_optimizers_schedulers(model, is_staged=False, stage='direct')
            total_epochs = config.DIRECT_TRAINING_EPOCHS
            
        # ==================== 内存优化的主训练循环 (用于Stage 2或直接训练) ====================
        for epoch in range(total_epochs):
            # --- 训练阶段 ---
            model.train()
            train_loss_sum = 0
            num_train_batches = 0
            accumulation_steps = 8 # 每2个批次更新一次梯度以节省显存
            optimizer.zero_grad()

            for i, batch in enumerate(train_loader):
                # 前向传播
                outputs = model(batch)
                
                # 计算损失
                efficacy_labels = batch.get('efficacy_label').to(config.DEVICE).view(-1)
                toxicity_labels = batch.get('toxicity_label').to(config.DEVICE).view(-1)
                survival_times = batch.get('survival_time').to(config.DEVICE).view(-1)
                survival_status = batch.get('survival_status').to(config.DEVICE).view(-1)
                
                loss_efficacy = criterions['efficacy'](outputs['efficacy'], efficacy_labels)
                loss_toxicity = criterions['toxicity'](outputs['toxicity'], toxicity_labels)
                loss_survival = criterions['survival'](outputs['survival'], survival_times, survival_status)
                
                loss = (loss_efficacy + loss_toxicity + loss_survival) / accumulation_steps
                
                # 反向传播
                loss.backward()
                
                train_loss_sum += loss.item() * accumulation_steps
                num_train_batches += 1
                
                # 参数更新
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            # 清理训练循环中可能残留的变量和计算图
            del outputs, loss, loss_efficacy, loss_toxicity, loss_survival
            torch.cuda.empty_cache()

            # --- 验证阶段 ---
            val_metrics = validate(model, val_loader, criterions, config.DEVICE)
            torch.cuda.empty_cache()

            # --- 打印日志和执行早停 ---
            avg_train_loss = train_loss_sum / num_train_batches
            print(f"Epoch {epoch+1}/{total_epochs} | Train Loss: {avg_train_loss:.4f} | Val C-idx: {val_metrics.get('survival_c_index', 0):.4f}, Eff-AUC: {val_metrics.get('efficacy_auc', 0):.4f}")
            
            early_stopping_score = (val_metrics.get('efficacy_auc', 0) + val_metrics.get('toxicity_auc', 0) + val_metrics.get('survival_c_index', 0)) / 3.0
            print(f"          EarlyStopping Score (avg): {early_stopping_score:.4f}")

            if scheduler:
                scheduler.step()
            early_stopping(early_stopping_score, model, optimizer, epoch)
            
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        # ==================== 循环结束 ====================

        print(f"\nFold {fold_num} 重型模型训练完成。最佳模型已保存至: {model_save_path}")

    # --- 5. 保存本次重复的K-Fold划分信息 ---
    split_file_path = os.path.join(output_dir, 'kfold_splits.json')
    with open(split_file_path, 'w') as f:
        json.dump(fold_splits_info, f, indent=4)
    print(f"\nK-折划分信息 (seed {seed}) 已保存至: {split_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用特定的随机种子，为一次完整的K折运行训练重型多模态模型。")
    parser.add_argument('--seed', type=int, required=True, help='用于K折划分的随机种子，定义了本次重复。')
    
    args = parser.parse_args()
    
    # 确保您已将原始的train.py重命名为train_light.py
    run_heavy_model_training(seed=args.seed)