# dataset.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# 导入 MONAI 的核心组件
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    ResizeWithPadOrCropd,
    ToTensord,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

# 导入项目配置文件
import config

# --- get_train_transforms_for_npy 和 get_val_transforms_for_npy 保持不变 ---
def get_train_transforms_for_npy():
    """为【裸 .npy】文件构建的训练集MONAI流水线。"""
    transforms_list = [
        ScaleIntensityRanged(
            keys=["image"], a_min=config.LIVER_WINDOW_MIN, a_max=config.LIVER_WINDOW_MAX,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=config.CT_IMG_SIZE, mode="constant"),
    ]
    
    if config.USE_CT_AUGMENTATION:
        transforms_list.extend([
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ])
    
    transforms_list.append(ToTensord(keys=["image"]))
    return Compose(transforms_list)

def get_val_transforms_for_npy():
    """为【裸 .npy】文件构建的验证/测试集MONAI流水线。"""
    return Compose([
        ScaleIntensityRanged(
            keys=["image"], a_min=config.LIVER_WINDOW_MIN, a_max=config.LIVER_WINDOW_MAX,
            b_min=0.0, b_max=1.0, clip=True
        ),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=config.CT_IMG_SIZE, mode="constant"),
        ToTensord(keys=["image"]),
    ])
# -------------------------------------------------------------------------


class MultiModalDataset(Dataset):
    """
    用于加载多模态、不完整数据的数据集。
    (已更新以支持 .npy 文件和扁平化目录结构)
    """
    def __init__(self, sample_ids, manifest_df, data_paths, modality_names, 
                 microbiome_scaler=None, mode='train', 
                 selected_features_map=None):
        self.sample_ids = sample_ids
        self.manifest = manifest_df.set_index('sample_id')
        self.data_paths = data_paths
        self.modality_names = modality_names
        self.microbiome_scaler = microbiome_scaler
        self.selected_features_map = selected_features_map or {}
        
        if mode == 'train':
            self.ct_transform = get_train_transforms_for_npy()
            print(f"使用【训练 .npy】数据流水线 (数据增强: {'启用' if config.USE_CT_AUGMENTATION else '关闭'})。")
        else:
            self.ct_transform = get_val_transforms_for_npy()
            print(f"使用【验证/测试 .npy】数据流水线 (无增强) for mode: '{mode}'.")

        self.loaded_tabular_data = {}
        for modality, path in self.data_paths.items():
            if modality != "ct" and modality in self.modality_names:
                print(f"预加载激活的表格模态: {modality}")
                try:
                    df = pd.read_csv(path).set_index('sample_id')
                except UnicodeDecodeError:
                    print(f"    - 文件 {os.path.basename(path)} 编码非UTF-8, 尝试使用'gbk'解码...")
                    df = pd.read_csv(path, encoding='gbk').set_index('sample_id')
                
                if modality in self.selected_features_map:
                    selected_cols = self.selected_features_map[modality]
                    valid_cols = [col for col in selected_cols if col in df.columns]
                    if len(valid_cols) != len(selected_cols):
                        print(f"警告: {modality} 模态的一些已选特征在CSV中不存在。")
                    print(f"    -> 已筛选 {len(valid_cols)} 个特征。")
                    df = df[valid_cols]
                
                self.loaded_tabular_data[modality] = df.astype(np.float32)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        data_dict = {}
        
        if sample_id not in self.manifest.index:
            print(f"警告: sample_id '{sample_id}' 未在 manifest 中找到。跳过此样本。")
            return None

        # <--- 修改开始: 加载所有任务的标签 ---
        # 假设 manifest.csv 中有这些列名
        sample_info = self.manifest.loc[sample_id]
        data_dict['sample_id'] = sample_id
        
        # 任务1: 疗效 (原 label)
        if 'efficacy_label' in sample_info:
            data_dict['efficacy_label'] = torch.tensor(sample_info['efficacy_label'], dtype=torch.float)
        
        # 任务2: 不良反应
        if 'toxicity_label' in sample_info:
            data_dict['toxicity_label'] = torch.tensor(sample_info['toxicity_label'], dtype=torch.float)
            
        # 任务3: 生存
        if 'survival_time' in sample_info and 'survival_status' in sample_info:
            data_dict['survival_time'] = torch.tensor(sample_info['survival_time'], dtype=torch.float)
            data_dict['survival_status'] = torch.tensor(sample_info['survival_status'], dtype=torch.long) # status通常是整数
        # <--- 修改结束 ---
        # # 新增: 加载二分类生存标签
        # if 'survival_label' in sample_info:
        #     data_dict['survival_label'] = torch.tensor(sample_info['survival_label'], dtype=torch.float)
        # # ==================== ★ 修改结束 ★ ====================

        for modality in self.modality_names:
            if modality == "ct":
                # ... (这部分加载CT数据的逻辑保持完全不变) ...
                potential_filenames = [
                    f"{sample_id}.npy",
                    f"{sample_id}_ct.npy"
                ]
                image_path = None
                for fname in potential_filenames:
                    path_to_check = os.path.join(self.data_paths['ct'], fname)
                    if os.path.exists(path_to_check):
                        image_path = path_to_check
                        break
                
                if image_path:
                    try:
                        ct_array = np.load(image_path).astype(np.float32)
                        
                        if ct_array.ndim == 3:
                            ct_array = np.expand_dims(ct_array, axis=0) 
                        
                        if ct_array.ndim != 4 or ct_array.shape[0] != 1:
                            print(f"错误: 样本 '{sample_id}' 的CT数组无法转换为 (1, D, H, W) 格式 (当前 shape: {ct_array.shape})。跳过此CT。")
                            continue
                        
                        ct_data_for_transform = {"image": ct_array}
                        processed_ct = self.ct_transform(ct_data_for_transform) 
                        data_dict[modality] = processed_ct["image"]

                    except Exception as e:
                        import traceback
                        print(f"警告: MONAI处理文件 {sample_id} ({image_path}) 时发生错误: {e}")
            else: # 处理表格数据 (逻辑不变)
                df = self.loaded_tabular_data.get(modality)
                if df is not None and sample_id in df.index:
                    features = df.loc[sample_id].values
                    if modality == "microbiome" and self.microbiome_scaler is not None:
                        features = self.microbiome_scaler.transform(features.reshape(1, -1)).flatten()
                    data_dict[modality] = torch.tensor(np.atleast_1d(features), dtype=torch.float32)
        
        return data_dict


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return {}
        
    collated_batch = {}
    
    # <--- 修改开始: 定义所有可能的键，包括所有新标签 ---
    # 使用 set union 来合并模态名和所有可能的标签键
    label_keys = {'sample_id', 'efficacy_label', 'toxicity_label', 'survival_time', 'survival_status'}
    # label_keys = {'sample_id', 'efficacy_label', 'toxicity_label', 'survival_label'}
    all_keys = set(config.MODALITY_NAMES) | label_keys
    # <--- 修改结束 ---
    
    for key in all_keys:
        # 检查批次中至少有一个样本包含这个键
        if not any(key in item for item in batch):
            continue

        if key in config._ALL_MODALITY_NAMES:
            # 模态数据的处理逻辑不变
            modality_list = [item.get(key) for item in batch]
            collated_batch[key] = modality_list
        
        elif key == 'sample_id':
             collated_batch['sample_id'] = [item['sample_id'] for item in batch]
        
        # <--- 修改开始: 为每个标签键分别进行堆叠 ---
        elif key in label_keys:
            # 将所有存在的标签收集起来并堆叠
            # item.get(key) 会在样本缺少该标签时返回 None，我们需要过滤掉
            valid_labels = [item[key] for item in batch if key in item]
            if valid_labels:
                 # .view(-1) 确保它们都是一维的
                collated_batch[key] = torch.stack([lbl.view(-1) for lbl in valid_labels])
    
    # 兼容性修改：将原来的 'label' 键指向 'efficacy_label'
    if 'efficacy_label' in collated_batch:
        collated_batch['label'] = collated_batch['efficacy_label']
    # <--- 修改结束 ---
            
    return collated_batch