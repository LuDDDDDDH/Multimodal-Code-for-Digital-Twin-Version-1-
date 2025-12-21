# run_explainability_analysis.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import shap
import argparse

# --- 导入您项目中的模块 ---
import config
from model import MultiModalModel
from dataset import MultiModalDataset # 我们需要数据集来为SHAP准备数据

# =============================================================================
# Helper classes for SHAP analysis (采用"分而治之"策略)
# =============================================================================

class DataProcessor:
    """
    一个辅助类，用于在模型的字典格式和SHAP所需的扁平化数组格式之间转换数据。
    """
    def __init__(self, modality_names, tabular_dims):
        self.modality_names = modality_names
        self.tabular_dims = tabular_dims
        
        self.feature_info = {}
        current_pos = 0
        for name in self.modality_names:
            dim = self.tabular_dims.get(name, 0)
            if dim == 0:
                print(f"警告: 模态 '{name}' 在 tabular_dims 中的维度为 0。")
            self.feature_info[name] = {'start': current_pos, 'end': current_pos + dim, 'dim': dim}
            current_pos += dim
        self.total_dims = current_pos
        print(f"DataProcessor 已初始化。总扁平化特征维度: {self.total_dims}")

    def flatten_samples_to_numpy(self, dataset):
        """将数据集中的样本转换为一个大的NumPy数组以供SHAP使用。"""
        num_samples = len(dataset)
        flat_numpy_array = np.zeros((num_samples, self.total_dims), dtype=np.float32)
        sample_ids = []
        labels = {'efficacy': [], 'toxicity': [], 'survival_time': [], 'survival_status': []}

        for i, sample_dict in enumerate(tqdm(dataset, desc="正在为SHAP扁平化样本")):
            if sample_dict is None: continue
            
            sample_ids.append(sample_dict.get('sample_id', f'sample_{i}'))
            for name, info in self.feature_info.items():
                if name in sample_dict and sample_dict[name] is not None:
                    feature_data = sample_dict[name].numpy().flatten()
                    flat_numpy_array[i, info['start']:info['end']] = feature_data
            
            for key in labels:
                labels[key].append(sample_dict[key].item() if key in sample_dict else np.nan)
        
        return flat_numpy_array, sample_ids, {k: np.array(v) for k, v in labels.items()}

class BaseModelWrapper(nn.Module):
    """
    一个基础包装器，用于将扁平化的SHAP输入转换为模型所需的字典格式。
    这个类处理数据格式的转换，而不关心模型的输出。
    """
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.model.eval() # 确保模型处于评估模式

    def forward(self, flat_tensor):
        num_samples = flat_tensor.shape[0]
        batch_dict = {'sample_id': [f"shap_{i}" for i in range(num_samples)]}

        for name, info in self.processor.feature_info.items():
            modality_data_list = []
            for i in range(num_samples):
                data_slice = flat_tensor[i, info['start']:info['end']]
                modality_data_list.append(data_slice)
            
            if not all(x is None for x in modality_data_list):
                 batch_dict[name] = modality_data_list
        
        # 调用原始模型的前向传播，得到包含所有任务输出的字典
        original_output = self.model.forward(batch_dict)
        return original_output

class SingleTaskModelWrapper(nn.Module):
    """
    一个专门为单个任务服务的包装器。
    它使用BaseModelWrapper来获取所有任务的输出，然后只返回我们指定的那一个。
    这为SHAP提供了一个清晰的、单输出的目标。
    """
    def __init__(self, base_wrapper, task_key):
        super().__init__()
        self.base_wrapper = base_wrapper
        self.task_key = task_key

    def forward(self, flat_tensor):
        # 获取所有任务的输出
        all_task_outputs = self.base_wrapper(flat_tensor)
        
        # 从字典中获取当前任务的1D输出, e.g., shape: (batch_size,)
        task_output = all_task_outputs[self.task_key]
        
        # ==================== ★★★ 核心修复点 ★★★ ====================
        # 将输出从 (batch_size,) 重塑为 (batch_size, 1) 以满足SHAP的要求
        return task_output.view(-1, 1)


# =============================================================================
# 主执行函数
# =============================================================================

def main(fold_number, input_dir):
    """
    对来自特定实验重复和特定折叠的特定模型，运行完整的SHAP可解释性分析。
    """
    print("="*80)
    print(f"      工作流: SHAP可解释性分析")
    print(f"      目标: 位于目录 {os.path.basename(input_dir)} 中的第 {fold_number} 折")
    print("="*80)

    # --- 1. 定义路径和SHAP参数 ---
    best_model_path = os.path.join(input_dir, f"best_light_model_fold_{fold_number}.pth")
    kfold_splits_path = os.path.join(input_dir, 'kfold_splits.json')
    output_filename = f"explanation_data_package_fold_{fold_number}.joblib"
    output_path = os.path.join(input_dir, output_filename)
    
    # SHAP参数
    N_BACKGROUND_SAMPLES = 100 # 用于解释器的背景样本数
    SHAP_BATCH_SIZE = 32       # 计算SHAP值时的批大小以管理内存

    # --- 2. 为轻量模型模式设置config ---
    print("\n--- 正在为轻量模型模式设置config ---")
    config.ACTIVE_MODALITIES = ["ct_vector", "urine", "blood", "microbiome", "kegg", "clinical"]
    if 'ct' in config.ACTIVE_MODALITIES:
        config.ACTIVE_MODALITIES.remove('ct')
    config.setup_experiment_params()
    
    # --- 3. 加载训练好的轻量模型 ---
    print(f"\n--- 正在加载模型: {os.path.basename(best_model_path)} ---")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"错误: 在 {best_model_path} 未找到模型文件")

    original_model = MultiModalModel().to(config.DEVICE)
    checkpoint = torch.load(best_model_path, map_location=config.DEVICE)
    original_model.load_state_dict(checkpoint['model_state_dict'])
    original_model.eval()
    print("原始模型加载成功。")

    # --- 4. 准备该特定折叠的数据分割 ---
    print("\n--- 正在准备该折叠的训练/验证数据分割 ---")
    if not os.path.exists(kfold_splits_path):
        raise FileNotFoundError(f"错误: 在 {kfold_splits_path} 未找到K折分割文件")
    with open(kfold_splits_path, 'r') as f:
        splits = json.load(f)
    
    fold_key = f"fold_{fold_number}"
    train_ids, val_ids = splits[fold_key]['train_ids'], splits[fold_key]['val_ids']
    
    manifest_df = pd.read_csv(config.MANIFEST_PATH)

    # --- 关键: 使用正确的、该折叠专属的CT特征 ---
    ct_vector_path = os.path.join(input_dir, f"precomputed_ct_features_fold_{fold_number}.csv")
    if not os.path.exists(ct_vector_path):
        raise FileNotFoundError(f"错误: 在 {ct_vector_path} 未找到第 {fold_number} 折的CT向量文件")
    
    original_ct_path = config.DATA_PATHS.get('ct_vector')
    config.DATA_PATHS['ct_vector'] = ct_vector_path
    print(f"正在使用该折叠专属的CT特征: {os.path.basename(ct_vector_path)}")

    # 创建数据集
    train_dataset = MultiModalDataset(train_ids, manifest_df, config.DATA_PATHS, config.MODALITY_NAMES, mode='val') # 无增强
    val_dataset = MultiModalDataset(val_ids, manifest_df, config.DATA_PATHS, config.MODALITY_NAMES, mode='val')

    if original_ct_path:
        config.DATA_PATHS['ct_vector'] = original_ct_path
        
    # --- 5. 初始化DataProcessor ---
    processor = DataProcessor(config.MODALITY_NAMES, config.TABULAR_FEATURE_DIMS)
    
    # --- 6. 执行SHAP分析 (采用新的"分而治之"策略) ---
    print("\n--- 开始SHAP值计算 (采用逐任务策略) ---")
    
    # a) 准备背景数据
    print(f"正在从训练集中准备 {N_BACKGROUND_SAMPLES} 个背景样本...")
    background_indices = np.random.choice(len(train_dataset), N_BACKGROUND_SAMPLES, replace=False)
    background_samples = [train_dataset[i] for i in background_indices]
    background_data_numpy, _, _ = processor.flatten_samples_to_numpy(background_samples)
    background_data_torch = torch.from_numpy(background_data_numpy).to(config.DEVICE)
    
    # b) 准备待解释的验证数据
    print("正在准备待解释的验证集样本...")
    X_val_numpy, val_sample_ids, val_labels = processor.flatten_samples_to_numpy(val_dataset)

    # 修复: 确保样本数为偶数以兼容DeepExplainer
    if X_val_numpy.shape[0] % 2 != 0:
        print(f"警告: 验证集样本数为奇数 ({X_val_numpy.shape[0]})。将丢弃最后一个样本以保证兼容性。")
        X_val_numpy = X_val_numpy[:-1]
        val_sample_ids = val_sample_ids[:-1]
        for key in val_labels: val_labels[key] = val_labels[key][:-1]
    X_val_torch = torch.from_numpy(X_val_numpy).to(config.DEVICE)

    # c) 循环为每个任务初始化解释器并运行分析
    base_model_wrapper = BaseModelWrapper(original_model, processor).to(config.DEVICE)
    tasks_to_explain = ['efficacy', 'toxicity', 'survival']
    shap_values_by_task = {}

    for task in tasks_to_explain:
        print(f"\n--- 正在解释任务: {task.upper()} ---")
        
        # 1. 为当前任务创建一个单输出的包装器
        single_task_model = SingleTaskModelWrapper(base_model_wrapper, task)
        
        # 2. 使用DeepExplainer，因为它现在可以安全地处理清晰的单输出模型
        explainer = shap.GradientExplainer(single_task_model, background_data_torch)
        
        # 3. 计算SHAP值
        print(f"正在使用批大小 {SHAP_BATCH_SIZE} 计算SHAP值...")
        num_batches = int(np.ceil(X_val_torch.shape[0] / SHAP_BATCH_SIZE))
        task_shap_values_list = []

        for i in tqdm(range(num_batches), desc=f"SHAP批处理 for {task}"):
            start_idx = i * SHAP_BATCH_SIZE
            end_idx = min((i + 1) * SHAP_BATCH_SIZE, X_val_torch.shape[0])
            input_batch = X_val_torch[start_idx:end_idx]
            
            # explainer.shap_values 现在返回一个单一的 numpy 数组，而不是列表
            shap_values_batch = explainer.shap_values(input_batch)
            task_shap_values_list.append(shap_values_batch)
            
        # 4. 合并所有批次的结果
        shap_values_by_task[task] = np.concatenate(task_shap_values_list, axis=0)
        print(f"任务 '{task}' 的SHAP值计算完成。形状: {shap_values_by_task[task].shape}")

    print("\n所有任务已成功解释！")

    # --- 7. 将所有分析组件保存到单个文件 ---
    print(f"\n--- 正在将分析包保存至 {os.path.basename(output_path)} ---")
    
    # 按可用模态对样本进行分组
    sample_groups = {}
    for i, sample in enumerate(tqdm(val_dataset, desc="正按模态对样本进行分组")):
        if sample is None or i >= len(val_sample_ids): continue # 确保索引在范围内
        available_modalities = tuple(sorted([key for key in sample.keys() if key in config.MODALITY_NAMES]))
        if available_modalities not in sample_groups:
            sample_groups[available_modalities] = []
        sample_groups[available_modalities].append(i)
    sample_groups_str_keys = {",".join(key): val for key, val in sample_groups.items()}

    explanation_results = {
        'model_path': best_model_path,
        'fold_id': fold_number,
        'sample_ids': val_sample_ids,
        'sample_groups': sample_groups_str_keys,
        'feature_info': processor.feature_info,
        'shap_values': shap_values_by_task,
        'true_labels': val_labels
    }
    
    joblib.dump(explanation_results, output_path)
    
    print("\n" + "="*80)
    print("成功！SHAP分析包已保存至:")
    print(os.path.abspath(output_path))
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对一个训练好的轻量模型进行SHAP分析。")
    parser.add_argument('--fold', type=int, required=True, 
                        help='要分析的折叠编号 (例如, 1, 2, ...)。')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='特定重复的目录 (例如, .../repeat_seed_42/)，其中包含模型和数据。')
    
    args = parser.parse_args()

    main(fold_number=args.fold, input_dir=args.input_dir)