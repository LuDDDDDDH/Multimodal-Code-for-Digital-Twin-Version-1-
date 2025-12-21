# model.py (升级版 - 包含任务特定适配器)

import torch
import torch.nn as nn

# 从modules.py中导入所有可能用到的模型组件
from modules import (
    TabularFeatureExtractor, 
    CTFeatureExtractor, 
    ModalityFusionTransformer, 
    ConcatenateFusion
)
# 导入项目配置文件
import config

# (如果存在) 导入调试工具
# from debug_utils import debug_tensor_shapes

class MultiModalModel(nn.Module):
    """
    一个灵活的、可根据配置动态构建的多任务多模态模型。
    ★★★ 已升级为软参数共享架构 (Soft Parameter Sharing) ★★★
    通过为每个任务添加独立的适配器层，来缓解多任务学习中的负向迁移，
    旨在最大化每个任务的性能。
    """
    def __init__(self):
        super().__init__()
        
        # --- 1. 共享的特征提取器 (逻辑不变) ---
        print("\n" + "="*20 + " MODEL CONSTRUCTION (MULTI-TASK w/ Adapters) " + "="*20)
        print(f"--- Building model with ACTIVE MODALITIES -> {config.MODALITY_NAMES} ---\n")
        
        self.feature_extractors = nn.ModuleDict()
        
        # a) 表格数据特征提取器
        for name, dim in config.TABULAR_FEATURE_DIMS.items():
            if name in config.ACTIVE_MODALITIES:
                print(f"  - Initializing Tabular Extractor for '{name}' (input_dim: {dim})")
                self.feature_extractors[name] = TabularFeatureExtractor(input_dim=dim)
        
        # b) CT影像特征提取器
        if 'ct' in config.ACTIVE_MODALITIES:
            if config.CT_EXTRACTOR == 'swin_unetr':
                print(f"  - Initializing CT Extractor: '{config.CT_EXTRACTOR}'")
                self.feature_extractors['ct'] = CTFeatureExtractor()
            else:
                raise ValueError(f"Unknown CT extractor specified in config: {config.CT_EXTRACTOR}")

        # --- 2. 共享的融合模块 (逻辑不变) ---
        print(f"\n--- Using FUSION STRATEGY -> {config.FUSION_STRATEGY} ---")
        
        if len(config.ACTIVE_MODALITIES) > 1:
            if config.FUSION_STRATEGY == 'transformer_missing_token':
                self.fusion_module = ModalityFusionTransformer()
            elif config.FUSION_STRATEGY == 'concatenate':
                num_active_modalities = len(config.ACTIVE_MODALITIES)
                self.fusion_module = ConcatenateFusion(num_active_modalities=num_active_modalities, d_model=config.D_MODEL)
            else:
                raise ValueError(f"Unknown fusion strategy specified in config: {config.FUSION_STRATEGY}")
        else:
            self.fusion_module = None
            print("--- Running in UNIMODAL mode, fusion module is disabled. ---")

        # ==================== ★ 架构升级：共享主干，独立分支 ★ ====================
        # --- 3. 任务专属的独立解码器 (Task-specific Independent Decoders) ---
        # 我们用功能更强大的解码器替换掉简单的适配器。
        # 每个解码器都能独立地从完整的信息序列中提取对自己最重要的信息。
        print("\\n--- Initializing Task-specific INDEPENDENT DECODERS and Heads ---")
        
        # 从 modules.py 导入我们刚刚创建的新模块
        from modules import TaskSpecificDecoder
        
        self.efficacy_decoder = TaskSpecificDecoder()
        self.toxicity_decoder = TaskSpecificDecoder()
        self.survival_decoder = TaskSpecificDecoder()
        
        # --- 4. 最终的任务头 (Task Heads) ---
        # 任务头现在接收来自各自解码器的、经过任务个性化提炼的特征。
        # 输入维度仍然是 D_MODEL，因为我们的解码器输出维度与之保持一致。
        self.efficacy_head = nn.Linear(config.D_MODEL, 1)
        self.toxicity_head = nn.Linear(config.D_MODEL, 1)
        self.survival_head = nn.Linear(config.D_MODEL, 1)
        
        print("Independent Decoders for Efficacy, Toxicity, and Survival initialized.")
        print("="*71 + "\\n")
        # ==================== ★ 升级结束 ★ ====================
        
    def forward(self, batch):
        # """
        # 前向传播。
        # ★★★ 已升级为 "共享主干，独立分支" 架构 ★★★
        # 1. 融合模块输出完整的序列信息 (transformer_output)。
        # 2. 该序列信息被分发给三个独立的、任务专属的解码器。
        # 3. 每个解码器独立地提取对自己任务最重要的信息，打破了信息瓶颈。
        # """
        # --- 1. 初始化与模态随机失活 (逻辑不变) ---
        batch_size = len(batch['sample_id'])
        device = next(self.parameters()).device
        expected_dtype = next(self.parameters()).dtype

        if self.training and config.MODALITY_DROPOUT_RATE > 0:
            feature_source_dict = {}
            for name in config.MODALITY_NAMES:
                if name in batch:
                    feature_source_dict[name] = list(batch[name])
            
            for i in range(batch_size):
                available_modalities = [
                    name for name in config.MODALITY_NAMES 
                    if name in feature_source_dict and feature_source_dict[name][i] is not None
                ]
                
                if len(available_modalities) > 1:
                    modalities_to_drop = []
                    for modality_name in available_modalities:
                        if torch.rand(1).item() < config.MODALITY_DROPOUT_RATE:
                            modalities_to_drop.append(modality_name)
                    
                    if len(modalities_to_drop) == len(available_modalities):
                        import random
                        modalities_to_drop.remove(random.choice(modalities_to_drop))

                    for modality_name in modalities_to_drop:
                        feature_source_dict[modality_name][i] = None
        else:
            feature_source_dict = batch

        # --- 2. 逐模态特征提取 (逻辑不变) ---
        extracted_features_list_dict = {}
        for name in config.MODALITY_NAMES:
            modality_data_list = feature_source_dict.get(name)
            
            if modality_data_list is None:
                extracted_features_list_dict[name] = [None] * batch_size
                continue
                
            final_features_for_modality = [None] * batch_size
            valid_tensors, valid_indices = [], []
            for i, tensor in enumerate(modality_data_list):
                if tensor is not None:
                    valid_tensors.append(tensor.to(device=device, dtype=expected_dtype))
                    valid_indices.append(i)
            
            if valid_tensors:
                valid_batch = torch.stack(valid_tensors, dim=0)
                extracted = self.feature_extractors[name](valid_batch)
                for i, original_idx in enumerate(valid_indices):
                    final_features_for_modality[original_idx] = extracted[i]
            
            extracted_features_list_dict[name] = final_features_for_modality

        # --- 3. 多模态融合 (信息交互) ---
        if self.fusion_module is not None:
            # ★★★ 关键改变 1: 接收完整的、包含所有模态信息的序列输出 ★★★
            # transformer_output 的形状为: [batch_size, sequence_length, d_model]
            transformer_output = self.fusion_module(extracted_features_list_dict)
        else: # 单模态情况
            single_modality_name = config.MODALITY_NAMES[0]
            feature_list = extracted_features_list_dict[single_modality_name]
            
            # 为了与多模态输出的格式保持一致，我们也将其构建为序列形式
            transformer_output = torch.zeros(batch_size, 1, config.D_MODEL, device=device, dtype=expected_dtype)
            valid_features, valid_indices = [], []
            for i, feature in enumerate(feature_list):
                if feature is not None:
                    valid_features.append(feature)
                    valid_indices.append(i)
            
            if valid_features:
                # unsqueeze(1) 是为了创造一个长度为1的序列维度
                stacked_features = torch.stack(valid_features).unsqueeze(1)
                transformer_output[valid_indices] = stacked_features

        # --- 4. ★ 核心改变 2: 独立解码，打破瓶颈 ★ ---
        # 我们将完整的、富含信息的 transformer_output 序列分发给每个独立的任务解码器。
        # 每个解码器都会返回一个为自己任务量身定制的一维表征。
        efficacy_representation = self.efficacy_decoder(transformer_output)
        toxicity_representation = self.toxicity_decoder(transformer_output)
        survival_representation = self.survival_decoder(transformer_output)
        
        # --- 5. 任务头预测 ---
        # 每个任务头现在都在自己的、高度相关的专属表征上进行预测。
        efficacy_logits = self.efficacy_head(efficacy_representation)
        toxicity_logits = self.toxicity_head(toxicity_representation)
        survival_log_risk = self.survival_head(survival_representation)

        # --- 6. 格式化输出 (逻辑不变) ---
        outputs = {
            'efficacy': efficacy_logits.squeeze(1),
            'toxicity': toxicity_logits.squeeze(1),
            'survival': survival_log_risk.squeeze(1)
        }
        
        return outputs