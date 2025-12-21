# modules.py

import os
import torch
import torch.nn as nn
from swin_unetr import SwinUNETR
import config  # 导入项目配置文件以获取参数

from debug_utils import debug_tensor_shapes 
class TabularFeatureExtractor(nn.Module):
    """用于处理表格数据的MLP特征提取器"""
    def __init__(self, input_dim, output_dim=config.D_MODEL, dropout_rate=config.DROPOUT_RATE):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class CTFeatureExtractor(nn.Module):
    """使用SwinUNETR作为编码器的CT特征提取器"""
    def __init__(self, in_channels=1, output_dim=config.D_MODEL, pretrained_path=config.SWIN_UNETR_PRETRAINED_PATH):
        super().__init__()
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=2,  # 占位符, 我们只使用编码器
            feature_size=48,
            use_checkpoint=True
        )
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"加载SwinUNETR预训练权重: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=config.DEVICE)
            filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("out.")}
            self.swin_unetr.load_state_dict(filtered_dict, strict=False)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        swin_output_dim = 48 * 16
        self.projection_head = nn.Linear(swin_output_dim, output_dim)
    
    def forward(self, x):
        hidden_states = self.swin_unetr.swinViT(x, normalize=True)
        encoder_output = hidden_states[-1]
        pooled = self.pooling(encoder_output.flatten(2)).squeeze(-1)
        return self.projection_head(pooled)

class ModalityFusionTransformer(nn.Module):
    """
    基于Transformer的融合模块，使用"Missing Token"策略。
    此版本响应config中的消融实验开关。
    """
    def __init__(self, d_model=config.D_MODEL, 
                 nhead=config.TRANSFORMER_NHEAD, num_layers=config.TRANSFORMER_NLAYERS):
        super().__init__()
        self.modality_names = config.MODALITY_NAMES
        
        # 可学习的Tokens和嵌入
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.missing_token = nn.Parameter(torch.randn(1, d_model))
        
        # 根据config决定是否使用模态类型嵌入
        if config.USE_MODALITY_EMBEDDING:
            self.modality_type_embeddings = nn.Embedding(len(self.modality_names), d_model)
        
        # 标准Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    # modules.py -> ModalityFusionTransformer -> forward
    
    def forward(self, extracted_features_list_dict):
        # extracted_features_list_dict 的值是列表，例如: {'ct': [tensor1, tensor2], 'blood': [tensor3, None]}
        
        # 确定 batch_size
        first_key = list(extracted_features_list_dict.keys())[0]
        batch_size = len(extracted_features_list_dict[first_key])
        
        # 确定 device
        device = None
        for name in self.modality_names:
            for item in extracted_features_list_dict[name]:
                if item is not None:
                    device = item.device
                    break
            if device is not None:
                break
        if device is None:
            # 如果所有数据都是None，这是一个特殊情况，理论上不应发生，但为了安全
            # 我们用CLS token作为备用设备来源
            device = self.cls_token.device

        # 准备输入序列，这次是逐个样本构建
        sequences_for_batch = []
        for i in range(batch_size): # 遍历批次中的每个样本
            sequence_components = []
            for j, name in enumerate(self.modality_names):
                feature_tensor = extracted_features_list_dict[name][i]
                
                if feature_tensor is None:
                    component = self.missing_token.squeeze(0) # 使用 missing_token
                else:
                    component = feature_tensor

                if config.USE_MODALITY_EMBEDDING:
                    modality_type_emb = self.modality_type_embeddings(torch.tensor(j).to(device))
                    component = component + modality_type_emb
                
                sequence_components.append(component.unsqueeze(0))
            
            # 将单个样本的所有模态拼接成序列
            sample_sequence = torch.cat(sequence_components, dim=0)
            sequences_for_batch.append(sample_sequence.unsqueeze(0))

        # 将所有样本的序列拼接成一个批次
        modal_sequence = torch.cat(sequences_for_batch, dim=0)
        
        # --- 后续逻辑与之前基本相同 ---
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        full_sequence = torch.cat((cls_tokens, modal_sequence), dim=1)
        
        transformer_output = self.transformer_encoder(full_sequence)
        
        # ★★★ 核心修复：直接返回完整的、未经聚合的三维序列矩阵 ★★★
        # 它的形状是 [batch_size, sequence_length, d_model]
        return transformer_output

class ConcatenateFusion(nn.Module):
    """
    一个简单的基线融合模块 (用于消融实验)。
    它将所有模态特征拼接起来，然后通过一个MLP进行处理。
    """
    def __init__(self, num_active_modalities, d_model=config.D_MODEL, dropout_rate=config.DROPOUT_RATE):
        super().__init__()
        # 计算拼接后的总维度
        fused_dim = num_active_modalities * d_model
        
        # 定义一个MLP来处理拼接后的特征
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model) # 最终输出维度与其他策略保持一致
        )

    def forward(self, extracted_features_dict):
        # 按照config中MODALITY_NAMES的顺序从字典中提取特征
        # 这里的extracted_features_dict已经由model.py处理过，保证不含None
        feature_list = [extracted_features_dict[name] for name in config.MODALITY_NAMES]
        
        # 拼接所有特征向量
        concatenated_features = torch.cat(feature_list, dim=1)
        
        # 通过MLP进行融合
        fused_representation = self.fusion_mlp(concatenated_features)
        return fused_representation

class TaskSpecificDecoder(nn.Module):
    """
    一个任务专属的解码器，使用注意力池化从Transformer的完整输出序列中提取任务相关信息。
    这是 "共享主干，独立分支" 架构的核心组件。
    """
    def __init__(self, d_model=config.D_MODEL, nhead=config.TRANSFORMER_NHEAD, dropout_rate=config.DROPOUT_RATE):
        super().__init__()
        
        # 1. 任务查询向量 (Task Query Vector)
        # 这是一个可学习的参数，它将学会如何向信息序列“提问”以获取当前任务最需要的信息。
        self.task_query_vector = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 2. 多头注意力机制 (Attention Pooling)
        # 我们用它来实现查询向量和信息序列之间的交互。
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # 3. 前馈网络 (Feed-Forward Network)
        # 对注意力池化后的信息进行进一步的非线性处理，增强其表达能力。
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, transformer_output):
        """
        Args:
            transformer_output (Tensor): Transformer编码器的完整输出，
                                        形状为 [batch_size, sequence_length, d_model]。
        """
        # 获取批次大小，并将任务查询向量扩展以匹配该批次
        batch_size = transformer_output.shape[0]
        query = self.task_query_vector.expand(batch_size, -1, -1)
        
        # 执行注意力计算。query是提问者，transformer_output既是信息源(key)也是信息本身(value)。
        attn_output, _ = self.attention(query=query, key=transformer_output, value=transformer_output)
        
        # 注意力输出的形状是 [batch_size, 1, d_model]，我们去掉中间的“1”维度。
        pooled_representation = attn_output.squeeze(1)
        
        # 将池化后的表征送入前馈网络得到最终的任务专属表征。
        final_representation = self.ffn(pooled_representation)
        
        return final_representation