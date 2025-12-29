# config.py
import torch
import os
import json

# ==============================================================================
# 0. 工作流控制中心
# ==============================================================================
WORKFLOW = 'MULTI_MODAL_TRAINING' 
# ==============================================================================
# 1. 实验控制中心
# ==============================================================================
EXPERIMENT_MODE = 'FULL_TRAINING'# 模式开关: 'FULL_TRAINING' (保存模型) 或 'EXPLORATION' (不保存模型，用于快速对比)
# --- 新增开关 ---
USE_PRESELECTED_FEATURES = False # 设置为 False 来使用全部特征
# USE_PRESELECTED_FEATURES = True # 设置为 True 来使用筛选后的特征
# a) 选择要使用的模态 (用于模态贡献消融)
# ACTIVE_MODALITIES = ["ct", "urine", "blood", "microbiome", "kegg", "clinical"]

ACTIVE_MODALITIES = ["ct_vector", "urine", "blood", "microbiome", "kegg", "clinical"]
# ACTIVE_MODALITIES = ["urine", "blood",  "kegg"] #生物学机制
# b) 选择融合策略 (用于对比融合方法)
#    - 'transformer_missing_token': Transformer + Missing Token策略
#    - 'concatenate': 简单的特征拼接 + MLP融合
#    - 'transformer_attention_mask': Transformer + 注意力掩码策略 (待实现)
FUSION_STRATEGY = 'transformer_missing_token'
# c) 选择CT特征提取器
# - 'swin_unetr': 使用SwinUNETR
# - (未来可扩展其他选项)
CT_EXTRACTOR = 'swin_unetr'
# d) 消融Transformer内部组件
#    - USE_MODALITY_EMBEDDING: 是否添加模态类型嵌入 (消融模态身份)
#    - AGGREGATION_STRATEGY: 如何聚合Transformer输出 ('cls' 或 'mean_pool') (消融聚合标记)
USE_MODALITY_EMBEDDING = True
AGGREGATION_STRATEGY = 'cls'
# e) 选择训练策略 (消融训练策略)
#    - 'staged': 先冻结backbone，后微调
#    - 'direct': 从一开始就联合微调所有层
# TRAINING_STRATEGY = 'staged'
TRAINING_STRATEGY = 'direct' 
# f) 是否使用CT数据增强 (消融数据增强)
USE_CT_AUGMENTATION = True



# ==============================================================================
# 2. 静态参数
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 或者 os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ==============================================================================
# 动态输出目录管理 (替换旧的OUTPUT_DIR定义)
# ==============================================================================
# ★ 通过修改这个变量来切换实验 ★
# 例如: 'heavy_model_training', 'lightweight_model_training', 'feature_selection'
# EXPERIMENT_NAME = 'heavy_model_training' #使用所有特征运行完整模型
EXPERIMENT_NAME = 'lightweight_model_training'

# 动态构建输出目录路径
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', EXPERIMENT_NAME)

# 确保目录存在，如果不存在则创建
if not os.path.exists(OUTPUT_DIR):
    print(f"输出目录不存在，正在创建: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR)
# ==============================================================================

_ALL_MODALITY_NAMES = ["ct", "ct_vector", "urine", "blood", "microbiome", "kegg", "clinical"]
# _ALL_MODALITY_NAMES = ["ct", "urine", "blood", "microbiome", "kegg", "clinical"]
_ALL_ORIGINAL_DIMS = {
    "urine": 1728, "blood": 1512, "microbiome": 1729,"ct_vector": 256, 
    "kegg": 419, "clinical": 35 
}

# In config.py
MANIFEST_PATH = os.path.join(DATA_DIR, 'manifest_mock_labels.csv')
# MANIFEST_PATH = os.path.join(DATA_DIR, 'manifest.csv')
DATA_PATHS = {
    "ct_vector": os.path.join(DATA_DIR, "precomputed_ct_features.csv"), 
    "urine": os.path.join(DATA_DIR, "urine_all_samples.csv"),
    "blood": os.path.join(DATA_DIR, "blood_all_samples.csv"),
    "microbiome": os.path.join(DATA_DIR, "microbiome_all_samples.csv"),
    "kegg": os.path.join(DATA_DIR, "KEGG_microbiome_data.csv"),
    "clinical": os.path.join(DATA_DIR, "clinical_all_samples.csv"),
    "ct": os.path.join(DATA_DIR, "qilu_artery_segpreprocessed_ct_96x96x96")
}
SWIN_UNETR_PRETRAINED_PATH = os.path.join(DATA_DIR, "best_swinunetr_classifier.pth")

D_MODEL = 256
TRANSFORMER_NHEAD = 4
TRANSFORMER_NLAYERS = 2
DROPOUT_RATE = 0.45
# ==================== ★ 新增代码开始 (样本不平衡处理) ★ ====================
# 未来请根据真实数据分析结果修改以下权重值。
# 计算公式: pos_weight = 多数类样本数 / 少数类样本数
# 如果数据是平衡的，将这些值设为 1.0 即可。
# 注意：这里的权重是针对正类（label=1）的。如果少数类是负类（label=0），
# 需要调整BCEWithLogitsLoss的用法，但pos_weight策略通常假设正类是少数类。

# 示例值 (请替换)
EFFICACY_POS_WEIGHT = 1  # 假设疗效任务中，有效(1)是少数类
TOXICITY_POS_WEIGHT = 10.0    # 假设毒性任务中，发生(1)是少数类
# SURVIVAL_POS_WEIGHT = 2.3333 改回生存预测
# ==================== ★ 新增代码结束 ★ ====================

# ================== ★ 新增代码开始 (模态随机失活 Modality Dropout) ★ ==================
# 在训练期间，以这个概率随机丢弃一个样本中可用的模态。
# 这是一个正则化技术，用于提升模型在数据不全时的鲁棒性。
# 设为 0.0 则关闭此功能。推荐值范围: 0.1 ~ 0.3
MODALITY_DROPOUT_RATE = 0.25
# ================== ★ 新增代码结束 ★ ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_FOLDS = 3
FROZEN_EPOCHS = 5 # 冻结backbone的总轮数 (Staged Training)
FINETUNE_EPOCHS = 10 #目前为了快速测试，设置为10轮
# 直接训练的总轮数 (Direct Training)
# 当 TRAINING_STRATEGY 设置为 'direct' 时使用
DIRECT_TRAINING_EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 3e-5 # 从 1e-4 大幅降低到一个更保守、更稳定的值
FINETUNE_LR_RATIO = 0.1
WEIGHT_DECAY = 2e-4
EARLY_STOPPING_PATIENCE = 10
CT_IMG_SIZE = (96, 96, 96)  
LIVER_WINDOW_MIN = -35
LIVER_WINDOW_MAX = 115

# ==============================================================================
# 3. 动态参数生成函数
# ==============================================================================
def setup_experiment_params():
    global MODALITY_NAMES, TABULAR_FEATURE_DIMS, SELECTED_FEATURES_MAP, _ALL_TABULAR_DIMS
    
    SELECTED_FEATURES_MAP, _ALL_TABULAR_DIMS = {}, {}

    for name, original_dim in _ALL_ORIGINAL_DIMS.items():
        feature_file = os.path.join(OUTPUT_DIR, f"selected_features_{name}.json")
        
        # ★★★ 修改后的逻辑 ★★★
        # 只有当开关打开，并且是训练工作流，并且文件存在时，才加载
        if USE_PRESELECTED_FEATURES and WORKFLOW == 'MULTI_MODAL_TRAINING' and os.path.exists(feature_file):
            print(f"【使用已筛选特征】加载 {name} 的特征列表从: {feature_file}")
            with open(feature_file, 'r') as f:
                SELECTED_FEATURES_MAP[name] = json.load(f)
            _ALL_TABULAR_DIMS[name] = len(SELECTED_FEATURES_MAP[name])
        else:
            # 在其他所有情况下（包括我们当前的目标），都使用原始维度
            if WORKFLOW == 'MULTI_MODAL_TRAINING' and not USE_PRESELECTED_FEATURES:
                 print(f"【使用全部特征】模态 {name} 将使用其全部 {original_dim} 个原始特征。")
            _ALL_TABULAR_DIMS[name] = original_dim
    
    if WORKFLOW == 'SINGLE_MODALITY_SHAP':
        MODALITY_NAMES, TABULAR_FEATURE_DIMS = [], {}
    else:
        MODALITY_NAMES = [name for name in _ALL_MODALITY_NAMES if name in ACTIVE_MODALITIES]
        TABULAR_FEATURE_DIMS = {name: dim for name, dim in _ALL_TABULAR_DIMS.items() if name in MODALITY_NAMES}

    print("="*60 + f"\n              WORKFLOW: {WORKFLOW}\n" + "="*60)
    if WORKFLOW == 'MULTI_MODAL_TRAINING':
        print("              MULTI_MODAL_TRAINING CONFIGURATION" + "\n" + "-"*60)
        print(f"ACTIVE_MODALITIES: {MODALITY_NAMES}\n\nUSING FEATURE DIMENSIONS:")
        for name in MODALITY_NAMES:
            if name in _ALL_TABULAR_DIMS:
                 print(f"  - {name}: {_ALL_TABULAR_DIMS[name]} (Original: {_ALL_ORIGINAL_DIMS.get(name)})")
        print("-" * 60)
    elif WORKFLOW == 'SINGLE_MODALITY_SHAP':
        print("              SINGLE_MODALITY_SHAP CONFIGURATION" + "\n" + "-"*60)
        print(f"MODALITIES TO ANALYZE: {SHAP_CONFIG['MODALITIES_TO_SELECT']}")
        print(f"TOP K FEATURES TO SELECT: {SHAP_CONFIG['TOP_K_FEATURES']}")
        print("-" * 60)

MODALITY_NAMES, TABULAR_FEATURE_DIMS, SELECTED_FEATURES_MAP, _ALL_TABULAR_DIMS = [], {}, {}, {}