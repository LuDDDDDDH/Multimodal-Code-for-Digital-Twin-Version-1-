# utils.py
import shutil
import torch
import torch.nn as nn
import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
import os
import config
# =============================================================================
# 1. 生存分析损失函数 (Cox Proportional Hazards Loss)
# =============================================================================

class NegativeLogLikelihood(nn.Module):
    """
    Cox比例风险模型的负对数偏似然损失函数。
    这个损失函数的目标不是预测准确的生存时间，而是对患者的风险进行正确排序。
    它会惩罚那些“事件发生时间早，但模型预测风险低”的样本。
    """
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, log_risks, event_times, event_status):
        """
        参数:
        - log_risks (Tensor): 模型的输出, shape [batch_size]。代表每个样本的对数风险。
        - event_times (Tensor): 事件发生或删失的时间, shape [batch_size]。
        - event_status (Tensor): 事件是否发生的指示器 (1=事件, 0=删失), shape [batch_size]。
        """
        # 首先，按照事件发生时间的降序对所有样本进行排序
        # 这使得我们可以方便地为每个发生事件的样本定义其“风险集”
        sorted_indices = torch.argsort(event_times, descending=True)
        log_risks_sorted = log_risks[sorted_indices]
        event_status_sorted = event_status[sorted_indices]

        # 计算每个时间点上，风险集中所有样本风险的累积和（从时间最晚的开始）
        # torch.cumsum 是一个高效的实现方式
        log_sum_exp = torch.logcumsumexp(log_risks_sorted, dim=0)

        # 筛选出那些真正发生了事件的样本 (event_status == 1)
        # 只有这些样本对损失函数有贡献
        observed_events_mask = (event_status_sorted == 1)
        
        # 提取发生事件样本的 log_risk 和对应的风险集对数和
        log_risks_observed = log_risks_sorted[observed_events_mask]
        log_sum_exp_observed = log_sum_exp[observed_events_mask]

        # 计算损失：sum(log_risk_i - log(sum(exp(log_risk_j)))) for all observed events i
        # 其中 j 是在 i 事件发生时处于风险集中的所有样本
        loss = -torch.sum(log_risks_observed - log_sum_exp_observed)

        # 为了稳定性，返回每个事件的平均损失
        num_observed_events = torch.sum(event_status)
        return loss / (num_observed_events + 1e-8) # 加一个很小的数防止除以零

# =============================================================================
# 2. 生存分析评估指标
# =============================================================================

def calculate_c_index(log_risks, event_times, event_status):
    """
    计算一致性指数 (C-index)。
    C-index 衡量了模型预测的风险排序与真实事件发生时间排序的一致性。
    范围是 0.5 (随机) 到 1.0 (完美)。
    
    参数 (均为 NumPy 数组):
    - log_risks (np.array): 模型的对数风险预测。
    - event_times (np.array): 事件/删失时间。
    - event_status (np.array): 事件/删失状态。
    """
    # lifelines 需要的是风险值，log_risk越高，风险越高，这与lifelines的期望一致
    return concordance_index(event_times, -log_risks, event_status)

def calculate_time_dependent_auc(log_risks, event_times, event_status, time_point):
    """
    计算特定时间点的时间依赖性AUC (AUC-t)。
    (已更新以使用 scikit-learn)
    """
    y_true = []
    y_pred_scores = [] # 变量名改为 y_pred_scores 更清晰

    for i in range(len(event_times)):
        # 情况1: 在 time_point 之前发生事件 -> 正样本 (event=1)
        if event_status[i] == 1 and event_times[i] <= time_point:
            y_true.append(1)
            y_pred_scores.append(log_risks[i]) # 模型输出的log_risk越高，风险越高
        # 情况2: 存活超过 time_point -> 负样本 (event=0)
        elif event_times[i] > time_point:
            y_true.append(0)
            y_pred_scores.append(log_risks[i])
        # 情况3: 在 time_point 之前被删失 -> 排除此样本
        else:
            continue

    if len(np.unique(y_true)) < 2:
        return np.nan 
        
    # 直接调用 scikit-learn 的 roc_auc_score 函数
    return roc_auc_score(np.array(y_true), np.array(y_pred_scores))

class EarlyStopping:
    """
    如果验证AUC在给定的patience内没有改善，则提前停止训练。
    """
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = -float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_auc, model, optimizer, epoch):
        score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_auc, model, optimizer, epoch)
        # 这里之前有一个小bug，应该是 score <= self.best_score
        # 我已经修复为 score < self.best_score + self.delta
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_auc, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_auc, model, optimizer, epoch):
        '''当验证AUC提升时，准备数据并调用全局保存函数。'''
        if self.verbose:
            # 只有当分数真正增加时才打印，val_auc_max 记录了上一个最佳分数
            if val_auc > self.val_auc_max:
                self.trace_func(f'Validation AUC increased ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
        
        # 将可能的NumPy类型转换为Python原生类型，以兼容 weights_only=True
        checkpoint_dict = {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_auc': float(val_auc), # 强制转换为 float
        }

        # 调用全局的 save_checkpoint 函数来执行实际的保存操作
        save_checkpoint(state=checkpoint_dict, is_best=True, filename=self.path)
        
        # 更新记录的最佳分数
        self.val_auc_max = val_auc

# ==============================================================================
# 全局辅助函数 (不属于任何类)
# ==============================================================================

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    """
    实际执行保存操作的函数。
    如果 is_best 为 True，会额外保存一个名为 'best_model.pth' 的副本。
    
    Args:
        state (dict): 要保存的检查点字典。
        is_best (bool): 如果为 True, 保存一个副本。
        filename (str): 检查点文件的完整路径。
    """
    # 保存当前的检查点
    torch.save(state, filename)
    
    # 如果这是当前最好的模型，则创建一个副本
    if is_best:
        directory = os.path.dirname(filename)
        # best_model.pth 将总是指向当前fold中最好的那个模型
        best_filename = os.path.join(directory, 'best_model.pth')
        shutil.copyfile(filename, best_filename)

def setup_optimizers_schedulers(model, is_staged, stage='frozen'):
    """
    根据训练策略和阶段设置优化器和学习率调度器。
    """
    import config
    
    params_to_update = []
    
    if is_staged and 'ct' in config.MODALITY_NAMES:
        print(f"Setting up optimizer for STAGED training, stage: '{stage}'")
        if stage == 'frozen':
            for name, param in model.named_parameters():
                if not name.startswith('feature_extractors.ct'):
                    param.requires_grad = True
                    params_to_update.append(param)
                else:
                    param.requires_grad = False
            print("Optimizer will train: Fusion Module, Classifier, and Tabular Extractors.")
            current_lr = config.LEARNING_RATE
        
        elif stage == 'finetune':
            ct_backbone_params = []
            other_params = []
            for name, param in model.named_parameters():
                param.requires_grad = True
                if name.startswith('feature_extractors.ct'):
                    ct_backbone_params.append(param)
                else:
                    other_params.append(param)
            
            params_to_update = [
                {'params': other_params},
                {'params': ct_backbone_params, 'lr': config.LEARNING_RATE * config.FINETUNE_LR_RATIO}
            ]
            print(f"Optimizer will fine-tune all layers. CT backbone LR is scaled by {config.FINETUNE_LR_RATIO}.")
            current_lr = config.LEARNING_RATE
            
    else:
        print("Setting up optimizer for DIRECT training (all layers trainable from start).")
        for param in model.parameters():
             param.requires_grad = True
             params_to_update.append(param)
        current_lr = config.LEARNING_RATE

    optimizer = torch.optim.AdamW(params_to_update, lr=current_lr, weight_decay=config.WEIGHT_DECAY)
    
    # 使用 ReduceLROnPlateau 调度器，当指标不再提升时降低学习率
    # 这对于超参数不确定的情况更鲁棒
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.DIRECT_TRAINING_EPOCHS, eta_min=1e-6)
    
    return optimizer, scheduler

def forward_pass_for_training(model, dataloader, criterions, device):
    """一个只进行前向传播和损失计算的训练步骤"""
    model.train()
    total_loss = 0
    # ... (与train_one_epoch前半部分相同的逻辑) ...
    for batch in dataloader:
        # ... (获取标签) ...
        outputs = model(batch)
        loss_efficacy = criterions['efficacy'](...)
        loss_toxicity = criterions['toxicity'](...)
        loss_survival = criterions['survival'](...)
        
        # 将损失合并
        batch_loss = loss_efficacy + loss_toxicity + loss_survival
        total_loss += batch_loss.item()
        
        # 只返回损失，不进行backward
        yield batch_loss

    # 此处不需要返回avg_loss，因为我们在主循环中处理
import pandas as pd
import datetime

def log_fold_results(model_type, seed, fold, best_metrics, config_snapshot):
    """
    将单次折叠的最终结果和实验配置记录到全局CSV文件中。

    Args:
        model_type (str): 'heavy' 或 'light'。
        seed (int): 当前重复的随机种子。
        fold (int): 当前的折叠编号。
        best_metrics (dict): 从validate函数得到的最佳指标字典。
        config_snapshot (dict): 包含关键config参数的字典。
    """
    # 定义日志文件的路径
    log_file_path = os.path.join(config.BASE_DIR, 'outputs', 'full_experiment_results.csv')
    
    # 准备要写入的数据
    log_data = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_type': model_type,
        'seed': seed,
        'fold': fold,
    }
    
    # 合并配置快照和最佳指标
    log_data.update(config_snapshot)
    log_data.update(best_metrics)
    
    # 转换为DataFrame
    df_new_log = pd.DataFrame([log_data])
    
    # 检查文件是否存在，以决定是否写入表头
    if not os.path.exists(log_file_path):
        df_new_log.to_csv(log_file_path, index=False)
    else:
        df_new_log.to_csv(log_file_path, mode='a', header=False, index=False)
        
    print(f"\n[Logger] 已将 Seed {seed}, Fold {fold} ({model_type} model) 的结果记录到: {log_file_path}")    

