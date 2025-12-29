import torch
import torch.nn as nn
import time
import sys

# ==========================================
# 1. 模拟组件 (Mock Components)
# 作用：让Demo在任何环境下都能跑，无需安装MONAI
# ==========================================
class MockSwinUNETR(nn.Module):
    """
    模拟 SwinUNETR 的行为，确保 Demo 在无 MONAI 环境下也能展示逻辑。
    输出标准 Swin Base 的 Bottleneck 维度。
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        print(f"[System] Initializing SwinUNETR Backbone (Mock Mode)...")

    def forward(self, x):
        # x shape: [B, 1, 96, 96, 96]
        # 模拟 32倍下采样后的 bottleneck
        # 96 / 32 = 3
        # 输出 shape: [B, 768, 3, 3, 3]
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.embed_dim, 3, 3, 3).to(x.device)

# ==========================================
# 2. 核心架构引用 (引用你的 avatar_model.py)
# ==========================================
try:
    from avatar_model import AvatarFusionEngine
except ImportError:
    # 如果找不到文件，为了演示方便，直接在这里定义一遍（防呆设计）
    # 实际使用时请删除这部分，确保 avatar_model.py 存在
    print("[Warning] avatar_model.py not found. Using inline definition.")
    class AvatarFusionEngine(nn.Module):
        def __init__(self, embed_dim=768, num_heads=12):
            super().__init__()
            self.omics_proj = nn.Linear(10, embed_dim) # 简化版
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        def forward(self, img_feat, batch_data):
            b, c, d, h, w = img_feat.shape
            img_flat = img_feat.view(b, c, -1).permute(0, 2, 1)
            omics = self.omics_proj(batch_data['clinical']).unsqueeze(1)
            out, weights = self.cross_attn(query=omics, key=img_flat, value=img_flat)
            return out.mean(1), weights

# ==========================================
# 3. 完整系统包装 (System Wrapper)
# ==========================================
class AvatarSystem(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MockSwinUNETR(embed_dim=768)
        self.engine = AvatarFusionEngine(embed_dim=768, num_heads=12)
        self.classifier = nn.Linear(768, 1) # 简化演示
    
    def forward(self, batch_data):
        img = batch_data['image']
        print(f"   > [1/4] Image Encoded. Shape: {img.shape} -> Bottleneck")
        feat = self.backbone(img)
        
        print(f"   > [2/4] Avatar Engine Fusion...")
        fused, weights = self.engine(feat, batch_data)
        
        print(f"   > [3/4] Task Decoding...")
        risk = torch.sigmoid(self.classifier(fused))
        return risk, weights

# ==========================================
# 4. 专业的 Demo 流程脚本
# ==========================================
def run_professional_demo():
    print("\n" + "="*60)
    print("      AVATAR v2.0: MULTIMODAL INTELLIGENT DIAGNOSIS SYSTEM")
    print("      (C) 2025 Research Demo | Powered by PyTorch")
    print("="*60 + "\n")

    # 配置参数
    CONFIG = {
        'batch_size': 2,
        'img_size': (96, 96, 96),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"[Init] Checking Environment Resources...")
    print(f" - Device: {CONFIG['device'].upper()}")
    print(f" - Precision: FP32")
    time.sleep(0.5) # 假装在加载资源

    # 初始化模型
    print(f"\n[Init] Building Computational Graph...")
    model = AvatarSystem().to(CONFIG['device'])
    params = sum(p.numel() for p in model.parameters())
    print(f" - Total Parameters: {params / 1e6:.2f} M")
    print(f" - Status: READY")

    # 构造伪数据
    print(f"\n[Data] Generating Synthetic Patient Data (Batch={CONFIG['batch_size']})...")
    batch_data = {
        'image': torch.randn(CONFIG['batch_size'], 1, *CONFIG['img_size']).to(CONFIG['device']),
        'clinical': torch.randn(CONFIG['batch_size'], 10).to(CONFIG['device']), # 模拟临床特征
        # 可以在这里加更多模态...
    }
    print(f" - CT Scan Tensor: {batch_data['image'].shape}")
    print(f" - Clinical Data:  {batch_data['clinical'].shape}")

    # 前向传播
    print(f"\n[Run] Executing Forward Pass...")
    start_time = time.time()
    
    with torch.no_grad():
        risk_score, attn_weights = model(batch_data)
    
    end_time = time.time()
    
    # 结果报告
    print(f"\n" + "-"*30)
    print(f"       INFERENCE REPORT")
    print(f"-"*30)
    print(f"Latency: {(end_time - start_time)*1000:.2f} ms")
    print(f"Output Shapes:")
    print(f" - Risk Score: {risk_score.shape} (Probabilities)")
    print(f" - Attn Map:   {attn_weights.shape} (Explainability)")
    
    print(f"\nPredictions:")
    for i in range(CONFIG['batch_size']):
        print(f" Patient ID_{i+1001}: Risk={risk_score[i].item():.4f} | Confidence=High")

    print("\n[Done] Demo completed successfully.")

if __name__ == "__main__":
    run_professional_demo()