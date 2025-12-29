import torch
import sys
import os

# ==========================================
# 1. 动态生成 Config (Mock Config)
# 作用：原本代码需要读取 config.py，我们这里动态注入一个配置对象
# ==========================================
class MockConfig:
    def __init__(self):
        # 必须与你 PDF 代码里的参数对应
        self.MODALITY_NAMES = ['ct', 'clinical', 'blood'] # 假设用了这三个
        self.D_MODEL = 256
        self.TRANSFORMER_NHEAD = 4
        self.TRANSFORMER_NLAYERS = 2
        self.DROPOUT_RATE = 0.1
        self.FUSION_STRATEGY = 'transformer_missing_token'
        self.USE_MODALITY_EMBEDDING = True
        
        # 模态维度字典 (必须对应)
        self.TABULAR_FEATURE_DIMS = {
            'clinical': 10,
            'blood': 50
        }
        # CT 特征提取器配置
        self.CT_EXTRACTOR = 'swin_unetr' # 或者 'resnet'
        self.ACTIVE_MODALITIES = ['ct', 'clinical', 'blood']

# 将 mock config 注入系统，假装导入了 config
sys.modules['config'] = MockConfig()
import config # 现在 import config 就会拿到上面的 MockConfig

# ==========================================
# 2. 导入原版模型
# ==========================================
# 确保 model.py 和 modules.py 在当前目录下
try:
    from model import MultiModalModel
except ImportError:
    print("错误：请确保 'model.py' 和 'modules.py' 在当前目录下！")
    sys.exit(1)

# ==========================================
# 3. 运行 Demo
# ==========================================
def run_legacy_demo():
    print("="*50)
    print("   v1.0 LEGACY TRANSFORMER DEMO")
    print("="*50)
    
    # 初始化模型
    print("[1] Initializing MultiModalModel (from PDF code)...")
    # 注意：原本代码可能需要 modules.py 里的组件，确保它们都在
    try:
        model = MultiModalModel()
        model.eval()
        print("    -> Model loaded successfully.")
    except Exception as e:
        print(f"    -> Error loading model: {e}")
        print("    (提示：可能需要调整 MockConfig 以匹配你的 model.py)")
        return

    # 构造伪数据
    # 注意：v1.0 的 Light Mode 输入通常是提取好的特征
    # 假设 CT 已经被提取成了 vector [B, 256] 或者 feature map
    print("[2] Generating Mock Inputs...")
    B = 2
    batch_data = {
        # 假设这里是 Light Mode，CT 也是预提取特征
        # 如果是 Heavy Mode，这里要是 [B, 1, 96, 96, 96]
        'ct': torch.randn(B, 256), 
        'clinical': torch.randn(B, 10),
        'blood': torch.randn(B, 50),
        # 必须包含 sample_id 以防代码报错
        'sample_id': ['test_001', 'test_002']
    }
    
    # 前向传播
    print("[3] Running Forward Pass...")
    try:
        # 你的 model.py forward 可能只接受一个参数或者多个
        # 这里假设它接受一个字典
        # 如果 v1.0 是 heavy mode，可能需要 self.ct_extractor 处理
        # 这里我们模拟 Light Mode (直接输入特征)
        outputs = model(batch_data) # 或者是 model.forward(batch_data)
        
        print("\n[4] Output Verification:")
        # 打印输出结果（假设输出是一个字典）
        for key, val in outputs.items():
            if isinstance(val, torch.Tensor):
                print(f" - {key}: shape {val.shape}")
            else:
                print(f" - {key}: {val}")
                
        print("\n[Success] v1.0 Pipeline Verified.")
        
    except Exception as e:
        print(f"\n[Fail] Runtime Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_legacy_demo()