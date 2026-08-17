"""
论文复现统一配置
================

对应论文:
  Devoto et al., "Adaptive Semantic Token Communication for
  Transformer-Based Edge Inference"

本文件集中定义 Section VI 实验设置中的模型结构、训练超参与评估网格。

符号说明（与论文一致）:
  α ∈ [0,1]  — 语义 token 预算（保留比例），控制边缘端计算/传输开销
  r           — JSCC 编码器输出维度相对 token 维度的比例（output_size）
  ρ           — 端到端压缩比 = 实际传输符号数 / 输入像素数
  d           — Transformer token 嵌入维度（DeiT-tiny: d=192）
  SNR         — 加性高斯白噪声信道的信噪比（dB）
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Section III / VI — 模型与系统参数
# ---------------------------------------------------------------------------

# 骨干选择: "mambavision" | "vit"
# 也可通过环境变量 DJSCC_BACKBONE 覆盖
BACKBONE = os.environ.get("DJSCC_BACKBONE", "mambavision").lower()

# --- ViT (DeiT-tiny) ---
VIT_MODEL = "deit_tiny_patch16_224"
VIT_D_MODEL = 192
ENCODER_SPLIT = 6
ADAPTIVE_LAYERS = 5

# --- MambaVision-T（流程对齐论文；切分按层次结构，不照搬 ViT 的 6 层）---
# Stage2 共 8 block、14×14=196 token:
#   边缘: Stage0/1 + 全部 Stage2 blocks + 渐进式 S_l
#   信道: JSCC（在 Stage2↓ 之前，分辨率仍为 14×14）
#   服务器: Stage2 downsample + Stage3 + head
# MAMBA_ENCODER_SPLIT=None 表示 Stage2 全部留在边缘
MAMBA_MODEL = "mamba_vision_T"
MAMBA_D_MODEL = 320
MAMBA_WEIGHTS_NAME = "mambavision_tiny_1k.pth.tar"
MAMBA_ENCODER_SPLIT = None  # None → 使用 Stage2 全部 block
MAMBA_ADAPTIVE_LAYERS = 5   # 与论文 s=5 一致：block1..5 前插入选择器

# 当前骨干生效的 d / token 数
if BACKBONE == "mambavision":
    D_MODEL = MAMBA_D_MODEL
else:
    D_MODEL = VIT_D_MODEL
    BACKBONE = "vit"

# Imagenette 子集 10 类
NUM_CLASSES = 10

# 224×224 图像 → 14×14 = 196 个空间 token（ViT patch / Mamba Stage2）
N_PATCHES = 196

# 语义序列长度（ViT: CLS+patches；Mamba: patches；均不含 budget）
N_TOKENS = N_PATCHES + (1 if BACKBONE == "vit" else 0)

# p：输入图像像素总数，用于计算压缩比 ρ = q / p（Section III）
TOTAL_PIXELS = 224 * 224 * 3

# JSCC 编码器输出维度比例 r（论文称 output_size / compression）
# 对应 o_r = r · d，即每个 token 映射为 r·d 个复数符号
R_VALUES = [0.5, 0.25, 0.1]
R_VALUES_TRAIN = [0.5, 0.25, 0.1]

# ---------------------------------------------------------------------------
# Section VI — 数据集（Imagenette 224×224）
# ---------------------------------------------------------------------------

def _default_imagenette_root():
    """自动探测 Imagenette 根目录；可通过环境变量 IMAGENETTE_ROOT 覆盖。

    同时兼容 Windows 路径与 WSL 下的 /mnt/c/... 路径。
    """
    home = Path.home()
    candidates = [
        os.environ.get("IMAGENETTE_ROOT"),
        PROJECT_ROOT / "imagenette2",
        Path("C:/Users/hp/Downloads/imagenette2"),
        Path.home() / "Downloads" / "imagenette2",
        # WSL 访问 Windows 盘
        Path("/mnt/c/Users/hp/Downloads/imagenette2"),
        Path("/mnt/c/Users/hp/Documents/imagenette2"),
        home / "data" / "imagenette2",
    ]
    # 通用：若在 WSL，把 C:/Users/<name>/... 映射到 /mnt/c/Users/<name>/...
    win_downloads = Path("C:/Users/hp/Downloads/imagenette2")
    if not win_downloads.joinpath("train").is_dir():
        wsl_mapped = Path("/mnt/c/Users/hp/Downloads/imagenette2")
        if wsl_mapped not in candidates:
            candidates.append(wsl_mapped)

    for p in candidates:
        if p and Path(p).joinpath("train").is_dir():
            return Path(p)
    return Path("/mnt/c/Users/hp/Downloads/imagenette2")


IMAGENETTE_ROOT = _default_imagenette_root()

# ---------------------------------------------------------------------------
# Section VI-A — 训练超参数
# ---------------------------------------------------------------------------

# MambaVision-T 显存更大，默认更小 micro-batch；可用环境变量覆盖
MICRO_BATCH_SIZE = int(os.environ.get(
    "DJSCC_MICRO_BATCH",
    "16" if BACKBONE == "mambavision" else "64",
))
ACCUMULATION_STEPS = int(os.environ.get(
    "DJSCC_ACCUM_STEPS",
    "16" if BACKBONE == "mambavision" else "4",
))  # 默认仍接近等效 batch 256
LEARNING_RATE = 1e-3            # 官方 pretraining: Adam lr=0.001
WARMUP_EPOCHS = 10

# 两阶段训练（对齐官方 Section VI / main.py）:
#   Stage 1: 语义骨干 + 自适应 token 选择（无 JSCC），150 epoch
#   Stage 2: freeze_model=Yes → 训 JSCC + blocks_after，冻边缘与 head；100 epoch
SEMANTIC_EPOCHS = 150

# 官方无 progressive min_keep；保持关闭以对齐论文
PROGRESSIVE_EPOCHS = 0
MIN_KEEP_RATIO_START = 0.0
MIN_KEEP_RATIO_END = 0.0

EPSILON_START = 0.02
EPSILON_END = 0.02

# 官方 hydra: output_flops_w=2, inner_flops_w=1, margin=0.01
LAMBDA_S = 2.0
LAMBDA_R = 1.0
MARGIN = 0.01

# 官方 jscc proposal: epochs=100
DJSCC_EPOCHS = 100

# False = 论文默认：训 JSCC + 服务器 blocks_after（冻 head/norm）
# True  = 消融：仅训 JSCC
DJSCC_FREEZE_SERVER = False

# Stage 2 α：官方 SemanticVit 训练时 alpha∈(1e-3, 1-1e-3) 近均匀
ALPHA_LOW_MIN = 0.05
ALPHA_LOW_MAX = 0.2
ALPHA_LOW_BIAS = 0.0

# Stage 2 SNR：官方 GaussianNoiseChannel 在 [-10, 10] 上均匀采样
SNR_TRAIN_MIN = -10.0
SNR_TRAIN_MAX = 10.0
SNR_LOW_BIAS = 0.0

# Stage2 后期对出口 mask 使用 STE top-k，缩小 soft 训练与硬预算推理差距
DJSCC_STE_START_EPOCH = 20

# ---------------------------------------------------------------------------
#  checkpoint 与结果路径
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / BACKBONE
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
SEMANTIC_MODEL_PATH = CHECKPOINT_DIR / "semantic_pretrained.pth"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "final_model.pth"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper_figures"

# ---------------------------------------------------------------------------
# Section VI — 评估网格（Fig.6 / Fig.7）
# ---------------------------------------------------------------------------

# 扫描不同 α 以得到 accuracy–ρ Pareto 曲线
ALPHA_VALUES_EVAL = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Fig.6: 低 SNR；Fig.7: 高 SNR
SNR_LOW = -10.0
SNR_HIGH = 10.0
SNR_EVAL_LIST = [-10, -5, 0, 5, 10]
