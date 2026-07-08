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

# 论文 Section VI 使用 DeiT-tiny 作为 ViT backbone（12 层，patch 16×16）
VIT_MODEL = "deit_tiny_patch16_224"

# d：每个 token 的特征维度（DeiT-tiny embed_dim = 192）
D_MODEL = 192

# Imagenette 子集 10 类
NUM_CLASSES = 10

# 224×224 图像、patch 16 → 14×14 = 196 个 patch token
N_PATCHES = 196

# 语义序列长度（CLS + patch tokens；不含末尾 budget token）
N_TOKENS = N_PATCHES + 1

# 论文 JSCC 分割点 splitting_point=6：
#   前 6 个 Transformer block 在边缘端（编码器侧），之后插入语义 JSCC 信道
ENCODER_SPLIT = 6

# block 1..5 替换为自适应 block（含 token 选择模块 S_l）
# block 0 保持标准 ViT block，不做 token 丢弃
ADAPTIVE_LAYERS = 5

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
    """自动探测 Imagenette 根目录；可通过环境变量 IMAGENETTE_ROOT 覆盖。"""
    candidates = [
        os.environ.get("IMAGENETTE_ROOT"),
        PROJECT_ROOT / "imagenette2",
        Path("C:/Users/hp/Downloads/imagenette2"),
        Path.home() / "Downloads" / "imagenette2",
    ]
    for p in candidates:
        if p and Path(p).joinpath("train").is_dir():
            return Path(p)
    return Path("C:/Users/hp/Downloads/imagenette2")


IMAGENETTE_ROOT = _default_imagenette_root()

# ---------------------------------------------------------------------------
# Section VI-A — 训练超参数
# ---------------------------------------------------------------------------

MICRO_BATCH_SIZE = 64
ACCUMULATION_STEPS = 4          # 等效 batch size = 64 × 4 = 256
LEARNING_RATE = 1e-3            # 官方 pretraining: Adam lr=0.001
WARMUP_EPOCHS = 10

# 两阶段训练（Section VI-B）:
#   Stage 1: 语义 ViT + 自适应 token 选择（无 JSCC）
#   Stage 2: 冻结边缘 encoder + 服务器 block，仅训练 JSCC（DJSCC_FREEZE_SERVER=True）
SEMANTIC_EPOCHS = 150

# 渐进式训练：早期保留更多 token，避免 mask 过稀疏导致梯度消失
PROGRESSIVE_EPOCHS = 50
MIN_KEEP_RATIO_START = 1.0
MIN_KEEP_RATIO_END = 0.0

# 预算约束损失中的 margin ε（论文式(9) 中 |m̄ - α| - ε 的 ε）
EPSILON_START = 0.02
EPSILON_END = 0.02

# 式(9) 损失权重 — 加强预算约束，使 m̄_l ≈ α（此前 α=0.05 时实际保留 ~22% patch）
LAMBDA_S = 5.0
LAMBDA_R = 2.0
MARGIN = 0.01

# Stage 2: 150 epoch
DJSCC_EPOCHS = 150

# Stage 2 仅训练 JSCC 编解码器，冻结服务器端 block（压高 ρ 准确率）
DJSCC_FREEZE_SERVER = True

# Stage 2 低 ρ 预算采样：55% 来自 α∈[0.05, 0.2]，强化极低压缩端
ALPHA_LOW_MIN = 0.05
ALPHA_LOW_MAX = 0.2
ALPHA_LOW_BIAS = 0.55

# Stage 2 SNR 采样：70% 来自低 SNR 区间
SNR_TRAIN_MIN = -10.0
SNR_TRAIN_MAX = 10.0
SNR_LOW_BIAS = 0.7

# Stage 2 起用 hard-mask STE 的 epoch
DJSCC_STE_START_EPOCH = 20

# ---------------------------------------------------------------------------
#  checkpoint 与结果路径
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"
SEMANTIC_MODEL_PATH = CHECKPOINT_DIR / "semantic_pretrained.pth"
FINAL_MODEL_PATH = CHECKPOINT_DIR / "final_model.pth"
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
