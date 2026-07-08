"""
模型工厂（论文 Section VI 骨干网络配置）
======================================

论文使用 DeiT-tiny (deit_tiny_patch16_224):
  - 12 层 Transformer, embed_dim d=192
  - patch size 16×16, 输入 224×224
  - 前 6 层为边缘端编码器，后 6 层为服务器端解码器

支持从本地权重加载（环境变量 DEIT_WEIGHTS 或 weights/ 目录）。
"""
import os
from pathlib import Path

import timm
import torch

from config import (
    ADAPTIVE_LAYERS,
    BEST_MODEL_PATH,
    D_MODEL,
    ENCODER_SPLIT,
    NUM_CLASSES,
    R_VALUES,
    SEMANTIC_MODEL_PATH,
    VIT_MODEL,
)
from djscc_model import FullDJSCCModel


def _local_weights_path():
    """查找本地 DeiT 预训练权重（避免 HuggingFace 下载失败）。"""
    candidates = [
        Path(os.environ.get("DEIT_WEIGHTS", "")),
        Path(__file__).resolve().parent.parent / "weights" / f"{VIT_MODEL}.pth",
        Path.home() / ".cache" / "timm" / "models" / f"{VIT_MODEL}.pth",
    ]
    for p in candidates:
        if p and p.is_file():
            return p
    return None


def create_vit_backbone(pretrained=True):
    """
    创建 DeiT-tiny 骨干并替换为 Imagenette 10 类分类头。
    """
    local = _local_weights_path()
    if local:
        vit = timm.create_model(VIT_MODEL, pretrained=False, num_classes=NUM_CLASSES)
        state = torch.load(local, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        vit.load_state_dict(state, strict=False)
    else:
        vit = timm.create_model(VIT_MODEL, pretrained=pretrained, num_classes=NUM_CLASSES)

    torch.nn.init.trunc_normal_(vit.head.weight, std=0.02)
    torch.nn.init.zeros_(vit.head.bias)
    return vit


def build_model(pretrained=True):
    """构建完整 FullDJSCCModel（Section III–IV 全流程）。"""
    vit = create_vit_backbone(pretrained=pretrained)
    return FullDJSCCModel(
        vit,
        s=ADAPTIVE_LAYERS,
        encoder_split=ENCODER_SPLIT,
        r_values=R_VALUES,
        d_model=D_MODEL,
    )


def _valid_checkpoint(path):
    """检查 checkpoint 是否存在且非空。"""
    path = Path(path)
    if not path.is_file():
        return False
    if path.stat().st_size < 1024:
        return False
    return True


def load_checkpoint_state(path, device="cuda"):
    path = Path(path)
    if not _valid_checkpoint(path):
        raise FileNotFoundError(
            f"Checkpoint 无效或为空: {path}\n"
            f"  文件大小: {path.stat().st_size if path.exists() else 0} bytes\n"
            "  请先运行: python train.py --stage semantic"
        )
    return torch.load(path, map_location=device, weights_only=False)


def save_checkpoint(state_dict, path):
    """原子写入，避免中断导致 0 字节损坏文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, tmp)
    tmp.replace(path)


def load_trained_model(checkpoint_path=None, device="cuda"):
    """加载 Stage 2 训练完成的最优 JSCC 模型。"""
    path = Path(checkpoint_path or BEST_MODEL_PATH)
    model = build_model(pretrained=False)
    if _valid_checkpoint(path):
        state = load_checkpoint_state(path, device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {path}")
    else:
        print(f"Warning: checkpoint not found or invalid: {path}")
    model.to(device)
    model.eval()
    return model


def load_semantic_checkpoint(model, path=None, device="cuda"):
    """Stage 2 开始前加载 Stage 1 语义预训练权重。"""
    ckpt = Path(path or SEMANTIC_MODEL_PATH)
    state = load_checkpoint_state(ckpt, device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded semantic checkpoint: {ckpt} ({ckpt.stat().st_size // 1024} KB)")
    return model
