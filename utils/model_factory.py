"""
模型工厂（ViT / MambaVision 可切换）
====================================

BACKBONE=vit         → DeiT-tiny + FullDJSCCModel
BACKBONE=mambavision → MambaVision-T + FullDJSCCMambaModel
"""
import os
from pathlib import Path

import torch
from torch import nn

from config import (
    ADAPTIVE_LAYERS,
    BACKBONE,
    BEST_MODEL_PATH,
    ENCODER_SPLIT,
    MAMBA_ADAPTIVE_LAYERS,
    MAMBA_D_MODEL,
    MAMBA_ENCODER_SPLIT,
    MAMBA_MODEL,
    MAMBA_WEIGHTS_NAME,
    NUM_CLASSES,
    R_VALUES,
    SEMANTIC_MODEL_PATH,
    VIT_D_MODEL,
    VIT_MODEL,
    WEIGHTS_DIR,
)


def _local_vit_weights_path():
    candidates = [
        Path(os.environ.get("DEIT_WEIGHTS", "")),
        WEIGHTS_DIR / f"{VIT_MODEL}.pth",
        Path.home() / ".cache" / "timm" / "models" / f"{VIT_MODEL}.pth",
    ]
    for p in candidates:
        if p and p.is_file():
            return p
    return None


def _local_mamba_weights_path():
    candidates = [
        Path(os.environ.get("MAMBA_WEIGHTS", "")),
        WEIGHTS_DIR / MAMBA_WEIGHTS_NAME,
        Path("/tmp") / "mamba_vision_T.pth.tar",
        Path("/tmp") / MAMBA_WEIGHTS_NAME,
    ]
    for p in candidates:
        if p and p.is_file():
            return p
    return None


def create_vit_backbone(pretrained=True):
    import timm

    local = _local_vit_weights_path()
    if local:
        vit = timm.create_model(VIT_MODEL, pretrained=False, num_classes=NUM_CLASSES)
        state = torch.load(local, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        vit.load_state_dict(state, strict=False)
    else:
        vit = timm.create_model(VIT_MODEL, pretrained=pretrained, num_classes=NUM_CLASSES)

    nn.init.trunc_normal_(vit.head.weight, std=0.02)
    nn.init.zeros_(vit.head.bias)
    return vit


def _extract_state_dict(checkpoint):
    """从常见 checkpoint 包装中取出 state_dict。"""
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model", "model_ema", "module"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    # 已是纯参数字典
    if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint
    return checkpoint


def create_mamba_backbone(pretrained=True):
    """
    创建 MambaVision-T，分类头改为 Imagenette 10 类。

    预训练权重优先本地 weights/，避免 WSL 无法访问 HuggingFace。
    使用 weights_only=False 加载（官方 ckpt 含 argparse.Namespace）。
    """
    try:
        from mambavision import create_model
    except ImportError as e:
        raise ImportError(
            "未安装 mambavision。请在 WSL mamba_djscc 环境中安装:\n"
            "  pip install mamba-ssm --no-build-isolation --no-deps\n"
            "  pip install mambavision --no-deps\n"
            "  pip install timm==1.0.15 transformers==4.50.0"
        ) from e

    local = _local_mamba_weights_path()
    model = create_model(MAMBA_MODEL, pretrained=False)

    if pretrained and local is not None:
        print(f"Loading MambaVision weights: {local}")
        checkpoint = torch.load(str(local), map_location="cpu", weights_only=False)
        state = _extract_state_dict(checkpoint)
        # 去掉可能的 module. 前缀
        cleaned = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned[nk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"  loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    elif pretrained and local is None:
        print(
            "Warning: 未找到本地 MambaVision 权重，使用随机初始化。\n"
            f"  请下载到: {WEIGHTS_DIR / MAMBA_WEIGHTS_NAME}\n"
            "  https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar\n"
            "  或镜像: https://hf-mirror.com/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar"
        )

    in_features = model.head.in_features
    model.head = nn.Linear(in_features, NUM_CLASSES)
    nn.init.trunc_normal_(model.head.weight, std=0.02)
    nn.init.zeros_(model.head.bias)
    return model


def build_model(pretrained=True, backbone=None):
    """构建完整 DJSCC 模型（按 BACKBONE 选择 ViT 或 MambaVision）。"""
    bb = (backbone or BACKBONE).lower()
    if bb == "mambavision":
        from djscc_mamba_model import FullDJSCCMambaModel

        mamba = create_mamba_backbone(pretrained=pretrained)
        model = FullDJSCCMambaModel(
            mamba,
            s=MAMBA_ADAPTIVE_LAYERS,
            encoder_split=MAMBA_ENCODER_SPLIT,
            r_values=R_VALUES,
            d_model=MAMBA_D_MODEL,
        )
        print(
            f"Built FullDJSCCMambaModel (d={MAMBA_D_MODEL}, "
            f"stage2_edge_blocks={model.encoder_split}, "
            f"adaptive_layers={MAMBA_ADAPTIVE_LAYERS})"
        )
        return model

    from djscc_model import FullDJSCCModel

    vit = create_vit_backbone(pretrained=pretrained)
    model = FullDJSCCModel(
        vit,
        s=ADAPTIVE_LAYERS,
        encoder_split=ENCODER_SPLIT,
        r_values=R_VALUES,
        d_model=VIT_D_MODEL,
    )
    print(f"Built FullDJSCCModel / ViT (d={VIT_D_MODEL})")
    return model


def _valid_checkpoint(path):
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, tmp)
    tmp.replace(path)


def load_trained_model(checkpoint_path=None, device="cuda", backbone=None):
    path = Path(checkpoint_path or BEST_MODEL_PATH)
    model = build_model(pretrained=False, backbone=backbone)
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
    ckpt = Path(path or SEMANTIC_MODEL_PATH)
    state = load_checkpoint_state(ckpt, device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded semantic checkpoint: {ckpt} ({ckpt.stat().st_size // 1024} KB)")
    return model
