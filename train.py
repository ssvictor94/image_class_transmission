"""
论文 Section VI 两阶段训练脚本（对齐官方 main.py）

Stage 2（freeze_model=Yes）:
  - 训练 JSCC + blocks_after；冻结边缘 encoder 与 head/norm
  - α 近均匀 ∈ (1e-3, 1-1e-3)；SNR 均匀 ∈ [-10, 10]
  - 训练 soft mask；推理阈值二值化

用法:
  python train.py --stage semantic
  python train.py --stage djscc
"""
import argparse
import os
import random
from datetime import datetime

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from config import (
    ACCUMULATION_STEPS,
    ALPHA_LOW_BIAS,
    ALPHA_LOW_MAX,
    ALPHA_LOW_MIN,
    BACKBONE,
    DJSCC_EPOCHS,
    DJSCC_FREEZE_SERVER,
    DJSCC_STE_START_EPOCH,
    LEARNING_RATE,
    MICRO_BATCH_SIZE,
    MIN_KEEP_RATIO_END,
    MIN_KEEP_RATIO_START,
    PROGRESSIVE_EPOCHS,
    PROJECT_ROOT,
    R_VALUES_TRAIN,
    SEMANTIC_EPOCHS,
    SNR_LOW_BIAS,
    SNR_TRAIN_MAX,
    SNR_TRAIN_MIN,
    WARMUP_EPOCHS,
)
from utils.data import get_dataloaders
from utils.evaluation import evaluate_model, evaluate_semantic
from utils.loss import adaptive_token_loss
from utils.model_factory import build_model, load_semantic_checkpoint, save_checkpoint


def progressive_min_keep(epoch, total_prog=PROGRESSIVE_EPOCHS):
    if epoch <= total_prog:
        t = (epoch - 1) / max(total_prog - 1, 1)
        return MIN_KEEP_RATIO_START + t * (MIN_KEEP_RATIO_END - MIN_KEEP_RATIO_START)
    return MIN_KEEP_RATIO_END


def sample_alpha(batch_size, device, low_bias=ALPHA_LOW_BIAS):
    """官方训练 alpha∈(1e-3, 1-1e-3)；low_bias>0 时为可选消融偏置。"""
    if low_bias <= 0:
        return torch.empty(batch_size, device=device).uniform_(1e-3, 1.0 - 1e-3)
    n_low = int(batch_size * low_bias)
    n_high = batch_size - n_low
    low = torch.empty(n_low, device=device).uniform_(ALPHA_LOW_MIN, ALPHA_LOW_MAX)
    high = torch.empty(n_high, device=device).uniform_(ALPHA_LOW_MAX, 1.0)
    alpha = torch.cat([low, high], dim=0)
    perm = torch.randperm(batch_size, device=device)
    return alpha[perm]


def sample_snr(batch_size, device):
    """官方：SNR 在 [SNR_TRAIN_MIN, SNR_TRAIN_MAX] 均匀采样。"""
    if SNR_LOW_BIAS <= 0:
        return torch.empty(batch_size, device=device).uniform_(
            SNR_TRAIN_MIN, SNR_TRAIN_MAX,
        )
    n_low = int(batch_size * SNR_LOW_BIAS)
    n_high = batch_size - n_low
    low = torch.empty(n_low, device=device).uniform_(SNR_TRAIN_MIN, 0.0)
    high = torch.empty(n_high, device=device).uniform_(0.0, SNR_TRAIN_MAX)
    snr = torch.cat([low, high], dim=0)
    perm = torch.randperm(batch_size, device=device)
    return snr[perm]


def semantic_step(model, images, labels, alpha, epoch):
    min_keep = progressive_min_keep(epoch)
    logits, layer_masks, _ = model.forward_semantic(
        images, alpha, min_keep_ratio=min_keep, discretize=False,
    )
    ce = F.cross_entropy(logits, labels)
    bl = adaptive_token_loss(layer_masks, alpha)
    return ce + bl


def djscc_step(model, images, labels, alpha, r, snr_db, epoch):
    # 官方：训练全程 soft mask（无 STE top-k）
    use_ste = epoch >= DJSCC_STE_START_EPOCH
    logits, _, _ = model.forward_full(
        images, alpha, r, snr_db,
        min_keep_ratio=0.0,
        discretize=False,
        use_ste=use_ste,
    )
    return F.cross_entropy(logits, labels)


def train_epoch(model, loader, optimizer, epoch, stage, device, accum_steps=ACCUMULATION_STEPS):
    if stage == "djscc":
        model.set_djscc_train_mode()
    else:
        model.train()

    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        if stage == "semantic":
            alpha = torch.empty(images.size(0), device=device).uniform_(0.0, 1.0)
            loss = semantic_step(model, images, labels, alpha, epoch) / accum_steps
        else:
            alpha = sample_alpha(images.size(0), device)
            r = random.choice(R_VALUES_TRAIN)
            snr_db = sample_snr(images.size(0), device)
            loss = djscc_step(model, images, labels, alpha, r, snr_db, epoch) / accum_steps

        loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        n_batches += 1
        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}, loss={loss.item() * accum_steps:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validation_score(model, val_loader, device):
    """选模侧重低 ρ 多 SNR 点（与训练采样一致）。"""
    configs = [
        (0.05, 0.1, -10.0, 0.25),
        (0.05, 0.1, 10.0, 0.25),
        (0.1, 0.25, -10.0, 0.25),
        (0.5, 0.25, 10.0, 0.25),
    ]
    score = 0.0
    parts = []
    for alpha, r, snr, w in configs:
        acc, _ = evaluate_model(
            model, val_loader, {"alpha": alpha, "r": r}, snr_db=snr, device=device,
        )
        score += w * acc
        parts.append(f"a={alpha} r={r} snr={snr:.0f}:{acc:.3f}")
    print(f"  val {' | '.join(parts)} | score={score:.4f}")
    return score


def main():
    parser = argparse.ArgumentParser(description="论文 Section VI 两阶段训练")
    parser.add_argument("--stage", choices=["semantic", "djscc"], default="semantic")
    parser.add_argument(
        "--backbone", choices=["mambavision", "vit"], default=None,
        help="默认读 config.BACKBONE / 环境变量 DJSCC_BACKBONE",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    backbone = (args.backbone or BACKBONE).lower()
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / backbone
    semantic_path = checkpoint_dir / "semantic_pretrained.pth"
    best_path = checkpoint_dir / "best_model.pth"
    final_path = checkpoint_dir / "final_model.pth"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if backbone == "mambavision":
        micro_bs = int(os.environ.get("DJSCC_MICRO_BATCH", "16"))
        accum = int(os.environ.get("DJSCC_ACCUM_STEPS", "16"))
    else:
        micro_bs = int(os.environ.get("DJSCC_MICRO_BATCH", str(MICRO_BATCH_SIZE)))
        accum = int(os.environ.get("DJSCC_ACCUM_STEPS", str(ACCUMULATION_STEPS)))

    train_loader, val_loader = get_dataloaders(batch_size=micro_bs)

    print(f"骨干: {backbone}")
    print(f"训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")
    print(f"Stage: {args.stage}, micro_bs={micro_bs}, accum={accum}, 等效 batch={micro_bs * accum}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    model = build_model(pretrained=True, backbone=backbone).to(args.device)

    if args.stage == "djscc":
        from utils.model_factory import _valid_checkpoint
        if not _valid_checkpoint(semantic_path):
            raise SystemExit(
                f"\n错误: Stage 1 权重无效: {semantic_path}\n"
                "  该文件为空或不存在，请先完成语义预训练:\n"
                f"  python train.py --stage semantic --backbone {backbone}\n"
            )
        load_semantic_checkpoint(model, path=semantic_path, device=args.device)
        if DJSCC_FREEZE_SERVER:
            model.freeze_jscc_only()
            print("Stage2: 消融模式 — 仅训练 JSCC")
        else:
            model.freeze_djscc_with_server()
            print("Stage2: 论文模式 — 训 JSCC + blocks_after，冻边缘与 head/norm")
        if ALPHA_LOW_BIAS > 0:
            print(f"Stage2: α 低预算偏置 {ALPHA_LOW_BIAS:.0%} ∈ [{ALPHA_LOW_MIN}, {ALPHA_LOW_MAX}]")
        else:
            print("Stage2: α ~ Uniform(1e-3, 1-1e-3)（对齐官方）")
        print(f"Stage2: SNR ~ Uniform([{SNR_TRAIN_MIN}, {SNR_TRAIN_MAX}])")
        total_epochs = DJSCC_EPOCHS
        save_path = best_path
    else:
        model.unfreeze_semantic()
        total_epochs = SEMANTIC_EPOCHS
        save_path = semantic_path

    if args.resume and save_path.is_file():
        from utils.model_factory import _valid_checkpoint, load_checkpoint_state
        if _valid_checkpoint(save_path):
            model.load_state_dict(load_checkpoint_state(save_path, args.device), strict=False)
            print(f"Resumed from {save_path}")
        else:
            print(f"Warning: resume 跳过无效 checkpoint: {save_path}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - WARMUP_EPOCHS, 1), eta_min=1e-6)

    best_metric = -1.0
    log_path = checkpoint_dir / f"train_{args.stage}.log"

    for epoch in range(1, total_epochs + 1):
        print(f"\n=== Epoch {epoch}/{total_epochs} ({args.stage}) ===")
        avg_loss = train_epoch(
            model, train_loader, optimizer, epoch, args.stage, args.device,
            accum_steps=accum,
        )

        if epoch <= WARMUP_EPOCHS:
            warmup.step()
        else:
            cosine.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} avg_loss={avg_loss:.4f}, lr={lr:.6f}")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} epoch={epoch} loss={avg_loss:.4f} lr={lr}\n")

        if epoch % 10 == 0 or epoch == total_epochs:
            if args.stage == "semantic":
                metric = evaluate_semantic(model, val_loader, alpha=0.5, device=args.device)
                print(f"  val acc={metric:.4f}")
            else:
                metric = validation_score(model, val_loader, args.device)

            if metric > best_metric:
                best_metric = metric
                save_checkpoint(model.state_dict(), save_path)
                print(f"  -> saved {save_path}")

    if args.stage == "djscc":
        save_checkpoint(model.state_dict(), final_path)
        print(f"Final model: {final_path}")
    else:
        save_checkpoint(model.state_dict(), semantic_path)
        print(f"Semantic model: {semantic_path}")


if __name__ == "__main__":
    main()
