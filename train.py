"""
论文 Section VI 两阶段训练脚本

Stage 2 策略 v3（更接近论文数值）:
  - 仅训练 JSCC 编解码器，冻结服务器 block（DJSCC_FREEZE_SERVER）
  - 55% batch 采样 α∈[0.05, 0.2]，强化低 ρ 端
  - 70% batch 采样 SNR∈[-10, 0]

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
    BEST_MODEL_PATH,
    CHECKPOINT_DIR,
    DJSCC_EPOCHS,
    DJSCC_FREEZE_SERVER,
    DJSCC_STE_START_EPOCH,
    FINAL_MODEL_PATH,
    LEARNING_RATE,
    MICRO_BATCH_SIZE,
    MIN_KEEP_RATIO_END,
    MIN_KEEP_RATIO_START,
    PROGRESSIVE_EPOCHS,
    R_VALUES_TRAIN,
    SEMANTIC_EPOCHS,
    SEMANTIC_MODEL_PATH,
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
    """
    低 ρ 偏置采样：low_bias 比例来自 α∈[ALPHA_LOW_MIN, ALPHA_LOW_MAX]，
    其余来自 α∈[ALPHA_LOW_MAX, 1.0]。
    """
    n_low = int(batch_size * low_bias)
    n_high = batch_size - n_low
    low = torch.empty(n_low, device=device).uniform_(ALPHA_LOW_MIN, ALPHA_LOW_MAX)
    high = torch.empty(n_high, device=device).uniform_(ALPHA_LOW_MAX, 1.0)
    alpha = torch.cat([low, high], dim=0)
    perm = torch.randperm(batch_size, device=device)
    return alpha[perm]


def sample_snr(batch_size, device):
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
    use_ste = epoch >= DJSCC_STE_START_EPOCH
    logits, _, _ = model.forward_full(
        images, alpha, r, snr_db,
        min_keep_ratio=0.0,
        discretize=False,
        use_ste=use_ste,
    )
    return F.cross_entropy(logits, labels)


def train_epoch(model, loader, optimizer, epoch, stage, device):
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
            loss = semantic_step(model, images, labels, alpha, epoch) / ACCUMULATION_STEPS
        else:
            alpha = sample_alpha(images.size(0), device)
            r = random.choice(R_VALUES_TRAIN)
            snr_db = sample_snr(images.size(0), device)
            loss = djscc_step(model, images, labels, alpha, r, snr_db, epoch) / ACCUMULATION_STEPS

        loss.backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * ACCUMULATION_STEPS
        n_batches += 1
        if batch_idx % 50 == 0:
            print(f"  batch {batch_idx}, loss={loss.item() * ACCUMULATION_STEPS:.4f}")

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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_dataloaders(batch_size=MICRO_BATCH_SIZE)

    print(f"训练集: {len(train_loader.dataset)}, 验证集: {len(val_loader.dataset)}")
    print(f"Stage: {args.stage}, 等效 batch={MICRO_BATCH_SIZE * ACCUMULATION_STEPS}")

    model = build_model(pretrained=True).to(args.device)

    if args.stage == "djscc":
        from utils.model_factory import _valid_checkpoint
        if not _valid_checkpoint(SEMANTIC_MODEL_PATH):
            raise SystemExit(
                f"\n错误: Stage 1 权重无效: {SEMANTIC_MODEL_PATH}\n"
                "  该文件为空或不存在，请先完成语义预训练:\n"
                "  python train.py --stage semantic\n"
            )
        load_semantic_checkpoint(model, device=args.device)
        if DJSCC_FREEZE_SERVER:
            model.freeze_jscc_only()
            print("Stage2: 仅训练 JSCC，服务器 block 已冻结")
        else:
            model.freeze_djscc_with_server()
            print("Stage2: 训练 JSCC + 服务器 block")
        print(f"Stage2: α 低预算采样 {ALPHA_LOW_BIAS:.0%} ∈ [{ALPHA_LOW_MIN}, {ALPHA_LOW_MAX}]")
        total_epochs = DJSCC_EPOCHS
        save_path = BEST_MODEL_PATH
    else:
        model.unfreeze_semantic()
        total_epochs = SEMANTIC_EPOCHS
        save_path = SEMANTIC_MODEL_PATH

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
    log_path = CHECKPOINT_DIR / f"train_{args.stage}.log"

    for epoch in range(1, total_epochs + 1):
        print(f"\n=== Epoch {epoch}/{total_epochs} ({args.stage}) ===")
        avg_loss = train_epoch(model, train_loader, optimizer, epoch, args.stage, args.device)

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
        save_checkpoint(model.state_dict(), FINAL_MODEL_PATH)
        print(f"Final model: {FINAL_MODEL_PATH}")
    else:
        save_checkpoint(model.state_dict(), SEMANTIC_MODEL_PATH)
        print(f"Semantic model: {SEMANTIC_MODEL_PATH}")


if __name__ == "__main__":
    main()
