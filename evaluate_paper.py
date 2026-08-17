"""
论文 Figure 6 / 7 复现脚本
==========================

Fig.6: Accuracy vs ρ，低 SNR（-10 dB）
Fig.7: Accuracy vs ρ，高 SNR（+10 dB）

实验方法（Section VI）:
  1. 固定 JSCC 压缩率 r ∈ {0.5, 0.25, 0.1}
  2. 扫描 token budget α ∈ [0.05, 1.0]
  3. 对每个 (α, r, SNR) 在验证集上评估准确率和 **实测** ρ
  4. 以 ρ 为横轴、准确率为纵轴绘制 Pareto 曲线

注意: 横轴 ρ 必须使用实测活跃 token 数，不能用 α·N 的设计值。

用法:
  python evaluate_paper.py
  python evaluate_paper.py --resume   # 断点续跑
"""
import argparse
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    ALPHA_VALUES_EVAL,
    BACKBONE,
    FIGURES_DIR,
    PROJECT_ROOT,
    R_VALUES,
    RESULTS_DIR,
    SNR_EVAL_LIST,
    SNR_HIGH,
    SNR_LOW,
)
from utils.data import get_dataloaders
from utils.evaluation import evaluate_model
from utils.model_factory import load_trained_model


def sweep(model, val_loader, device, csv_path, resume=False, snr_list=None):
    """扫描 (α, r, SNR) 网格，记录 accuracy 与实测 ρ。"""
    rows = []
    done = set()
    snrs = list(snr_list) if snr_list is not None else list(SNR_EVAL_LIST)

    if resume and csv_path.is_file():
        existing = pd.read_csv(csv_path)
        rows = existing.to_dict("records")
        for _, r in existing.iterrows():
            done.add((float(r["alpha"]), float(r["r"]), float(r["snr_db"])))

    combos = [(a, r, s) for r in R_VALUES for a in ALPHA_VALUES_EVAL for s in snrs]
    remaining = [c for c in combos if (float(c[0]), float(c[1]), float(c[2])) not in done]
    print(f"共 {len(combos)} 组，剩余 {len(remaining)} 组 (SNR={snrs})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(remaining, desc="Fig.6/7 评估")

    for alpha, r, snr in pbar:
        acc, rho = evaluate_model(
            model, val_loader, {"alpha": alpha, "r": r}, snr_db=snr, device=device,
        )
        row = {"alpha": alpha, "r": r, "snr_db": snr, "accuracy": acc, "rho": rho}
        rows.append(row)
        pbar.set_postfix(acc=f"{acc:.3f}", rho=f"{rho:.4f}")

        pd.DataFrame(rows).to_csv(csv_path, index=False)

    return pd.DataFrame(rows)


def plot_fig67(df, out_dir=FIGURES_DIR, title_prefix=""):
    """绘制 Fig.6（SNR=-10dB）与 Fig.7（SNR=+10dB），风格对齐 ViT 版。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{title_prefix} " if title_prefix else ""

    for snr, tag in [(SNR_LOW, "snr_m10"), (SNR_HIGH, "snr_10")]:
        sub = df[df["snr_db"] == snr]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for r in sorted(sub["r"].unique()):
            rs = sub[sub["r"] == r].sort_values("rho")
            ax.plot(rs["rho"], rs["accuracy"], "o-", label=f"r={r}", linewidth=2, markersize=5)

        ax.set_xlabel("Compression ratio ρ")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{prefix}Accuracy vs ρ (SNR={snr:.0f} dB)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        path = out_dir / f"fig67_{tag}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")

    for r in sorted(df["r"].unique()):
        sub = df[(df["r"] == r) & (df["snr_db"] == 10)]
        if sub.empty:
            continue
        sub = sub.sort_values("rho")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(sub["rho"], sub["accuracy"], "o-", linewidth=2, markersize=5)
        ax.set_xlabel("ρ")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{prefix}Semantic JSCC r={r}, SNR=10dB")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.0, 1.0)
        fig.tight_layout()
        path = out_dir / f"fig67_r{r}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="复现论文 Fig.6/7")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--backbone", choices=["mambavision", "vit"], default=None,
    )
    parser.add_argument(
        "--snr", type=float, nargs="*", default=None,
        help="只评估这些 SNR；默认 Fig.6/7 用 -10 10（更快）。全量用 --snr -10 -5 0 5 10",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    backbone = (args.backbone or BACKBONE).lower()
    best_path = PROJECT_ROOT / "checkpoints" / backbone / "best_model.pth"
    csv_path = RESULTS_DIR / f"fig6_7_data_{backbone}.csv"

    if not best_path.is_file():
        print(f"未找到 {best_path}，请先运行: python train.py --stage djscc --backbone {backbone}")
        return

    snr_list = args.snr if args.snr is not None else [SNR_LOW, SNR_HIGH]
    model = load_trained_model(best_path, device=args.device, backbone=backbone)
    eval_bs = args.batch_size or (64 if backbone == "mambavision" else 128)
    _, val_loader = get_dataloaders(batch_size=eval_bs)

    t0 = time.time()
    df = sweep(
        model, val_loader, args.device, csv_path=csv_path,
        resume=args.resume, snr_list=snr_list,
    )
    print(f"评估耗时 {time.time() - t0:.0f}s")

    # 绘图只用 Fig.6/7 相关 SNR（即使 CSV 含更多点）
    df_plot = df[df["snr_db"].isin([SNR_LOW, SNR_HIGH])].copy()
    out_dir = FIGURES_DIR / backbone
    plot_fig67(df_plot, out_dir=out_dir, title_prefix=backbone)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for png in out_dir.glob("fig67_*.png"):
        dst = FIGURES_DIR / f"{backbone}_{png.name}"
        shutil.copy2(png, dst)
        print(f"Copied {dst}")


if __name__ == "__main__":
    main()
