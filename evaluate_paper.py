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
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    ALPHA_VALUES_EVAL,
    BEST_MODEL_PATH,
    FIGURES_DIR,
    R_VALUES,
    RESULTS_DIR,
    SNR_EVAL_LIST,
    SNR_HIGH,
    SNR_LOW,
)
from utils.data import get_dataloaders
from utils.evaluation import evaluate_model
from utils.model_factory import load_trained_model

FIG67_CSV = RESULTS_DIR / "fig6_7_data.csv"


def sweep(model, val_loader, device, csv_path=FIG67_CSV, resume=False):
    """扫描 (α, r, SNR) 网格，记录 accuracy 与实测 ρ。"""
    rows = []
    done = set()

    if resume and csv_path.is_file():
        existing = pd.read_csv(csv_path)
        rows = existing.to_dict("records")
        for _, r in existing.iterrows():
            done.add((r["alpha"], r["r"], r["snr_db"]))

    combos = [(a, r, s) for r in R_VALUES for a in ALPHA_VALUES_EVAL for s in SNR_EVAL_LIST]
    remaining = [c for c in combos if c not in done]
    print(f"共 {len(combos)} 组，剩余 {len(remaining)} 组")

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


def plot_fig67(df, out_dir=FIGURES_DIR):
    """绘制 Fig.6（SNR=-10dB）与 Fig.7（SNR=+10dB）。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    for snr, tag in [(SNR_LOW, "snr_m10"), (SNR_HIGH, "snr_10")]:
        sub = df[df["snr_db"] == snr]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for r in sorted(sub["r"].unique()):
            rs = sub[sub["r"] == r].sort_values("rho")
            ax.plot(rs["rho"], rs["accuracy"], "o-", label=f"r={r}")

        ax.set_xlabel("Compression ratio ρ")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy vs ρ (SNR={snr:.0f} dB)")
        ax.legend()
        ax.grid(True, alpha=0.3)
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
        ax.plot(sub["rho"], sub["accuracy"], "o-")
        ax.set_xlabel("ρ")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Semantic JSCC r={r}, SNR=10dB")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"fig67_r{r}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="复现论文 Fig.6/7")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not BEST_MODEL_PATH.is_file():
        print(f"未找到 {BEST_MODEL_PATH}，请先运行: python train.py --stage djscc")
        return

    model = load_trained_model(BEST_MODEL_PATH, device=args.device)
    _, val_loader = get_dataloaders()

    t0 = time.time()
    df = sweep(model, val_loader, args.device, resume=args.resume)
    print(f"评估耗时 {time.time() - t0:.0f}s")
    plot_fig67(df)


if __name__ == "__main__":
    main()
