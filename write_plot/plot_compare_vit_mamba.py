"""
ViT vs MambaVision Fig.6/7 对照图
=================================

读取:
  results/fig6_7_data.csv              (ViT)
  results/fig6_7_data_mambavision.csv  (MambaVision)

输出到 paper_figures/compare/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper_figures" / "compare"


def load():
    vit = pd.read_csv(RESULTS / "fig6_7_data.csv")
    mamba = pd.read_csv(RESULTS / "fig6_7_data_mambavision.csv")
    return vit, mamba


def plot_snr(vit, mamba, snr, tag):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r, marker in [(0.5, "o"), (0.25, "s"), (0.1, "^")]:
        for df, name, ls in [
            (vit, "ViT", "-"),
            (mamba, "MambaVision", "--"),
        ]:
            sub = df[(df["snr_db"] == snr) & (df["r"] == r)].sort_values("rho")
            if sub.empty:
                continue
            ax.plot(
                sub["rho"], sub["accuracy"], marker=marker, linestyle=ls,
                label=f"{name} r={r}", linewidth=2, markersize=5,
            )
    ax.set_xlabel("Compression ratio ρ")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"ViT vs MambaVision  Accuracy vs ρ (SNR={snr:.0f} dB)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = OUT / f"compare_snr_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vit, mamba = load()
    plot_snr(vit, mamba, -10, "m10")
    plot_snr(vit, mamba, 10, "10")
    print("Done.")


if __name__ == "__main__":
    main()
