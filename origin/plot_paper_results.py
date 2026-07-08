"""Read official-repo evaluation JSON and plot accuracy vs compression ratio rho."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Image: 224x224x3; DeiT-tiny embed dim = 192 (see main.py kn formula)
EMBED_DIM = 192
IMAGE_PIXELS = 224 * 224 * 3

# JSCC experiment folder -> encoder output_size (from configs/jscc/proposal.yaml)
JSCC_COMPRESSION = {
    "gaussian_noise": 0.5,
    "gaussian_no_training": 0.5,
    "gaussian_noise_025": 0.25,
    "gaussian_no_training_025": 0.25,
    "gaussian_noise_01": 0.1,
    "gaussian_no_training_01": 0.1,
}

JSCC_LABELS = {
    "gaussian_noise": "Proposal + JSCC (trained, c=0.5)",
    "gaussian_no_training": "Proposal + JSCC (frozen, c=0.5)",
    "gaussian_noise_025": "Proposal + JSCC (trained, c=0.25)",
    "gaussian_no_training_025": "Proposal + JSCC (frozen, c=0.25)",
    "gaussian_noise_01": "Proposal + JSCC (trained, c=0.1)",
    "gaussian_no_training_01": "Proposal + JSCC (frozen, c=0.1)",
}


def load_semantic_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "0" in data and len(data) <= 5:
        data = data["0"]
    return data


def compute_rho(all_sizes, alpha, compression):
    """rho = transmitted_symbols / raw_pixels (same as kn in main.py)."""
    block_sizes = all_sizes[str(alpha)] if str(alpha) in all_sizes else all_sizes[alpha]
    mean_tokens = list(block_sizes.values())[-1][0]
    transmitted = mean_tokens * (EMBED_DIM * compression)
    return transmitted / IMAGE_PIXELS


def extract_curve(semantic_data, snr, compression):
    closest = min(semantic_data.keys(), key=lambda s: abs(float(s) - snr))
    block = semantic_data[closest]
    acc_map = block["accuracy"]
    sizes = block["all_sizes"]

    alphas = sorted(acc_map.keys(), key=float)
    rhos, accs = [], []
    for a in alphas:
        rhos.append(compute_rho(sizes, a, compression))
        accs.append(acc_map[a])
    return np.array(rhos), np.array(accs) * 100.0, float(closest), alphas


def load_semantic_only(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    acc = data["accuracy"]
    flops = data.get("flops", {})
    full_flops = data.get("full_flops")
    rows = []
    for a in sorted(acc.keys(), key=float):
        row = {"alpha": float(a), "accuracy_pct": acc[a] * 100.0}
        if full_flops and a in flops:
            row["flops_ratio"] = flops[a][0] / full_flops
        rows.append(row)
    return rows


def find_latest_run(results_root):
    candidates = list(Path(results_root).glob("jscc/**/0/evaluation_results/semantic_flops.json"))
    if not candidates:
        raise FileNotFoundError(f"No runs under {results_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime).parents[2]


def main():
    parser = argparse.ArgumentParser(description="Plot paper-style accuracy vs rho from saved JSON.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Experiment dir containing 0/ subfolder (default: latest under ./results)",
    )
    parser.add_argument("--snr", type=float, default=10.0, help="SNR in dB for JSCC curves")
    parser.add_argument("--out", default="paper_plots", help="Output directory for PNG/CSV")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run("./results")
    run_dir = run_dir.resolve()
    exp0 = run_dir / "0"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using run: {run_dir}")

    # --- Fig A: semantic-only (no channel), accuracy vs FLOPs ratio ---
    sem_path = exp0 / "evaluation_results" / "semantic_flops.json"
    if sem_path.is_file():
        rows = load_semantic_only(sem_path)
        if rows and "flops_ratio" in rows[0]:
            xs = [r["flops_ratio"] for r in rows]
            ys = [r["accuracy_pct"] for r in rows]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(xs, ys, "o-", label="Semantic ViT (no channel)")
            ax.set_xlabel("FLOPs ratio (relative to full model)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Semantic evaluation (validation, AWGN off)")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "fig_semantic_flops.png", dpi=150)
            plt.close(fig)

        csv_lines = ["alpha,accuracy_pct,flops_ratio"]
        for r in rows:
            fr = r.get("flops_ratio", "")
            csv_lines.append(f"{r['alpha']},{r['accuracy_pct']:.4f},{fr}")
        (out_dir / "semantic_flops.csv").write_text("\n".join(csv_lines), encoding="utf-8")
        print(f"Wrote {out_dir / 'fig_semantic_flops.png'} and semantic_flops.csv")

    # --- Fig B: JSCC Pareto at fixed SNR ---
    fig, ax = plt.subplots(figsize=(7, 5))
    csv_rows = ["method,snr_db,rho,accuracy_pct,alpha"]

    for name, compression in JSCC_COMPRESSION.items():
        json_path = exp0 / name / "semantic.json"
        if not json_path.is_file():
            continue
        data = load_semantic_json(json_path)
        rhos, accs, used_snr, alphas = extract_curve(data, args.snr, compression)
        label = JSCC_LABELS.get(name, name)
        ax.plot(rhos, accs, "o-", markersize=4, label=label)
        for rho, acc, alpha in zip(rhos, accs, alphas):
            csv_rows.append(f"{name},{used_snr},{rho:.6f},{acc:.4f},{alpha}")

    ax.set_xlabel(r"Compression ratio $\rho$ (transmitted symbols / image pixels)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"JSCC evaluation @ SNR ≈ {args.snr} dB (Imagenette val)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_jscc_snr{int(args.snr)}.png", dpi=150)
    plt.close(fig)

    (out_dir / f"jscc_snr{int(args.snr)}.csv").write_text("\n".join(csv_rows), encoding="utf-8")
    print(f"Wrote {out_dir / f'fig_jscc_snr{int(args.snr)}.png'} and jscc CSV")


if __name__ == "__main__":
    main()
