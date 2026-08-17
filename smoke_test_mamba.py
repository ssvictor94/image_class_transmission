"""
MambaVision DJSCC 冒烟测试（建议在 WSL mamba_djscc 环境运行）

用法:
  python smoke_test_mamba.py
"""
import torch

from utils.model_factory import build_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model = build_model(pretrained=True, backbone="mambavision").to(device)
    model.eval()

    x = torch.randn(2, 3, 224, 224, device=device)
    alpha = torch.tensor([0.5, 0.25], device=device)

    with torch.no_grad():
        logits_s, masks, n_s = model.forward_semantic(x, alpha, discretize=True)
        logits_f, _, n_f = model.forward_full(x, alpha, r=0.25, snr_db=10.0)
        _, _, n_high = model.forward_semantic(x, 1.0, discretize=True)

    print("semantic logits:", tuple(logits_s.shape), "n_active:", n_s.tolist())
    print("full logits:", tuple(logits_f.shape), "n_active:", n_f.tolist())
    print("adaptive layers:", len(masks), "mean keep:", [m.mean().item() for m in masks])
    print("n_active alpha=1.0:", n_high.tolist())

    assert logits_s.shape == (2, 10)
    assert logits_f.shape == (2, 10)
    assert len(masks) == 5, f"期望 5 层预算损失，得到 {len(masks)}"
    assert float(n_high.float().mean()) == 196.0, "α=1 应保留全部 196 patch"
    # 未训练时官方阈值离散会几乎全保留（fl bias≫fh）；预算约束靠 Stage1 损失学会
    print("OK")


if __name__ == "__main__":
    main()
