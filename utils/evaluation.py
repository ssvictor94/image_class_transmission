"""
模型评估（论文 Section VI, Fig.6/7/8）
====================================

评估时使用与推理一致的路径:
  - mask 硬离散化（Section IV-A, 式(12)）
  - masked self-attention 隔离被丢弃 token
  - 仅传输活跃 token（gather/scatter）
  - 实测压缩比 ρ 作为 Fig.6/7 横轴
"""
import torch


@torch.no_grad()
def evaluate_model(model, dataloader, gamma, snr_db, device="cuda", discretize=True):
    """
    评估给定配置 γ=(α, r) 与 SNR 下的分类准确率 Λ 和实测压缩比 ρ。

    对应论文 Fig.6（低 SNR）/ Fig.7（高 SNR）中的一个数据点。

    Args:
        gamma:   {'alpha': α, 'r': r}
        snr_db:  信道 SNR（dB）

    Returns:
        accuracy: Top-1 分类准确率
        rho:      平均实测压缩比 ρ = q/p
    """
    model.eval()
    correct = 0
    total = 0
    rho_sum = 0.0

    alpha = gamma["alpha"]
    r = gamma["r"]

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        logits, _, n_active = model.forward_full(
            images, alpha, r, snr_db,
            min_keep_ratio=0.0, discretize=discretize,
        )
        batch_rho = model.average_rho(n_active, r).item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        rho_sum += batch_rho * labels.size(0)

    return correct / max(total, 1), rho_sum / max(total, 1)


@torch.no_grad()
def evaluate_semantic(model, dataloader, alpha, device="cuda", discretize=True):
    """
    Stage 1 语义 ViT 评估（无 JSCC）。

    对应论文 Fig.8: 不同 token budget α 下的准确率（无语信道）。
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits, _, _ = model.forward_semantic(
            images, alpha, min_keep_ratio=0.0, discretize=discretize,
        )
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)
