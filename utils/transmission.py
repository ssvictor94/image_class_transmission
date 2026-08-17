"""
语义 Token 传输与压缩比计算（论文 Section III–IV）
==============================================

本模块实现:
  1. 推理时 mask 离散化（Section IV-A, 式(12) 思想）
  2. gather/scatter — 仅传输活跃 token（Section IV-B）
  3. 功率归一化（式(4)）
  4. 实测压缩比 ρ（Section III, Fig.6/7 横轴）

压缩比定义（论文 Section III）:
  ρ = q / p

  q = n_α · o_r = n_α · (r · d)   — 实际传输的复数符号总数
  p = H · W · C = 224 × 224 × 3    — 输入图像像素数

注意: ρ 必须使用 **实测** n_α（推理离散化后的活跃 token 数），
      不能用设计值 α·N 代替，否则 Fig.6/7 曲线会失真。
"""
import torch

from config import D_MODEL, TOTAL_PIXELS

# 推理时将 soft mask 二值化的阈值（官方 eval_threshold=1e-3）
EVAL_MASK_THRESHOLD = 1e-3


def progressive_layer_hard_mask(soft_mask, has_cls=True):
    """官方 AdaptiveBlock 推理硬掩码：阈值二值化（eval_threshold=1e-3）。"""
    return discretize_mask(soft_mask, threshold=EVAL_MASK_THRESHOLD, has_cls=has_cls)


def topk_patch_mask(scores, alpha, prev_patch_mask=None, min_k=1):
    """
    编码器出口按预算 α 保留 top-k patch（保证 Fig.6/7 的 n_α≈αN）。

    仅用于最终 mask / STE，不要在每个 S_l 中间层调用。
    """
    B, N = scores.shape
    if prev_patch_mask is None:
        prev_patch_mask = torch.ones(B, N, device=scores.device, dtype=scores.dtype)
    masked = scores * prev_patch_mask

    if not isinstance(alpha, torch.Tensor):
        alpha = torch.full((B,), float(alpha), device=scores.device)
    elif alpha.dim() == 0:
        alpha = alpha.expand(B)
    ks = (alpha.float() * N).round().long().clamp(min=min_k, max=N)

    out = torch.zeros_like(scores)
    max_k = int(ks.max().item())
    _, idx = torch.topk(masked, max_k, dim=1)
    for b in range(B):
        k = int(ks[b].item())
        out[b, idx[b, :k]] = 1.0
    return out


def apply_budget_hard_mask(soft_mask, scores, alpha, has_cls=True):
    """
    将 soft mask 替换为 top-k 硬 mask（budget 恒为 1；ViT 时 CLS 恒为 1）。
    scores: [B, N_patch]  token 重要性（f_l 输出）
    """
    B = soft_mask.size(0)
    if has_cls:
        prev = soft_mask[:, 1:-1] if soft_mask.size(1) > 2 else None
    else:
        prev = soft_mask[:, :-1] if soft_mask.size(1) > 1 else None
    patch_hard = topk_patch_mask(scores, alpha, prev_patch_mask=prev)
    tail_one = torch.ones(B, 1, device=soft_mask.device, dtype=patch_hard.dtype)
    if has_cls:
        cls_one = torch.ones(B, 1, device=soft_mask.device, dtype=patch_hard.dtype)
        return torch.cat([cls_one, patch_hard, tail_one], dim=1)
    return torch.cat([patch_hard, tail_one], dim=1)


def ste_topk_mask(soft_mask, scores, alpha, has_cls=True):
    """训练后期 STE：前向 top-k，反向走 soft mask。"""
    hard = apply_budget_hard_mask(soft_mask, scores, alpha, has_cls=has_cls)
    return hard + soft_mask - soft_mask.detach()


def discretize_mask(mask, threshold=EVAL_MASK_THRESHOLD, has_cls=True):
    """
    推理离散化（Section IV-A, 式(12) 思想）:
      m_i = 1  if m_i > τ
      m_i = 0  otherwise

    budget token 恒为 1；ViT 时 CLS 也恒为 1。
    """
    hard = (mask > threshold).float()
    if has_cls:
        hard[:, 0] = 1.0
    if hard.size(1) > 1:
        hard[:, -1] = 1.0
    return hard


def ste_discretize(mask, threshold=EVAL_MASK_THRESHOLD, has_cls=True):
    """
    Straight-Through Estimator: 前向硬离散化，反向对 soft mask 传梯度。
    用于 Stage 2 缩小训练 soft mask 与推理 hard mask 的差距。
    """
    hard = discretize_mask(mask, threshold, has_cls=has_cls)
    return hard + mask - mask.detach()


def count_active_tokens(mask, threshold=EVAL_MASK_THRESHOLD, has_cls=True):
    """统计活跃 patch token 数（不含 CLS / budget）。"""
    active = mask > threshold
    if has_cls:
        if active.size(1) > 2:
            active = active[:, 1:-1]
    else:
        if active.size(1) > 1:
            active = active[:, :-1]
    return active.float().sum(dim=1)


def gather_active_tokens(tokens, mask, threshold=EVAL_MASK_THRESHOLD, has_cls=True):
    """
    将活跃 token gather 为紧凑序列，供 JSCC 编码器输入。

    论文语义: 边缘端仅传输 mask=1 的 token 特征，不传输被丢弃 token。
    budget token 仅用于编码器侧选择，不参与信道传输。

    Returns:
        out:    [B, max_n_α, D]  padded 活跃 token
        counts: [B] 每个样本的活跃 token 数 n_α
    """
    B, N, D = tokens.shape
    hard = discretize_mask(mask, threshold, has_cls=has_cls)
    if hard.dtype != torch.bool:
        hard = hard > threshold

    # budget token（序列末尾）不参与传输
    if N > 1:
        hard = hard.clone()
        hard[:, -1] = False

    counts = hard.sum(dim=1)
    max_n = int(counts.max().item())
    max_n = max(max_n, 1)

    out = tokens.new_zeros(B, max_n, D)
    for b in range(B):
        idx = hard[b].nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            idx = torch.tensor([0], device=tokens.device)
        sel = tokens[b, idx]
        out[b, : sel.size(0)] = sel

    return out, counts


def scatter_tokens(recovered, original_shape, mask, threshold=EVAL_MASK_THRESHOLD, has_cls=True):
    """
    JSCC 解码后将恢复的 token scatter 回原序列位置。

    被丢弃位置保持零向量；budget token 位置不写入（由编码器本地生成）。
    """
    B, N, D = original_shape
    hard = discretize_mask(mask, threshold, has_cls=has_cls)
    out = recovered.new_zeros(B, N, D)

    for b in range(B):
        idx = (hard[b] > threshold).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() > 0 and idx[-1].item() == N - 1:
            idx = idx[:-1]
        n = min(idx.numel(), recovered.size(1))
        if n > 0:
            out[b, idx[:n]] = recovered[b, :n]

    return out


def compression_ratio_from_transmission(n_active_tokens, r, d_model=D_MODEL, total_pixels=TOTAL_PIXELS):
    """
    实测压缩比 ρ（对齐官方 main.py 绘图公式）:

      ρ = (N_tokens · r · d) / p

    N_tokens: 最后自适应层活跃 token 数（含 CLS，不含 budget）
    """
    if isinstance(n_active_tokens, torch.Tensor):
        n_active_tokens = n_active_tokens.float().mean()
    symbols = float(n_active_tokens) * d_model * r
    return symbols / total_pixels


def count_transmission_tokens(mask, threshold=EVAL_MASK_THRESHOLD, has_cls=True):
    """统计参与 JSCC 的 token 数（活跃 patch [+CLS]，不含 budget）。"""
    hard = discretize_mask(mask, threshold, has_cls=has_cls)
    if hard.size(1) > 1:
        hard = hard.clone()
        hard[:, -1] = 0.0
    return hard.sum(dim=1).float()
