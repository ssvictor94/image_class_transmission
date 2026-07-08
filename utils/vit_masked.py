"""
Masked Self-Attention（论文 Section IV-A 关键实现细节）
======================================================

问题:
  若仅将被丢弃 token 特征置零（h_i ← m_i · h_i），它们仍作为
  Key/Value 参与 self-attention，活跃 token 的 query 可以从
  被丢弃位置「泄漏」语义信息，导致:
    - 推理时压缩比 ρ 变化对准确率几乎无影响
    - 无法复现论文 Fig.6/7 中低 ρ 时准确率显著下降的现象

解决（论文/官方语义）:
  对 mask=0 的 token，在 attention 中同时屏蔽其 Query 与 Key/Value:
    - Key/Value 不可见 → 其他 token 不能 attend 到它
    - Query 不参与     → 它不能从其他 token 聚合信息

本模块实现带 token 掩码的 ViT Block 前向，用于编码器与解码器所有 block。
"""
import torch

from utils.transmission import EVAL_MASK_THRESHOLD

# 足够大的负数，使 softmax 后权重≈0；避免 -inf 导致 NaN
ATTN_MASK_VALUE = -1e4


def _num_heads(attn):
    if hasattr(attn, "num_heads"):
        return attn.num_heads
    return attn.head_count


def _active_mask(token_keep_mask):
    """将 soft mask 转为 bool 活跃标记（阈值 EVAL_MASK_THRESHOLD）。"""
    if token_keep_mask.dtype == torch.bool:
        return token_keep_mask
    return token_keep_mask > EVAL_MASK_THRESHOLD


def forward_block_with_token_mask(block, x, token_keep_mask):
    """
    带 token 掩码的 Transformer block 前向。

    Args:
        block:            timm VisionTransformer Block（含 attn + mlp）
        x:                [B, N, C] token 特征
        token_keep_mask:  [B, N] 1=活跃参与 attention，0=完全隔离

    Returns:
        x: [B, N, C] 更新后的 token 特征
    """
    B, N, C = x.shape
    attn_mod = block.attn
    nh = _num_heads(attn_mod)
    head_dim = C // nh
    active = _active_mask(token_keep_mask)

    # --- Multi-Head Self-Attention ---
    x_norm = block.norm1(x)
    qkv = attn_mod.qkv(x_norm).reshape(B, N, 3, nh, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn = (q @ k.transpose(-2, -1)) * attn_mod.scale

    # 屏蔽非活跃 token 的 Key/Value（其他 token 不能 attend 到被丢弃位置）
    key_pad = active.unsqueeze(1).unsqueeze(2)
    attn = attn.masked_fill(~key_pad, ATTN_MASK_VALUE)

    # 非活跃 token 的 Query 行置零（被丢弃 token 不聚合任何信息）
    row_active = active.unsqueeze(1).unsqueeze(-1)
    attn = attn.masked_fill(~row_active, 0.0)

    attn = torch.softmax(attn, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)

    x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x_attn = attn_mod.proj(x_attn)
    x_attn = x_attn * active.unsqueeze(-1).float()

    x = x + block.drop_path1(x_attn)

    # --- FFN：被丢弃 token 的 MLP 输出也置零 ---
    x_norm2 = block.norm2(x)
    x_mlp = block.mlp(x_norm2)
    x_mlp = x_mlp * active.unsqueeze(-1).float()
    x = x + block.drop_path2(x_mlp)

    return x
