"""
MambaVision Block 的 token 掩码前向（对齐论文 / 官方物理删除语义）
================================================================

- Attention mixer: 屏蔽非活跃 Query/Key/Value（与 ViT 一致）
- Mamba mixer: 仅对活跃 token 子序列跑 SSM 再 scatter（对齐官方 bs=1 gather）
- MLP / 残差输出均乘 mask，丢弃 token 不更新
"""
import torch
from torch.nn import functional as F

from utils.transmission import EVAL_MASK_THRESHOLD

ATTN_MASK_VALUE = -1e4


def _active_mask(token_keep_mask):
    if token_keep_mask.dtype == torch.bool:
        return token_keep_mask
    return token_keep_mask > EVAL_MASK_THRESHOLD


def forward_attention_mixer_masked(attn_mod, x, active):
    """MambaVision Attention，带 token 隔离。x: [B, N, C], active: [B, N] bool。"""
    B, N, C = x.shape
    nh = attn_mod.num_heads
    head_dim = attn_mod.head_dim
    qkv = attn_mod.qkv(x).reshape(B, N, 3, nh, head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)

    attn = (q * attn_mod.scale) @ k.transpose(-2, -1)
    key_pad = active.unsqueeze(1).unsqueeze(2)
    attn = attn.masked_fill(~key_pad, ATTN_MASK_VALUE)
    row_active = active.unsqueeze(1).unsqueeze(-1)
    attn = attn.masked_fill(~row_active, 0.0)
    attn = torch.softmax(attn, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    if attn_mod.attn_drop.p > 0 and attn_mod.training:
        attn = attn_mod.attn_drop(attn)

    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
    out = attn_mod.proj(out)
    out = attn_mod.proj_drop(out)
    return out * active.unsqueeze(-1).float()


def forward_mamba_mixer_masked(mixer, x, active):
    """
    MambaVisionMixer：对齐官方 eval 的「物理删除」语义。

    仅对活跃 token 构成子序列跑 SSM，再 scatter 回原位置。
    丢弃位置不占用序列步、不写入状态（优于单纯置零）。
    """
    B, N, C = x.shape
    active_f = active.unsqueeze(-1).float()
    counts = active.sum(dim=1)

    # 全活跃：无需 gather
    if bool((counts == N).all().item()):
        return mixer(x) * active_f

    # 全不活跃
    if bool((counts == 0).all().item()):
        return x.new_zeros(B, N, C)

    out = x.new_zeros(B, N, C)
    for b in range(B):
        idx = active[b].nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        if idx.numel() == N:
            out[b] = mixer(x[b:b + 1])[0]
        else:
            y = mixer(x[b:b + 1, idx])  # [1, n_act, C]
            out[b, idx] = y[0]
    return out * active_f


def forward_mamba_block_masked(block, x, token_keep_mask):
    """
    单个 MambaVision Block（Attention 或 MambaVisionMixer）+ MLP，带 mask。

    Args:
        block: mambavision.models.mamba_vision.Block
        x: [B, N, C]
        token_keep_mask: [B, N]
    """
    active = _active_mask(token_keep_mask)
    active_f = active.unsqueeze(-1).float()
    x = x * active_f

    x_norm = block.norm1(x)
    mixer = block.mixer
    # Attention 有 qkv；MambaVisionMixer 有 in_proj / A_log
    if hasattr(mixer, "qkv"):
        mixed = forward_attention_mixer_masked(mixer, x_norm, active)
    else:
        mixed = forward_mamba_mixer_masked(mixer, x_norm, active)

    gamma1 = block.gamma_1
    if not isinstance(gamma1, torch.Tensor):
        mixed = gamma1 * mixed
    else:
        mixed = gamma1 * mixed
    x = x + block.drop_path(mixed)
    x = x * active_f

    x_norm2 = block.norm2(x)
    mlp_out = block.mlp(x_norm2) * active_f
    gamma2 = block.gamma_2
    if isinstance(gamma2, torch.Tensor):
        mlp_out = gamma2 * mlp_out
    else:
        mlp_out = gamma2 * mlp_out
    x = x + block.drop_path(mlp_out)
    return x * active_f


def window_partition_mask(mask_hw, window_size):
    """
    mask_hw: [B, H, W] → window tokens mask [nW*B, window_size*window_size]
    """
    B, H, W = mask_hw.shape
    x = mask_hw.unsqueeze(1)  # [B,1,H,W]
    x = x.view(B, 1, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size)
    return windows


def downsample_token_mask(patch_mask, h, w):
    """
    14×14 patch mask → 7×7（与 stride-2 conv downsample 对齐）。
    2×2 区域内任一活跃则下采样格点活跃（max-pool）。
    patch_mask: [B, H*W]
    """
    B = patch_mask.size(0)
    m = patch_mask.view(B, 1, h, w).float()
    m = F.max_pool2d(m, kernel_size=2, stride=2)
    return m.view(B, -1)
