"""
自适应语义 MambaVision 编码器（对齐论文 Section IV-A）
====================================================

控制变量仅骨干；token 选择/掩码语义与 AdaptiveViTEncoder 一致。
切分按 MambaVision 层次，而非照搬 ViT 的 6 层:

  E(x) → Stage0/1
       → Stage2 全部 blocks（默认）:
            block0 无选择；block1..s 前插入 S_l
            被丢弃 token 经 masked Attention / 置零 Mamba 隔离
       → 传输 [patches, budget]（仍在 14×14，尚未 downsample）

服务器侧再做 Stage2↓ + Stage3。
"""
import torch
from torch import nn
from torch.nn import functional as F

from token_selection_module import TokenSelectionModule
from utils.mamba_masked import (
    forward_mamba_block_masked,
    window_partition_mask,
)
from utils.transmission import apply_budget_hard_mask, ste_topk_mask


def _window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def _window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


def run_one_stage2_block(block, x, patch_mask, window_size):
    """
    单个 Stage2 block：window 划分 → masked block → 还原。

    x: [B, C, H, W]
    patch_mask: [B, H*W] （不含 budget）
    """
    B, C, H, W = x.shape
    pad_r = (window_size - W % window_size) % window_size
    pad_b = (window_size - H % window_size) % window_size
    if pad_r > 0 or pad_b > 0:
        x = F.pad(x, (0, pad_r, 0, pad_b))
        # mask pad 为 0（填充区不参与）
        mask_hw = patch_mask.view(B, H, W)
        mask_hw = F.pad(mask_hw, (0, pad_r, 0, pad_b), value=0.0)
    else:
        mask_hw = patch_mask.view(B, H, W)

    _, _, Hp, Wp = x.shape
    x_win = _window_partition(x, window_size)
    m_win = window_partition_mask(mask_hw, window_size)
    x_win = forward_mamba_block_masked(block, x_win, m_win)
    x = _window_reverse(x_win, window_size, Hp, Wp)
    if pad_r > 0 or pad_b > 0:
        x = x[:, :, :H, :W].contiguous()
    return x


class AdaptiveMambaEncoder(nn.Module):
    """
    论文 Figure 2 左侧 Adaptive Semantic Encoder 的 MambaVision 对应实现。

    输出序列: [patch_0, ..., patch_195, budget]  （无 CLS，属骨干差异）
    """

    def __init__(
        self,
        mamba_backbone,
        s=5,
        d_model=320,
        encoder_split=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.s = s
        self.has_cls = False

        self.patch_embed = mamba_backbone.patch_embed
        self.level0 = mamba_backbone.levels[0]
        self.level1 = mamba_backbone.levels[1]
        self.level2 = mamba_backbone.levels[2]
        self.level3 = mamba_backbone.levels[3]
        self.norm = mamba_backbone.norm
        self.avgpool = mamba_backbone.avgpool
        self.head = mamba_backbone.head

        n_stage2 = len(self.level2.blocks)
        # None → Stage2 全部在边缘（推荐）；也可传入 int 做消融
        self.encoder_split = n_stage2 if encoder_split is None else int(encoder_split)
        self.encoder_split = max(1, min(self.encoder_split, n_stage2))

        self.b_l = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.b_h = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.selection_modules = nn.ModuleList([
            TokenSelectionModule(d_model=d_model, has_cls=False) for _ in range(s)
        ])

    @property
    def stage2_downsample(self):
        return self.level2.downsample

    @property
    def stage2_window_size(self):
        return self.level2.window_size

    def _budget_token(self, alpha, batch_size, device):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, device=device).expand(batch_size)
        elif alpha.dim() == 0:
            alpha = alpha.expand(batch_size)
        elif alpha.size(0) == 1 and batch_size > 1:
            alpha = alpha.expand(batch_size)
        alpha_exp = alpha.view(-1, 1, 1)
        return alpha_exp * self.b_h + (1 - alpha_exp) * self.b_l

    def forward(self, x, alpha, min_keep_ratio=0.0, use_hard_mask=None, use_ste=False):
        if use_hard_mask is None:
            use_hard_mask = not self.training and not use_ste

        if not isinstance(alpha, torch.Tensor):
            alpha_t = torch.tensor(alpha, device=x.device)
        else:
            alpha_t = alpha

        B = x.shape[0]
        budget = self._budget_token(alpha_t, B, x.device)

        # Stage0/1：早期卷积特征（对应 ViT 的 patch embed + 浅层）
        feat = self.patch_embed(x)
        feat = self.level0(feat)
        feat = self.level1(feat)
        # feat: [B, 320, 14, 14]
        _, C, H, W = feat.shape
        self._last_spatial = (H, W)

        tokens = feat.flatten(2).transpose(1, 2).contiguous()  # [B, 196, C]
        tokens = torch.cat([tokens, budget], dim=1)
        mask = torch.ones(B, tokens.size(1), device=tokens.device)
        layer_avg_masks = []
        last_patch_scores = None

        win = self.stage2_window_size
        edge_blocks = self.level2.blocks[: self.encoder_split]

        for layer_idx, block in enumerate(edge_blocks):
            # 与 ViT 一致：block 1..s 前插入 S_l
            if layer_idx > 0 and (layer_idx - 1) < len(self.selection_modules):
                sel_idx = layer_idx - 1
                new_mask = self.selection_modules[sel_idx](
                    tokens, budget.squeeze(1), mask, min_keep_ratio,
                )
                last_patch_scores = self.selection_modules[sel_idx].fl(
                    tokens[:, :-1]
                ).squeeze(-1)
                # 中间层 soft 累积；勿对微小正数做阈值二值化（否则 n_α 与 α 脱钩）
                mask = new_mask

            patch_mask = mask[:, :-1]
            tokens = tokens * mask.unsqueeze(-1)
            feat = tokens[:, :-1].transpose(1, 2).reshape(B, C, H, W)
            feat = feat * patch_mask.view(B, 1, H, W)
            feat = run_one_stage2_block(block, feat, patch_mask, win)
            tokens = torch.cat([
                feat.flatten(2).transpose(1, 2).contiguous(),
                budget,
            ], dim=1)
            tokens = tokens * mask.unsqueeze(-1)

            if layer_idx > 0 and (layer_idx - 1) < len(self.selection_modules):
                layer_avg_masks.append(mask[:, :-1].mean(dim=1))

        # 出口按 α hard top-k，保证传输 n_α≈αN、ρ 随 α 变化
        if (use_hard_mask or use_ste) and last_patch_scores is not None:
            if use_ste and self.training:
                mask = ste_topk_mask(
                    mask, last_patch_scores, alpha_t, has_cls=False,
                )
            else:
                mask = apply_budget_hard_mask(
                    mask, last_patch_scores, alpha_t, has_cls=False,
                )
            tokens = tokens * mask.unsqueeze(-1)

        return tokens, layer_avg_masks, mask
