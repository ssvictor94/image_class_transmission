"""
自适应语义 ViT 编码器（论文 Section IV-A, 式(1)–(3)）
====================================================

论文边缘端编码器可写为:
  Z = (S_s ∘ B_s ∘ ... ∘ S_1 ∘ B_1)(E(x), b_α)

其中:
  E(·)      — Patch Embedding + 位置编码
  B_l(·)    — 第 l 个 Transformer block
  S_l(·)    — 第 l 个 token 选择模块
  b_α       — 由预算 α 生成的 budget token

Budget token 插值（Section IV-A）:
  b_α = α · b_h + (1 − α) · b_l

  b_h, b_l 为可学习参数，α 越大表示允许保留/传输更多 token。

本实现:
  - 在前 ENCODER_SPLIT=6 个 block 中，block 1..5 前插入 S_l
  - 被丢弃 token 通过 masked self-attention 完全隔离（Section IV-A）
  - 推理时对 mask 二值化（式(12) 思想），仅传输活跃 token
"""
import torch
from torch import nn

from token_selection_module import TokenSelectionModule
from utils.transmission import EVAL_MASK_THRESHOLD, apply_budget_hard_mask, ste_topk_mask
from utils.vit_masked import forward_block_with_token_mask


class AdaptiveViTEncoder(nn.Module):
    """
    论文 Figure 2 左侧「Adaptive Semantic Encoder」的实现。

    输出:
      tokens:         编码器末端的 token 序列（含 budget token）
      layer_avg_masks: 各自适应层的平均 patch 掩码（用于式(9) 预算损失）
      mask:           最终 token 级掩码（用于 JSCC gather 与解码器 masked attention）
    """

    def __init__(self, vit_backbone, s=5, d_model=192, encoder_split=6):
        super().__init__()
        self.vit = vit_backbone
        self.s = s
        self.encoder_split = encoder_split
        self.d_model = d_model

        # 可学习 budget token 端点 b_l（低预算）与 b_h（高预算）
        self.b_l = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.b_h = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # s 个 token 选择模块 {S_1, ..., S_s}
        self.selection_modules = nn.ModuleList([
            TokenSelectionModule(d_model=d_model) for _ in range(s)
        ])

    def _budget_token(self, alpha, batch_size, device):
        """
        生成 budget embedding b_α（论文 Section IV-A）:
          b_α = α · b_h + (1 − α) · b_l
        """
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, device=device).expand(batch_size)
        elif alpha.dim() == 0:
            alpha = alpha.expand(batch_size)
        elif alpha.size(0) == 1 and batch_size > 1:
            alpha = alpha.expand(batch_size)

        alpha_exp = alpha.view(-1, 1, 1)
        return alpha_exp * self.b_h + (1 - alpha_exp) * self.b_l

    def forward(self, x, alpha, min_keep_ratio=0.0, use_hard_mask=None, use_ste=False):
        """
        Args:
            use_hard_mask: True → 推理硬离散化；False → soft mask
            use_ste:       True → STE 硬离散化（Stage 2 训练后期）
        """
        if use_hard_mask is None:
            use_hard_mask = not self.training and not use_ste

        if not isinstance(alpha, torch.Tensor):
            alpha_t = torch.tensor(alpha, device=x.device)
        else:
            alpha_t = alpha
        B = x.shape[0]
        budget = self._budget_token(alpha_t, B, x.device)

        # --- Patch Embedding E(x) ---
        tokens = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)   # [CLS; patch_1; ...; patch_N]
        tokens = tokens + self.vit.pos_embed

        # 将 budget token 拼接到序列末尾（官方 SemanticVit 做法）
        tokens = torch.cat([tokens, budget], dim=1)

        mask = torch.ones(B, tokens.size(1), device=tokens.device)
        layer_avg_masks = []
        last_patch_scores = None

        # --- 前 ENCODER_SPLIT 层: S_l → masked B_l ---
        for layer_idx, block in enumerate(self.vit.blocks):
            if layer_idx >= self.encoder_split:
                break

            # block 1..s 对应 selection_modules[0..s-1]
            if layer_idx > 0 and (layer_idx - 1) < len(self.selection_modules):
                sel_idx = layer_idx - 1
                new_mask = self.selection_modules[sel_idx](
                    tokens, budget.squeeze(1), mask, min_keep_ratio
                )
                last_patch_scores = self.selection_modules[sel_idx].fl(tokens[:, 1:-1]).squeeze(-1)

                if use_ste:
                    from utils.transmission import ste_topk_mask
                    new_mask = ste_topk_mask(new_mask, last_patch_scores, alpha_t)
                elif use_hard_mask:
                    from utils.transmission import apply_budget_hard_mask
                    new_mask = apply_budget_hard_mask(new_mask, last_patch_scores, alpha_t)
                mask = new_mask

            # 被丢弃 token 置零后再过 masked Transformer block
            tokens = tokens * mask.unsqueeze(-1)
            tokens = forward_block_with_token_mask(block, tokens, mask)

            if layer_idx > 0 and (layer_idx - 1) < len(self.selection_modules):
                tokens = tokens * mask.unsqueeze(-1)
                # 记录该层 patch 平均掩码 m̄_l（用于式(9) 内层/外层预算损失）
                layer_avg_masks.append(mask[:, 1:-1].mean(dim=1))

        if use_hard_mask and last_patch_scores is not None:
            from utils.transmission import apply_budget_hard_mask
            mask = apply_budget_hard_mask(mask, last_patch_scores, alpha_t)
            tokens = tokens * mask.unsqueeze(-1)
        elif use_ste and last_patch_scores is not None:
            from utils.transmission import ste_topk_mask
            mask = ste_topk_mask(mask, last_patch_scores, alpha_t)
            tokens = tokens * mask.unsqueeze(-1)

        return tokens, layer_avg_masks, mask
