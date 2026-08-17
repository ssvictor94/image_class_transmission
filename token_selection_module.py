"""
自适应 Token 选择模块（论文 Section IV-A, Algorithm 1）
======================================================

论文在每个自适应 Transformer block 前插入选择模块 S_l，根据预算 α
决定哪些 patch token 参与后续计算与传输。

核心网络（与官方 AdaptiveBlock 一致）:
  f_l(H) : R^{N×d} → R^{N×1}   — 为每个 token 预测重要性分数（门控）
  f_h(b) : R^{d}    → R         — 由 budget token 预测阈值

训练阶段 soft mask（官方实现）:
  m_l = ReLU( σ(f_l(h_i)) − σ(f_h(b_α)) )     ... 对每个 patch token

推理阶段:
  中间层保持 soft 累积丢弃；出口按 α 做 top-k 硬预算
  （纯阈值二值化会把微小正 soft 全变成 1，导致 n_α 与 α 脱钩、Fig 成直线）

累积掩码（多层选择）:
  m ← m ⊙ m_prev                              ... 已丢弃 token 不会恢复

CLS token 与 budget token 始终保留（不参与丢弃）。
"""
import torch
from torch import nn


class TokenSelectionModule(nn.Module):
    """
    单层 token 选择器 S_l，对应论文 Algorithm 1 中的 (l_t, l_g) 角色。

    在官方代码中:
      f_l = Linear(d,1) + Sigmoid   （重要性）
      f_h = Linear(d,1) + Sigmoid   （阈值，输入 budget token）
    """

    def __init__(self, d_model, has_cls=True):
        super().__init__()
        self.has_cls = has_cls

        # f_l: token 重要性门控网络 l_g(H) 的可学习近似
        self.fl = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        # f_h: 阈值网络 l_t(b)，输入 budget token b_α
        self.fh = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

        # 初始化偏置使训练初期多数 token 被保留（官方: fl.bias~N(5,0.1), fh.bias~N(-5,0.1)）
        self.fl[0].bias.data.normal_(5, 0.1)
        self.fh[0].bias.data.normal_(-5, 0.1)

    def _patch_slice(self, tokens):
        """ViT: [CLS, patches, budget]；Mamba: [patches, budget]。"""
        if self.has_cls:
            return tokens[:, 1:-1]
        return tokens[:, :-1]

    def forward(self, tokens, budget_token, prev_mask, min_keep_ratio=0.0):
        """
        Args:
            tokens:       [B, N, D] 含 patches + budget（可选 CLS）
            budget_token: [B, D]    由 α 插值得到的 budget embedding b_α
            prev_mask:    [B, N]    上一层累积掩码
            min_keep_ratio: 训练早期保底保留比例（稳定训练技巧，非论文核心）

        Returns:
            new_mask: [B, N] soft/hard 掩码
        """
        B, N, D = tokens.shape
        patches = self._patch_slice(tokens)
        n_patch = patches.size(1)

        if n_patch > 0:
            patch_scores = self.fl(patches)
            th = self.fh(budget_token).unsqueeze(1)
            patch_mask = torch.relu(patch_scores - th).squeeze(-1)
            tail_one = torch.ones(B, 1, device=tokens.device, dtype=patch_mask.dtype)
            if self.has_cls:
                cls_one = torch.ones(B, 1, device=tokens.device, dtype=patch_mask.dtype)
                new_mask = torch.cat([cls_one, patch_mask, tail_one], dim=1)
            else:
                new_mask = torch.cat([patch_mask, tail_one], dim=1)
        else:
            new_mask = torch.ones(B, N, device=tokens.device)

        if prev_mask is not None:
            new_mask = new_mask * prev_mask

        if self.training and min_keep_ratio > 0 and n_patch > 0:
            min_keep = max(1, int(n_patch * min_keep_ratio))
            patch_start = 1 if self.has_cls else 0
            for b in range(B):
                n_active = (new_mask[b, patch_start:-1] > 0).sum().item()
                if n_active < min_keep:
                    scores = self.fl(patches[b]).squeeze(-1)
                    inactive = new_mask[b, patch_start:-1] <= 0
                    candidates = scores.clone()
                    candidates[~inactive] = -1e9
                    k = min_keep - int(n_active)
                    top_idx = torch.topk(candidates, k).indices
                    new_mask[b, patch_start + top_idx] = 1.0

        return new_mask
