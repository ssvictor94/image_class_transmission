"""论文式(9) 预算约束损失（对齐官方 methods.proposal.AdaptiveTokenLoss）。"""
import torch
from torch.nn import functional as F

from config import LAMBDA_R, LAMBDA_S, MARGIN


def adaptive_token_loss(layer_masks, alpha, margin=MARGIN):
    """
    式(9) / 官方 AdaptiveTokenLoss (margin 型):

      B_s = ReLU( |m̄_s − α| − margin )
      R   = mean_l ReLU( |m̄_l − α| − margin )

      L = λ_s·B_s + λ_r·R
    """
    if not layer_masks:
        return torch.tensor(0.0, device=alpha.device)

    alpha = alpha.view(-1)
    last = layer_masks[-1]
    output_loss = F.relu(torch.abs(last - alpha) - margin).mean()

    if len(layer_masks) > 1:
        inner = torch.stack(layer_masks[:-1], dim=1)
        inner_loss = F.relu(torch.abs(inner - alpha.unsqueeze(1)) - margin).mean()
    else:
        inner_loss = torch.tensor(0.0, device=alpha.device)

    return LAMBDA_S * output_loss + LAMBDA_R * inner_loss
