"""
复数域 JSCC 编解码器（论文 Section IV-B, 式(4)–(6)）
====================================================

论文语义 JSCC 管道:
  1. 编码: s = C_E^r(Z) ∈ C^{q}        — 将活跃 token 映射为复数符号
  2. 功率约束: ||s||_2^2 ≤ q            — 式(4)
  3. 信道:   y = s + n,  n~CN(0, σ_n^2) — 式(6) AWGN
  4. 解码:   Ẑ = C_D^r(y)              — 恢复 token 特征

其中:
  r       — 压缩率，每个 d 维 token 映射为 o_r = r·d 个复数符号
  q       — 总符号数 = n_α · o_r（n_α 为活跃 token 数）

官方实现使用 3 层 MLP:
  C_E: 两个独立 MLP 分别输出实部/虚部（BaseRealToComplexNN）
  C_D: 拼接 [Re(y), Im(y)] 后 MLP 恢复（ConcatComplexToRealNN）
"""
import numpy as np
import torch
from torch import nn


def _build_mlp(in_dim, out_dim, n_layers=3):
    """构建 n_layers 层全连接 MLP（与官方 get_layers 一致）。"""
    shapes = [int(x) for x in np.linspace(in_dim, out_dim, n_layers + 1)]
    layers = []
    for i in range(len(shapes) - 1):
        layers.append(nn.Linear(shapes[i], shapes[i + 1]))
        if i < len(shapes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ComplexCompressionEncoder(nn.Module):
    """
    JSCC 编码器 C_E^r（论文 Section IV-B）。

    对每个 d 维 token 特征 h_i:
      s_i = c_R(h_i) + j·c_I(h_i) ∈ C^{o_r},  o_r = r·d

    随后按式(4) 归一化使 ||s||_2^2 = q。
    """

    def __init__(self, in_dim, compression_ratio, n_layers=3):
        super().__init__()
        out_dim = max(int(in_dim * compression_ratio), 1)
        self.out_dim = out_dim
        self.compression_ratio = compression_ratio

        # c_R, c_I: 两个独立 MLP 生成复数符号的实部与虚部
        self.r_net = _build_mlp(in_dim, out_dim, n_layers)
        self.c_net = _build_mlp(in_dim, out_dim, n_layers)

    def forward(self, x, n_symbols=None):
        """
        Args:
            x: [B, n_α, d] 活跃 token 特征
        Returns:
            复数符号，每个 token 做 L2 归一化（官方 BaseRealToComplexNN.normalize）
        """
        real = self.r_net(x)
        imag = self.c_net(x)
        complex_sym = torch.complex(real, imag)
        # 官方: 逐 token 单位范数，再由信道按实际功率加噪
        norm = torch.norm(complex_sym, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        return complex_sym / norm


class ComplexCompressionDecoder(nn.Module):
    """
    JSCC 解码器 C_D^r（论文 Section IV-B）。

    将接收符号 y_i 恢复为 d 维 token 特征:
      concat(Re(y_i), Im(y_i)) → MLP → ĥ_i
    """

    def __init__(self, in_dim, out_dim, n_layers=3):
        super().__init__()
        self.net = _build_mlp(in_dim * 2, out_dim, n_layers)

    def forward(self, complex_sym):
        x = torch.cat([complex_sym.real, complex_sym.imag], dim=-1)
        out = self.net(x)
        # 零填充位置（非活跃 token）解码输出置零
        active = (complex_sym.abs().sum(dim=-1, keepdim=True) > 0).float()
        return out * active
