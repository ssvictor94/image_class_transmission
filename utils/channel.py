"""AWGN 信道（对齐官方 comm.channel.GaussianNoiseChannel）。"""
import torch


def awgn_channel(x, snr_db, dims=-1):
    """
    式(6) AWGN: y = s + n

    噪声功率由实际信号功率决定（官方实现）:
      P_signal = ||s||_2 / size(dims)
      σ_n = sqrt(P_signal / SNR_linear)
    """
    if not isinstance(snr_db, torch.Tensor):
        dtype = x.real.dtype if x.is_complex() else x.dtype
        snr_db = torch.tensor(snr_db, device=x.device, dtype=dtype)
    while snr_db.dim() < x.dim():
        snr_db = snr_db.unsqueeze(-1)

    signal_power = torch.linalg.vector_norm(x, ord=2, dim=dims, keepdim=True)
    if isinstance(dims, int):
        size = x.size(dims)
    else:
        size = 1
        for d in dims:
            size *= x.size(d)
    signal_power = signal_power / max(size, 1)

    noise_power = signal_power / (10 ** (snr_db / 10))
    std = torch.sqrt(noise_power.clamp(min=1e-12))

    if x.is_complex():
        noise = torch.complex(
            torch.randn_like(x.real) * std,
            torch.randn_like(x.imag) * std,
        )
    else:
        noise = torch.randn_like(x) * std
    return x + noise
