"""
Lyapunov 在线优化器（论文 Section V, 式(17)(19)）
=================================================

论文将自适应参数选择建模为约束优化:
  max  E[Λ(γ_t)]           — 最大化长期平均准确率
  s.t. E[ρ(γ_t)] ≤ Γ_th   — 平均压缩比不超过阈值

采用 Lyapunov drift-plus-penalty 框架:

  虚拟队列（式(17)）:
    Z_{t+1} = max(0, Z_t + μ · (ρ(γ_t) − Γ_th))

  每时隙参数选择（式(19)）:
    γ_t = argmin_{γ∈C}  { −V·Λ(γ) + Z_t·ρ(γ) }

  V — 准确率与压缩比的权衡系数
  C — 可行参数集 {(α, r)} 的离散网格
  proxy_table — 离线预计算的 (α, r, SNR) → (Λ, ρ) 查表
"""
import numpy as np


class LyapunovOptimizer:
    """
    论文 Section V 的在线 Lyapunov 优化器。

    在运行时根据当前 SNR 和虚拟队列状态 Z，从可行集 C 中选择最优 γ=(α,r)。
    """

    def __init__(self, feasible_set_C, Gamma_th, total_pixels, V=100, mu=1):
        """
        Args:
            feasible_set_C: 可行参数列表 [{'alpha': 0.5, 'r': 0.25}, ...]
            Gamma_th:       平均压缩比约束 Γ_th
            total_pixels:   输入像素数 p（用于名义 ρ 计算，实际使用 proxy 中的实测 ρ）
            V:              Lyapunov 权衡参数（式(19)）
            mu:             虚拟队列步长（式(17)）
        """
        self.C = feasible_set_C
        self.Gamma_th = Gamma_th
        self.total_pixels = total_pixels
        self.V = V
        self.mu = mu
        self.Z = 0
        self.snr_values_sorted = None

    def _find_closest_snr(self, snr_db, proxy_table):
        """SNR 为连续值时，在 proxy 表中找最近离散 SNR。"""
        if self.snr_values_sorted is None:
            all_snrs = set(key[2] for key in proxy_table.keys())
            self.snr_values_sorted = sorted(all_snrs)

        if any(abs(snr_db - s) < 1e-6 for s in self.snr_values_sorted):
            return snr_db

        idx = np.searchsorted(self.snr_values_sorted, snr_db)
        if idx == 0:
            return self.snr_values_sorted[0]
        if idx == len(self.snr_values_sorted):
            return self.snr_values_sorted[-1]
        left = self.snr_values_sorted[idx - 1]
        right = self.snr_values_sorted[idx]
        return left if (snr_db - left) < (right - snr_db) else right

    def select_parameters(self, snr_db, proxy_table):
        """
        根据当前 SNR 选择最优参数 γ（式(19)）。

        Args:
            snr_db:      当前时隙信道 SNR（dB）
            proxy_table: key=(alpha, r, snr), value={'acc': Λ, 'rho': ρ}

        Returns:
            best_gamma: {'alpha': ..., 'r': ...}
        """
        closest_snr = self._find_closest_snr(snr_db, proxy_table)
        best_gamma = None
        best_cost = float("inf")

        for gamma in self.C:
            key = (gamma["alpha"], gamma["r"], closest_snr)
            entry = proxy_table.get(key, {"acc": 0.0, "rho": 1.0})
            acc = entry["acc"] if isinstance(entry, dict) else entry
            rho = entry.get("rho", 1.0) if isinstance(entry, dict) else gamma.get("rho", 1.0)

            # 式(19): cost = −V·Λ + Z·ρ
            cost = -self.V * acc + self.Z * rho
            if cost < best_cost:
                best_cost = cost
                best_gamma = gamma

        # 式(17): 更新虚拟队列 Z
        key = (best_gamma["alpha"], best_gamma["r"], closest_snr)
        entry = proxy_table.get(key, {"rho": 1.0})
        rho_selected = entry.get("rho", 1.0) if isinstance(entry, dict) else 1.0
        self.Z = max(0, self.Z + self.mu * (rho_selected - self.Gamma_th))

        return best_gamma
