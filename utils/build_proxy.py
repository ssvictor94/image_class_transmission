"""
Lyapunov 优化用精度代理表构建（论文 Section V）
==============================================

离线遍历可行参数集 C 与 SNR 网格，预计算:
  proxy_table[(α, r, SNR)] = {'acc': Λ, 'rho': ρ}

供 LyapunovOptimizer.select_parameters() 在线查表（式(19)）。
ρ 使用评估时的实测压缩比，而非设计值 α·N·r·d。
"""
from utils.evaluation import evaluate_model


def build_accuracy_proxy(model, val_loader, feasible_set_C, snr_values, device="cuda"):
    """
    构建 (α, r, SNR) → (accuracy, rho) 代理表。

    Args:
        model:           训练完成的 FullDJSCCModel
        val_loader:      验证集 DataLoader
        feasible_set_C:  可行参数列表 [{'alpha': 0.5, 'r': 0.25}, ...]
        snr_values:      SNR 网格（dB）

    Returns:
        proxy_table: dict, key=(alpha, r, snr), value={'acc', 'rho'}
    """
    proxy_table = {}
    total = len(feasible_set_C) * len(snr_values)
    current = 0

    print(f"开始构建代理表，共 {total} 个配置...")

    for gamma in feasible_set_C:
        for snr in snr_values:
            acc, rho = evaluate_model(model, val_loader, gamma, snr, device)
            proxy_table[(gamma["alpha"], gamma["r"], snr)] = {"acc": acc, "rho": rho}

            current += 1
            if current % 10 == 0:
                print(f"进度: {current}/{total}, acc={acc:.4f}, rho={rho:.6f}")

    print("代理表构建完成！")
    return proxy_table
