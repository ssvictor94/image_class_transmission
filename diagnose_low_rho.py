"""诊断低 ρ 高准确率：核对 n_active，并做消融。"""
import torch
from utils.data import get_dataloaders
from utils.model_factory import load_trained_model
from utils.transmission import gather_active_tokens, count_transmission_tokens


@torch.no_grad()
def eval_variant(model, loader, alpha, r, snr, device, mode="normal"):
    model.eval()
    correct = total = 0
    n_sum = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if mode == "normal":
            logits, _, n_active = model.forward_full(images, alpha, r, snr)
        elif mode == "random_mask":
            # 用随机 top-k 替代选择器，看是否仍很高
            tokens, _, mask = model.adaptive_encoder(
                images, alpha, use_hard_mask=True,
            )
            B, N = mask.shape
            n_patch = N - 1
            k = max(1, int(round(alpha * n_patch)))
            rand = torch.rand(B, n_patch, device=device)
            hard = torch.zeros_like(mask)
            idx = rand.topk(k, dim=1).indices
            hard.scatter_(1, idx, 1.0)
            hard[:, -1] = 1.0
            tokens = tokens * hard.unsqueeze(-1)
            active, _ = gather_active_tokens(tokens, hard, has_cls=False)
            n_active = count_transmission_tokens(hard, has_cls=False)
            enc = model.get_encoder(r) if hasattr(model, "get_encoder") else model.compression_encoders[model.r_key_map[r]]
            dec = model.compression_decoders[model.r_key_map[r]]
            from utils.channel import awgn_channel
            from utils.transmission import scatter_tokens
            s = enc(active)
            snr_t = torch.full((s.size(0),), snr, device=device)
            y = awgn_channel(s, snr_t, dims=-1)
            rec = scatter_tokens(dec(y), tokens.shape, hard, has_cls=False)
            logits = model.forward_after_recovery(rec, hard)
        elif mode == "zero_channel":
            # 传全零符号：若准确率仍高则服务器在偷用不该有的信息
            tokens, _, mask = model.adaptive_encoder(images, alpha, use_hard_mask=True)
            active, _ = gather_active_tokens(tokens, mask, has_cls=False)
            n_active = count_transmission_tokens(mask, has_cls=False)
            zeros = torch.zeros_like(active)
            from utils.transmission import scatter_tokens
            # 走 encoder 维度
            enc = model.compression_encoders[model.r_key_map[r]]
            dec = model.compression_decoders[model.r_key_map[r]]
            s = enc(zeros)
            rec = scatter_tokens(dec(s * 0), tokens.shape, mask, has_cls=False)
            logits = model.forward_after_recovery(rec, mask)
        else:
            raise ValueError(mode)

        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        n_sum += n_active.float().mean().item() * labels.size(0)
    return correct / total, n_sum / total


def main():
    device = "cuda"
    model = load_trained_model(
        backbone="mambavision", device=device,
    )
    _, loader = get_dataloaders(batch_size=64)
    alpha, r, snr = 0.05, 0.1, 10.0

    for mode in ("normal", "random_mask", "zero_channel"):
        acc, n_act = eval_variant(model, loader, alpha, r, snr, device, mode)
        rho = model.average_rho(torch.tensor(n_act), r).item()
        print(f"{mode:14s}  acc={acc:.4f}  n_active={n_act:.2f}  rho={rho:.6f}")


if __name__ == "__main__":
    main()
