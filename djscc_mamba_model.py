"""
基于 MambaVision 的 DJSCC（对齐论文 Section III–IV，仅替换骨干）
================================================================

边缘: Stage0/1 + Stage2 全部 block + 渐进式 S_l + 掩码隔离
信道: 复数 JSCC + AWGN（14×14 token）
服务器: Stage2 downsample → Stage3（掩码）→ 活跃位置均值池化 → head

切分按 MambaVision 阶段边界，不强制对齐 ViT 的 6 层。
"""
import torch
from torch import nn

from adaptive_mamba_encoder import AdaptiveMambaEncoder, run_one_stage2_block
from complex_compression_coder import ComplexCompressionEncoder, ComplexCompressionDecoder
from config import MAMBA_ADAPTIVE_LAYERS, MAMBA_D_MODEL, MAMBA_ENCODER_SPLIT
from utils.channel import awgn_channel
from utils.mamba_masked import downsample_token_mask
from utils.transmission import (
    compression_ratio_from_transmission,
    count_transmission_tokens,
    gather_active_tokens,
    scatter_tokens,
)


class FullDJSCCMambaModel(nn.Module):

    def __init__(
        self,
        mamba_backbone,
        s=MAMBA_ADAPTIVE_LAYERS,
        encoder_split=MAMBA_ENCODER_SPLIT,
        r_values=(0.5, 0.25, 0.1),
        d_model=MAMBA_D_MODEL,
    ):
        super().__init__()
        self.d_model = d_model
        self.has_cls = False
        self.s = s
        self.r_values = list(r_values)
        self.backbone_type = "mambavision"

        self.adaptive_encoder = AdaptiveMambaEncoder(
            mamba_backbone,
            s=s,
            d_model=d_model,
            encoder_split=encoder_split,
        )
        self.encoder_split = self.adaptive_encoder.encoder_split

        # MambaVision Stage2 特征幅度极大（token L2 常达数百），而 ViT 经
        # LayerNorm 后幅度平稳。不归一化时 3 层 JSCC MLP 几乎学不会重建。
        self.pre_jscc_norm = nn.LayerNorm(d_model)

        self.r_key_map = {}
        self.compression_encoders = nn.ModuleDict()
        self.compression_decoders = nn.ModuleDict()
        for r in r_values:
            key = str(r).replace(".", "_")
            self.r_key_map[r] = key
            self.compression_encoders[key] = ComplexCompressionEncoder(d_model, r)
            self.compression_decoders[key] = ComplexCompressionDecoder(
                max(int(d_model * r), 1), d_model
            )

    def average_rho(self, n_active, r):
        rho = compression_ratio_from_transmission(
            n_active, r, d_model=self.d_model,
        )
        if isinstance(rho, torch.Tensor):
            return rho
        return torch.tensor(
            rho,
            device=n_active.device if isinstance(n_active, torch.Tensor) else "cpu",
        )

    def _iter_server_blocks(self):
        """
        服务器后续计算块（对齐官方 blocks_after）。
        不含 norm/head：官方 freeze_model=Yes 时 head/norm 不进优化器。
        """
        enc = self.adaptive_encoder
        for blk in enc.level2.blocks[self.encoder_split:]:
            yield blk
        if enc.stage2_downsample is not None:
            yield enc.stage2_downsample
        yield enc.level3

    def _iter_server_modules(self):
        """服务器全部模块（含 head），仅用于模式切换等。"""
        yield from self._iter_server_blocks()
        enc = self.adaptive_encoder
        yield enc.norm
        yield enc.head

    def tokens_to_feature_map(self, tokens, mask):
        patch_tokens = tokens[:, :-1]
        patch_mask = mask[:, :-1]
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        if H * W != N:
            spatial = getattr(self.adaptive_encoder, "_last_spatial", None)
            if spatial is None:
                raise ValueError(f"无法将 N={N} reshape 为特征图")
            H, W = spatial
        feat = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        return feat * patch_mask.view(B, 1, H, W), (H, W)

    def forward_after_recovery(self, tokens, mask):
        """
        服务器侧（层次切分）:
          若 Stage2 有剩余 block → 掩码前向
          → downsample + mask 下采样
          → Stage3 掩码前向 → 活跃位置均值池化 → head
        """
        enc = self.adaptive_encoder
        feat, (H, W) = self.tokens_to_feature_map(tokens, mask)
        patch_mask = mask[:, :-1]
        win2 = enc.stage2_window_size

        for block in enc.level2.blocks[self.encoder_split:]:
            feat = run_one_stage2_block(block, feat, patch_mask, win2)
            feat = feat * patch_mask.view(feat.size(0), 1, H, W)

        if enc.stage2_downsample is not None:
            feat = enc.stage2_downsample(feat)
            patch_mask_ds = downsample_token_mask(patch_mask, H, W)
            _, _, H2, W2 = feat.shape
            feat = feat * patch_mask_ds.view(feat.size(0), 1, H2, W2)
        else:
            patch_mask_ds = patch_mask
            H2, W2 = H, W

        win3 = enc.level3.window_size
        for block in enc.level3.blocks:
            feat = run_one_stage2_block(block, feat, patch_mask_ds, win3)
            feat = feat * patch_mask_ds.view(feat.size(0), 1, H2, W2)

        feat = enc.norm(feat)
        mask_map = patch_mask_ds.view(feat.size(0), 1, H2, W2)
        feat = feat * mask_map
        denom = mask_map.sum(dim=(2, 3)).clamp(min=1e-6)
        feat = feat.sum(dim=(2, 3)) / denom
        return enc.head(feat)

    def forward_semantic(self, x, alpha, min_keep_ratio=0.0, discretize=True, use_ste=False):
        use_hard = discretize and not self.training and not use_ste
        tokens, layer_masks, mask = self.adaptive_encoder(
            x, alpha,
            min_keep_ratio=min_keep_ratio,
            use_hard_mask=use_hard,
            use_ste=use_ste,
        )
        logits = self.forward_after_recovery(tokens, mask)
        n_active = count_transmission_tokens(mask, has_cls=False)
        return logits, layer_masks, n_active

    def forward_full(
        self, x, alpha, r, snr_db,
        min_keep_ratio=0.0, discretize=True, use_ste=False,
    ):
        use_hard = discretize and not self.training and not use_ste
        tokens, layer_masks, mask = self.adaptive_encoder(
            x, alpha,
            min_keep_ratio=min_keep_ratio,
            use_hard_mask=use_hard,
            use_ste=use_ste,
        )

        active_tokens, _ = gather_active_tokens(tokens, mask, has_cls=False)
        n_active = count_transmission_tokens(mask, has_cls=False)
        active_tokens = self.pre_jscc_norm(active_tokens)

        r_key = self.r_key_map[r]
        enc = self.compression_encoders[r_key]
        dec = self.compression_decoders[r_key]

        complex_s = enc(active_tokens)

        if not isinstance(snr_db, torch.Tensor):
            snr_db = torch.tensor(
                snr_db, device=complex_s.device,
                dtype=complex_s.real.dtype,
            )
        if snr_db.dim() == 0:
            snr_db = snr_db.expand(complex_s.size(0))
        received = awgn_channel(complex_s, snr_db, dims=-1)

        recovered_active = dec(received)
        recovered_tokens = scatter_tokens(
            recovered_active, tokens.shape, mask, has_cls=False,
        )
        logits = self.forward_after_recovery(recovered_tokens, mask)
        return logits, layer_masks, n_active

    def _unfreeze_jscc_modules(self):
        for module in (
            self.pre_jscc_norm,
            self.compression_encoders,
            self.compression_decoders,
        ):
            for p in module.parameters():
                p.requires_grad = True

    def freeze_jscc_only(self):
        """消融：仅训 JSCC（非论文默认）。"""
        for p in self.parameters():
            p.requires_grad = False
        self._unfreeze_jscc_modules()

    def freeze_djscc_with_server(self):
        """
        对齐官方 freeze_model=Yes:
          训 JSCC + blocks_after；冻边缘 encoder 与 head/norm。
        """
        for p in self.parameters():
            p.requires_grad = False
        self._unfreeze_jscc_modules()
        for module in self._iter_server_blocks():
            for p in module.parameters():
                p.requires_grad = True

    def freeze_for_djscc(self):
        self.freeze_djscc_with_server()

    def freeze_semantic(self):
        self.freeze_jscc_only()

    def set_djscc_train_mode(self):
        self.adaptive_encoder.eval()
        self.pre_jscc_norm.train()
        self.compression_encoders.train()
        self.compression_decoders.train()
        # head/norm 保持 eval（论文不训）
        self.adaptive_encoder.norm.eval()
        self.adaptive_encoder.head.eval()
        for m in self._iter_server_blocks():
            m.train() if any(p.requires_grad for p in m.parameters()) else m.eval()

    def unfreeze_semantic(self):
        for p in self.parameters():
            p.requires_grad = True
