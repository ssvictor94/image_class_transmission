"""
完整 DJSCC 端到端模型（论文 Section III–IV）
"""
import torch
from torch import nn

from adaptive_vit_encoder import AdaptiveViTEncoder
from complex_compression_coder import ComplexCompressionEncoder, ComplexCompressionDecoder
from config import D_MODEL
from utils.channel import awgn_channel
from utils.transmission import (
    compression_ratio_from_transmission,
    count_transmission_tokens,
    gather_active_tokens,
    scatter_tokens,
)
from utils.vit_masked import forward_block_with_token_mask


class FullDJSCCModel(nn.Module):

    def __init__(
        self,
        vit_backbone,
        s=5,
        encoder_split=6,
        r_values=(0.5, 0.25, 0.1),
        d_model=D_MODEL,
    ):
        super().__init__()
        self.d_model = d_model
        self.s = s
        self.encoder_split = encoder_split
        self.r_values = list(r_values)

        self.adaptive_encoder = AdaptiveViTEncoder(
            vit_backbone, s=s, d_model=d_model, encoder_split=encoder_split
        )

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

        self.remaining_blocks = nn.ModuleList(vit_backbone.blocks[encoder_split:])
        self.norm = vit_backbone.norm
        self.head = vit_backbone.head

    def average_rho(self, n_active, r):
        rho = compression_ratio_from_transmission(n_active, r, d_model=self.d_model)
        if isinstance(rho, torch.Tensor):
            return rho
        return torch.tensor(rho, device=n_active.device if isinstance(n_active, torch.Tensor) else "cpu")

    def forward_after_recovery(self, tokens, mask):
        for block in self.remaining_blocks:
            tokens = forward_block_with_token_mask(block, tokens, mask)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])

    def forward_semantic(self, x, alpha, min_keep_ratio=0.0, discretize=True, use_ste=False):
        use_hard = discretize and not self.training and not use_ste
        tokens, layer_masks, mask = self.adaptive_encoder(
            x, alpha,
            min_keep_ratio=min_keep_ratio,
            use_hard_mask=use_hard,
            use_ste=use_ste,
        )
        logits = self.forward_after_recovery(tokens, mask)
        n_active = count_transmission_tokens(mask)
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

        active_tokens, _ = gather_active_tokens(tokens, mask)
        n_active = count_transmission_tokens(mask)

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
        recovered_tokens = scatter_tokens(recovered_active, tokens.shape, mask)
        logits = self.forward_after_recovery(recovered_tokens, mask)

        return logits, layer_masks, n_active

    def get_encoder(self, r):
        return self.compression_encoders[self.r_key_map[r]]

    def get_decoder(self, r):
        return self.compression_decoders[self.r_key_map[r]]

    def freeze_jscc_only(self):
        """消融：仅训 JSCC（非论文默认）。"""
        for p in self.parameters():
            p.requires_grad = False
        for module in (self.compression_encoders, self.compression_decoders):
            for p in module.parameters():
                p.requires_grad = True

    def freeze_djscc_with_server(self):
        """
        对齐官方 freeze_model=Yes:
          训 JSCC + blocks_after；冻边缘 encoder 与 head/norm。
        """
        for p in self.parameters():
            p.requires_grad = False
        for module in (
            self.compression_encoders,
            self.compression_decoders,
            self.remaining_blocks,
        ):
            for p in module.parameters():
                p.requires_grad = True

    def freeze_for_djscc(self):
        self.freeze_djscc_with_server()

    def freeze_semantic(self):
        self.freeze_jscc_only()

    def set_djscc_train_mode(self):
        """Stage 2：边缘 eval；JSCC/server blocks train；head/norm eval。"""
        self.adaptive_encoder.eval()
        self.compression_encoders.train()
        self.compression_decoders.train()
        self.norm.eval()
        self.head.eval()
        server_trainable = any(p.requires_grad for p in self.remaining_blocks.parameters())
        self.remaining_blocks.train() if server_trainable else self.remaining_blocks.eval()

    def unfreeze_semantic(self):
        for p in self.parameters():
            p.requires_grad = True
