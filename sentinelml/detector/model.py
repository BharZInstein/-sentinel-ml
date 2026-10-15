from __future__ import annotations

import torch
import torch.nn as nn

from sentinelml.detector.features import FEATURE_DIM, LOGIT_TOKEN_DIM, STAGE_DIMS, STAGES, TOKEN_LAYOUTS

# Token sources for the attention detector, in token order, with their input widths.
TOKEN_SOURCES = {**dict(zip(STAGES, STAGE_DIMS)), "logits": LOGIT_TOKEN_DIM}


class MLPDetector(nn.Module):
    """Baseline: pooled activations + logit stats -> one logit (> 0 means adversarial)."""

    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        return self.net(feats).squeeze(1)


class AttentionDetector(nn.Module):
    """[CLS] + per-stage tokens (see TOKEN_LAYOUTS) + logit token -> transformer -> one logit.

    hybrid=True also feeds the logit vector (10 logits + 3 stats) straight to the output layer,
    next to the [CLS] output, so confidence does not have to be routed through attention.
    """

    def __init__(self, width: int = 128, layers: int = 2, heads: int = 4,
                 layout: str = "late", hybrid: bool = False):
        super().__init__()
        self.sources = list(TOKEN_SOURCES)
        self.hybrid = hybrid
        self.proj = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
            for name, dim in TOKEN_SOURCES.items()
        })
        self.stage_emb = nn.Parameter(0.02 * torch.randn(len(self.sources), width))
        self.pos_emb = nn.ParameterDict({
            name: nn.Parameter(0.02 * torch.randn(g * g, width))
            for name, g in TOKEN_LAYOUTS[layout].items() if g > 1
        })
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        if hybrid:
            self.logit_norm = nn.LayerNorm(LOGIT_TOKEN_DIM)
        self.head = nn.Linear(width + (LOGIT_TOKEN_DIM if hybrid else 0), 1)

    def tokens(self, feats: dict) -> torch.Tensor:
        parts = []
        for i, name in enumerate(self.sources):
            t = feats[name]
            if t.dim() == 2:
                t = t.unsqueeze(1)
            t = self.proj[name](t) + self.stage_emb[i]
            if name in self.pos_emb:
                t = t + self.pos_emb[name]
            parts.append(t)
        cls = self.cls.expand(parts[0].shape[0], -1, -1)
        return torch.cat([cls] + parts, dim=1)

    def forward(self, feats: dict) -> torch.Tensor:
        h = self.encoder(self.tokens(feats))
        z = self.norm(h[:, 0])
        if self.hybrid:
            z = torch.cat([z, self.logit_norm(feats["logits"])], dim=1)
        return self.head(z).squeeze(1)

    @torch.no_grad()
    def cls_attention(self, feats: dict) -> list:
        """Per encoder layer, how much [CLS] attends to each token (heads averaged): (B, 1 + n_tokens)."""

        x = self.tokens(feats)
        weights = []
        for layer in self.encoder.layers:
            q = layer.norm1(x)
            _, w = layer.self_attn(q, q, q, need_weights=True, average_attn_weights=True)
            weights.append(w[:, 0])
            x = layer(x)
        return weights
