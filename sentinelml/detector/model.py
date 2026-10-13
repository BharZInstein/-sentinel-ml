from __future__ import annotations

import torch.nn as nn

from sentinelml.detector.features import FEATURE_DIM


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
