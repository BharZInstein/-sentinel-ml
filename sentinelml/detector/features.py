from __future__ import annotations

import torch
import torch.nn.functional as F

import config

STAGES = ("layer1", "layer2", "layer3", "layer4")
STAGE_DIMS = (64, 128, 256, 512)
LOGIT_STATS = 3
FEATURE_DIM = sum(STAGE_DIMS) + LOGIT_STATS

# Attention detector token layouts: stage -> grid side. 1 = pooled to a single token,
# g > 1 = average-pooled to a g x g grid, one token per cell.
#   late:  layer1/layer2 pooled, layer3 (8x8) and layer4 (4x4) at native resolution -> 83 tokens
#   early: layer1/layer2/layer3 on 8x8 grids, layer4 pooled                          -> 194 tokens
TOKEN_LAYOUTS = {
    "late": {"layer1": 1, "layer2": 1, "layer3": 8, "layer4": 4},
    "early": {"layer1": 8, "layer2": 8, "layer3": 8, "layer4": 1},
}
LOGIT_TOKEN_DIM = config.NUM_CLASSES + LOGIT_STATS


def logit_stats(logits: torch.Tensor) -> torch.Tensor:
    """Max softmax probability, entropy, and top-1 minus top-2 margin."""

    probs = logits.softmax(1)
    top2 = probs.topk(2, dim=1).values
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(1, keepdim=True)
    return torch.cat([top2[:, :1], entropy, top2[:, :1] - top2[:, 1:2]], dim=1)


class FeatureExtractor:
    """Runs the frozen target and returns one pooled vector per stage plus logit statistics.

    Forward hooks capture each stage's output; the graph is left differentiable so an
    adaptive attacker can later backpropagate through target + detector.
    """

    def __init__(self, target):
        self.target = target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self._captured = {}
        for name in STAGES:
            getattr(self.target, name).register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            self._captured[name] = output
        return hook

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.target(x)
        pooled = [F.adaptive_avg_pool2d(self._captured[n], 1).flatten(1) for n in STAGES]
        return torch.cat(pooled + [logit_stats(logits)], dim=1)


class TokenExtractor(FeatureExtractor):
    """Same hooks, but returns per-source token inputs for the attention detector.

    Pooled stages -> (B, C); grid stages -> (B, g*g, C), one row per grid cell;
    logits -> (B, 10 logits + 3 stats). See TOKEN_LAYOUTS.
    """

    def __init__(self, target, layout: str = "late"):
        super().__init__(target)
        self.grid = TOKEN_LAYOUTS[layout]

    def __call__(self, x: torch.Tensor) -> dict:
        logits = self.target(x)
        out = {}
        for n, g in self.grid.items():
            pooled = F.adaptive_avg_pool2d(self._captured[n], g)
            out[n] = pooled.flatten(1) if g == 1 else pooled.flatten(2).transpose(1, 2)
        out["logits"] = torch.cat([logits, logit_stats(logits)], dim=1)
        return out
