from __future__ import annotations

import torch
import torch.nn.functional as F

STAGES = ("layer1", "layer2", "layer3", "layer4")
STAGE_DIMS = (64, 128, 256, 512)
LOGIT_STATS = 3
FEATURE_DIM = sum(STAGE_DIMS) + LOGIT_STATS


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
        probs = logits.softmax(1)
        top2 = probs.topk(2, dim=1).values
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(1, keepdim=True)
        stats = torch.cat([top2[:, :1], entropy, top2[:, :1] - top2[:, 1:2]], dim=1)
        return torch.cat(pooled + [stats], dim=1)
