from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import config


class PatchAttack:
    """Unbounded square patch at a random location per image, optimised with signed gradient ascent."""

    def __init__(self, model, size: int = config.PATCH_SIZE, steps: int = config.PATCH_STEPS,
                 alpha: float = config.PATCH_ALPHA, seed: int = config.SEED):
        self.model = model
        self.size = size
        self.steps = steps
        self.alpha = alpha
        self.generator = torch.Generator().manual_seed(seed)

    def _masks(self, images):

        b, _, h, w = images.shape
        rows = torch.randint(0, h - self.size + 1, (b,), generator=self.generator)
        cols = torch.randint(0, w - self.size + 1, (b,), generator=self.generator)
        mask = torch.zeros(b, 1, h, w)
        for i in range(b):
            mask[i, :, rows[i]:rows[i] + self.size, cols[i]:cols[i] + self.size] = 1.0
        return mask.to(images.device)

    def __call__(self, images, labels):

        images = images.detach()
        mask = self._masks(images)
        patch = torch.rand(images.shape, generator=self.generator).to(images.device)
        for _ in range(self.steps):
            patch.requires_grad_(True)
            adv = images * (1 - mask) + patch * mask
            loss = F.cross_entropy(self.model(adv), labels)
            grad, = torch.autograd.grad(loss, patch)
            patch = (patch.detach() + self.alpha * grad.sign()).clamp(0, 1)
        return (images * (1 - mask) + patch * mask).detach()


class SpatialAttack:
    """Worst-case rotation + translation from a grid search (Engstrom et al., 2019). No additive noise."""

    def __init__(self, model, max_rot: float = config.SPATIAL_MAX_ROT, n_rot: int = config.SPATIAL_N_ROT,
                 max_shift: int = config.SPATIAL_MAX_SHIFT):
        self.model = model
        self.angles = torch.linspace(-max_rot, max_rot, n_rot).tolist()
        self.shifts = list(range(-max_shift, max_shift + 1))

    @staticmethod
    def transform(images, angle_deg: float, dx: int, dy: int):

        b, _, h, w = images.shape
        a = math.radians(angle_deg)
        theta = torch.tensor([[math.cos(a), -math.sin(a), 2 * dx / w],
                              [math.sin(a), math.cos(a), 2 * dy / h]],
                             dtype=images.dtype, device=images.device)
        grid = F.affine_grid(theta.expand(b, 2, 3), images.shape, align_corners=False)
        return F.grid_sample(images, grid, padding_mode="zeros", align_corners=False)

    @torch.no_grad()
    def __call__(self, images, labels):

        best = images.clone()
        best_loss = torch.full((images.shape[0],), -float("inf"), device=images.device)
        for angle in self.angles:
            for dx in self.shifts:
                for dy in self.shifts:
                    cand = self.transform(images, angle, dx, dy)
                    loss = F.cross_entropy(self.model(cand), labels, reduction="none")
                    better = loss > best_loss
                    best[better] = cand[better]
                    best_loss = torch.where(better, loss, best_loss)
        return best
