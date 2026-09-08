"""Device selection and global seeding.

Prefers Apple-silicon MPS, falls back to CUDA, then CPU. Seeding covers Python,
NumPy and torch so the whole pipeline is reproducible.
"""
from __future__ import annotations

import random

import numpy as np
import torch

import config


def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int = config.SEED) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = get_device()
