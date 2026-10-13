from __future__ import annotations

import torch

import config


def _pool_index(clean_pool: torch.Tensor) -> dict:
    """Map each clean image's bytes to its position in clean.pt (the global image id)."""

    return {clean_pool[i].numpy().tobytes(): i for i in range(clean_pool.shape[0])}


def split_image_ids(n: int, test_split: float = config.TEST_SPLIT, seed: int = config.SEED):
    """Split by image id so a clean image and its adversarial versions never straddle the split."""

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_test = int(n * test_split)
    return perm[n_test:], perm[:n_test]


def load_caches():
    """Returns (clean_pool, {family: (adv, image_ids)}, corruptions) from the cache files that exist."""

    clean_pool = torch.load(config.ADV_CACHE_DIR / "clean.pt", weights_only=True)["clean"]
    index = _pool_index(clean_pool)

    families = {}
    for name in config.ALL_ATTACKS:
        path = config.ADV_CACHE_DIR / f"{name}.pt"
        if not path.exists():
            continue
        cache = torch.load(path, weights_only=True)
        ids = torch.tensor([index[c.numpy().tobytes()] for c in cache["clean"]])
        families[name] = (cache["adv"], ids)

    corr_path = config.ADV_CACHE_DIR / "corruptions.pt"
    corruptions = torch.load(corr_path, weights_only=True)["images"] if corr_path.exists() else None
    return clean_pool, families, corruptions
