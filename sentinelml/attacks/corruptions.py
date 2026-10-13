from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from torchvision.io import decode_jpeg, encode_jpeg

import config

# Benign, CIFAR-10-C style distortions. These are NOT attacks: they are saved as clean
# negatives so the detector learns not to flag ordinary low-quality images.


def gaussian_noise(x, sigma: float = 0.06):
    return (x + sigma * torch.randn_like(x)).clamp(0, 1)


def gaussian_blur(x, sigma: float = 1.0):
    return TF.gaussian_blur(x, kernel_size=3, sigma=sigma)


def jpeg(x, quality: int = 30):

    out = []
    for img in x:
        encoded = encode_jpeg((img * 255).round().to(torch.uint8).cpu(), quality=quality)
        out.append(decode_jpeg(encoded).float() / 255)
    return torch.stack(out).to(x.device)


def brightness(x, factor: float = 1.3):
    return TF.adjust_brightness(x, factor).clamp(0, 1)


def contrast(x, factor: float = 0.6):
    return TF.adjust_contrast(x, factor).clamp(0, 1)


CORRUPTION_FNS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur": gaussian_blur,
    "jpeg": jpeg,
    "brightness": brightness,
    "contrast": contrast,
}


def corrupt_pool(images, labels, names=config.CORRUPTIONS) -> dict:
    """Split the pool evenly across corruption types; returns a cache dict like generate_family's."""

    chunks = torch.chunk(torch.arange(images.shape[0]), len(names))
    corrupted, kinds = [], []
    for name, idx in zip(names, chunks):
        corrupted.append(CORRUPTION_FNS[name](images[idx]))
        kinds += [name] * len(idx)
    return {
        "family": "corruptions",
        "images": torch.cat(corrupted),
        "labels": labels[torch.cat(chunks)],
        "kind": kinds,
        "meta": {"count": int(images.shape[0]), "types": list(names)},
    }
