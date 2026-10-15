from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import config
from sentinelml.device import DEVICE, seed_everything
from sentinelml.target.train_target import load_target
from sentinelml.detector.data import load_caches, split_image_ids
from sentinelml.detector.features import TOKEN_LAYOUTS, TokenExtractor
from sentinelml.detector.model import AttentionDetector

# Model C: early-layer 8x8 grids + direct logit path.
LAYOUT = "early"
WEIGHTS = config.ARTIFACTS_DIR / "detector_attention_early_hybrid.pt"
OUT_PNG = config.ARTIFACTS_DIR / "attention_c.png"
EXAMPLES = ("patch", "pgd_8")


def token_groups():
    """(label, slice into the [CLS]-attention vector) per token source; index 0 is [CLS] itself."""

    groups, start = [], 1
    for name, g in TOKEN_LAYOUTS[LAYOUT].items():
        n = g * g
        groups.append((f"{name} ({g}x{g})" if g > 1 else f"{name} (pooled)", slice(start, start + n)))
        start += n
    groups.append(("logits", slice(start, start + 1)))
    return groups


@torch.no_grad()
def cls_attention(extractor, detector, images, batch_size: int = 256):
    """[CLS] attention in the last encoder layer, (N, 1 + n_tokens)."""

    out = []
    for i in range(0, images.shape[0], batch_size):
        feats = extractor(images[i:i + batch_size].to(DEVICE))
        out.append(detector.cls_attention(feats)[-1].cpu())
    return torch.cat(out)


def grid_map(attn_row):
    """Sum [CLS] attention over the spatial grids (layer1-3 share one 8x8 grid) -> 32x32 map."""

    grids = [attn_row[s].view(8, 8) for label, s in token_groups() if "8x8" in label]
    m = torch.stack(grids).sum(0)[None, None]
    return F.interpolate(m, size=32, mode="bilinear", align_corners=False)[0, 0].numpy()


def main():
    seed_everything()
    extractor = TokenExtractor(load_target().to(DEVICE), layout=LAYOUT)
    detector = AttentionDetector(layout=LAYOUT, hybrid=True).to(DEVICE)
    detector.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE, weights_only=True))
    detector.eval()

    clean_pool, families, _ = load_caches()
    _, test_ids = split_image_ids(clean_pool.shape[0])
    test_set = set(test_ids.tolist())

    # Panel 1: share of [CLS] attention per token group, averaged over held-out images.
    rows, labels, examples = [], [], {}
    attn = cls_attention(extractor, detector, clean_pool[test_ids])
    rows.append(attn)
    labels.append("clean")
    examples["clean"] = (clean_pool[test_ids[0]], clean_pool[test_ids[0]], attn[0])
    for name, (adv, ids) in families.items():
        keep = torch.tensor([int(i) in test_set for i in ids])
        if keep.sum() == 0:
            continue
        attn = cls_attention(extractor, detector, adv[keep])
        rows.append(attn)
        labels.append(name if name in config.TRAIN_ATTACKS else f"{name} *")
        if name in EXAMPLES:
            first = int(torch.nonzero(keep)[0])
            examples[name] = (adv[first], clean_pool[int(ids[first])], attn[0])

    groups = token_groups()
    share = np.array([[a[:, s].sum(1).mean().item() for _, s in groups] for a in rows])

    fig = plt.figure(figsize=(11, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(share, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(groups)), [g for g, _ in groups])
    ax.set_yticks(range(len(labels)), labels)
    for r in range(share.shape[0]):
        for c in range(share.shape[1]):
            ax.text(c, r, f"{share[r, c]:.2f}", ha="center", va="center",
                    color="white" if share[r, c] < share.max() * 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, label="share of [CLS] attention (last layer)")
    ax.set_title("Model C: where [CLS] attends, per family (held-out images; * = held-out family)")

    # Panel 2: image, what the attack changed, and where [CLS] attends spatially.
    sub = gs[1].subgridspec(len(examples), 3, wspace=0.15, hspace=0.3)
    for r, (name, (img, clean, a)) in enumerate(examples.items()):
        diff = (img - clean).abs().sum(0).numpy()
        panels = [(img.permute(1, 2, 0).numpy(), None, f"{name}: image"),
                  (diff, "magma", "|adversarial - clean|"),
                  (grid_map(a), "viridis", "[CLS] attention, layer1-3 grids")]
        for c, (data, cmap, title) in enumerate(panels):
            ax = fig.add_subplot(sub[r, c])
            ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.text(0.5, 0.01, "Attention shows where the detector looks, not a full explanation of why it decided.",
             ha="center", fontsize=9, style="italic")
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"[attention] saved figure -> {OUT_PNG}")
    for label, row in zip(labels, share):
        print(f"{label:12} " + "  ".join(f"{g.split()[0]}={v:.2f}" for (g, _), v in zip(groups, row)))


if __name__ == "__main__":
    main()
