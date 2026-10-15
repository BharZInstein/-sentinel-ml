from __future__ import annotations

import argparse
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

import config
from sentinelml.device import DEVICE, seed_everything
from sentinelml.target.train_target import load_target
from sentinelml.detector.data import load_caches, split_image_ids
from sentinelml.detector.features import FeatureExtractor, TokenExtractor
from sentinelml.detector.model import AttentionDetector, MLPDetector

# Per model: (feature extractor, detector, weights path, storage dtype for precomputed features).
# Attention tokens are ~25k numbers per image, so they are kept in float16 (~50 KB/image).
MODELS = {
    "mlp": (FeatureExtractor, MLPDetector, config.DETECTOR_WEIGHTS, torch.float32),
    "attention": (TokenExtractor, AttentionDetector, config.DETECTOR_ATTN_WEIGHTS, torch.float16),
    # Variants: early-layer 8x8 token grids, a direct logit path to the output, or both.
    "attention_early": (partial(TokenExtractor, layout="early"), partial(AttentionDetector, layout="early"),
                        config.ARTIFACTS_DIR / "detector_attention_early.pt", torch.float16),
    "attention_hybrid": (TokenExtractor, partial(AttentionDetector, hybrid=True),
                         config.ARTIFACTS_DIR / "detector_attention_hybrid.pt", torch.float16),
    "attention_early_hybrid": (partial(TokenExtractor, layout="early"),
                               partial(AttentionDetector, layout="early", hybrid=True),
                               config.ARTIFACTS_DIR / "detector_attention_early_hybrid.pt", torch.float16),
}


# Features are a tensor (MLP) or a dict of tensors (attention); these helpers handle both.
def _take(f, idx):
    return {k: v[idx] for k, v in f.items()} if isinstance(f, dict) else f[idx]


def _cat(fs):
    return {k: torch.cat([f[k] for f in fs]) for k in fs[0]} if isinstance(fs[0], dict) else torch.cat(fs)


def _to_device(f):
    return {k: v.to(DEVICE).float() for k, v in f.items()} if isinstance(f, dict) else f.to(DEVICE).float()


def _len(f):
    return len(next(iter(f.values()))) if isinstance(f, dict) else len(f)


@torch.no_grad()
def extract(extractor, images, dtype=torch.float32, batch_size: int = 256):
    out = []
    for i in range(0, images.shape[0], batch_size):
        f = extractor(images[i:i + batch_size].to(DEVICE))
        out.append({k: v.cpu().to(dtype) for k, v in f.items()} if isinstance(f, dict) else f.cpu().to(dtype))
    return _cat(out)


def tpr_at_fpr(y_true, scores, fpr_target: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(fpr_target, fpr, tpr))


def _parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate an adversarial-input detector.")
    parser.add_argument("--model", choices=list(MODELS), default="mlp")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="training seed (weight init, batch order); the train/test split stays fixed")
    return parser.parse_args()


def main():
    args = _parse_args()
    extractor_cls, detector_cls, weights_path, dtype = MODELS[args.model]
    if args.seed != config.SEED:
        weights_path = weights_path.with_name(f"{weights_path.stem}_seed{args.seed}.pt")

    seed_everything(args.seed)
    extractor = extractor_cls(load_target().to(DEVICE))
    clean_pool, families, corruptions = load_caches()
    train_ids, test_ids = split_image_ids(clean_pool.shape[0])
    train_set, test_set = set(train_ids.tolist()), set(test_ids.tolist())

    train_fams = [f for f in config.TRAIN_ATTACKS if f in families]
    test_fams = [f for f in families if f not in train_fams]
    print(f"[detector] model: {args.model}  seed: {args.seed}")
    print(f"[detector] train families: {train_fams}")
    print(f"[detector] held-out families: {test_fams}")
    print(f"[detector] corruptions cache: {'yes' if corruptions is not None else 'MISSING'}")

    feat_clean = extract(extractor, clean_pool, dtype)
    feat_corr = extract(extractor, corruptions, dtype) if corruptions is not None else None

    def clean_side(ids):
        parts = [_take(feat_clean, ids)]
        if feat_corr is not None:
            parts.append(_take(feat_corr, ids))
        return _cat(parts)

    xs, ys = [clean_side(train_ids)], [torch.zeros(len(train_ids) * (2 if feat_corr is not None else 1))]
    for name in train_fams:
        adv, ids = families[name]
        keep = torch.tensor([int(i) in train_set for i in ids])
        feats = extract(extractor, adv[keep], dtype)
        xs.append(feats)
        ys.append(torch.ones(_len(feats)))
    X, y = _cat(xs), torch.cat(ys)
    print(f"[detector] train examples: {_len(X)} ({int(y.sum())} adversarial)")

    detector = detector_cls().to(DEVICE)
    print(f"[detector] parameters: {sum(p.numel() for p in detector.parameters()):,}")
    opt = torch.optim.Adam(detector.parameters(), lr=config.DETECTOR_LR)
    for epoch in range(config.DETECTOR_EPOCHS):
        detector.train()
        perm = torch.randperm(_len(X))
        total = 0.0
        for i in range(0, _len(X), config.DETECTOR_BATCH_SIZE):
            b = perm[i:i + config.DETECTOR_BATCH_SIZE]
            loss = F.binary_cross_entropy_with_logits(detector(_to_device(_take(X, b))), y[b].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"[detector] epoch {epoch + 1:2d}  loss {total / _len(X):.4f}")
    torch.save(detector.state_dict(), weights_path)

    detector.eval()
    with torch.no_grad():
        def score(f, batch_size: int = 512):
            return np.concatenate([
                detector(_to_device(_take(f, slice(i, i + batch_size)))).cpu().numpy()
                for i in range(0, _len(f), batch_size)
            ])

        clean_scores = score(clean_side(test_ids))
        thr = np.quantile(score(_take(feat_clean, test_ids)), 0.95)
        print(f"\n[detector] held-out images: {len(test_ids)}; threshold at 5% FPR on clean = {thr:.3f}")
        print(f"{'family':10} {'role':8} {'n':>5} {'AUC':>6} {'TPR@5%FPR':>10}")
        for name in train_fams + test_fams:
            adv, ids = families[name]
            keep = torch.tensor([int(i) in test_set for i in ids])
            if keep.sum() == 0:
                continue
            adv_scores = score(extract(extractor, adv[keep], dtype))
            y_true = np.r_[np.zeros(len(clean_scores)), np.ones(len(adv_scores))]
            s = np.r_[clean_scores, adv_scores]
            role = "train" if name in train_fams else "held-out"
            print(f"{name:10} {role:8} {len(adv_scores):5d} {roc_auc_score(y_true, s):6.3f} "
                  f"{tpr_at_fpr(y_true, s):10.3f}")
        if feat_corr is not None:
            fpr_corr = float((score(_take(feat_corr, test_ids)) > thr).mean())
            print(f"clean FPR on corruptions: {fpr_corr:.3f}")


if __name__ == "__main__":
    main()
