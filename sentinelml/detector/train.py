from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

import config
from sentinelml.device import DEVICE, seed_everything
from sentinelml.target.train_target import load_target
from sentinelml.detector.data import load_caches, split_image_ids
from sentinelml.detector.features import FeatureExtractor
from sentinelml.detector.model import MLPDetector


@torch.no_grad()
def extract(extractor, images, batch_size: int = 256):
    out = []
    for i in range(0, images.shape[0], batch_size):
        out.append(extractor(images[i:i + batch_size].to(DEVICE)).cpu())
    return torch.cat(out)


def tpr_at_fpr(y_true, scores, fpr_target: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(fpr_target, fpr, tpr))


def main():
    seed_everything()
    extractor = FeatureExtractor(load_target().to(DEVICE))
    clean_pool, families, corruptions = load_caches()
    train_ids, test_ids = split_image_ids(clean_pool.shape[0])
    train_set, test_set = set(train_ids.tolist()), set(test_ids.tolist())

    train_fams = [f for f in config.TRAIN_ATTACKS if f in families]
    test_fams = [f for f in families if f not in train_fams]
    print(f"[detector] train families: {train_fams}")
    print(f"[detector] held-out families: {test_fams}")
    print(f"[detector] corruptions cache: {'yes' if corruptions is not None else 'MISSING'}")

    feat_clean = extract(extractor, clean_pool)
    feat_corr = extract(extractor, corruptions) if corruptions is not None else None

    def clean_side(ids):
        parts = [feat_clean[ids]]
        if feat_corr is not None:
            parts.append(feat_corr[ids])
        return torch.cat(parts)

    xs, ys = [clean_side(train_ids)], [torch.zeros(len(train_ids) * (2 if feat_corr is not None else 1))]
    for name in train_fams:
        adv, ids = families[name]
        keep = torch.tensor([int(i) in train_set for i in ids])
        feats = extract(extractor, adv[keep])
        xs.append(feats)
        ys.append(torch.ones(len(feats)))
    X, y = torch.cat(xs), torch.cat(ys)
    print(f"[detector] train examples: {len(X)} ({int(y.sum())} adversarial)")

    detector = MLPDetector().to(DEVICE)
    opt = torch.optim.Adam(detector.parameters(), lr=config.DETECTOR_LR)
    for epoch in range(config.DETECTOR_EPOCHS):
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), config.DETECTOR_BATCH_SIZE):
            b = perm[i:i + config.DETECTOR_BATCH_SIZE]
            loss = F.binary_cross_entropy_with_logits(detector(X[b].to(DEVICE)), y[b].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"[detector] epoch {epoch + 1:2d}  loss {total / len(X):.4f}")
    torch.save(detector.state_dict(), config.DETECTOR_WEIGHTS)

    detector.eval()
    with torch.no_grad():
        score = lambda f: detector(f.to(DEVICE)).cpu().numpy()
        clean_scores = score(clean_side(test_ids))
        thr = np.quantile(score(feat_clean[test_ids]), 0.95)
        print(f"\n[detector] held-out images: {len(test_ids)}; threshold at 5% FPR on clean = {thr:.3f}")
        print(f"{'family':10} {'role':8} {'n':>5} {'AUC':>6} {'TPR@5%FPR':>10}")
        for name in train_fams + test_fams:
            adv, ids = families[name]
            keep = torch.tensor([int(i) in test_set for i in ids])
            if keep.sum() == 0:
                continue
            adv_scores = score(extract(extractor, adv[keep]))
            y_true = np.r_[np.zeros(len(clean_scores)), np.ones(len(adv_scores))]
            s = np.r_[clean_scores, adv_scores]
            role = "train" if name in train_fams else "held-out"
            print(f"{name:10} {role:8} {len(adv_scores):5d} {roc_auc_score(y_true, s):6.3f} "
                  f"{tpr_at_fpr(y_true, s):10.3f}")
        if feat_corr is not None:
            fpr_corr = float((score(feat_corr[test_ids]) > thr).mean())
            print(f"clean FPR on corruptions: {fpr_corr:.3f}")


if __name__ == "__main__":
    main()
