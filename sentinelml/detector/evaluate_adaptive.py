"""Measure the white-box adaptive attack curve for a trained detector."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

import config
from sentinelml.device import DEVICE, seed_everything
from sentinelml.target.train_target import load_target
from sentinelml.detector.data import load_caches, split_image_ids
from sentinelml.detector.train import MODELS
from sentinelml.attacks.adaptive import adaptive_pgd, linf_distances


@torch.no_grad()
def _labels_from_target(target, images):
    labels = []
    for start in range(0, len(images), config.MINIMAX_BATCH_SIZE):
        labels.append(target(images[start:start + config.MINIMAX_BATCH_SIZE].to(DEVICE)).argmax(1).cpu())
    return torch.cat(labels)


@torch.no_grad()
def _score_quantile(extractor, detector, images, quantile: float = 0.95):
    detector_training = detector.training
    detector.eval()
    scores = []
    try:
        for start in range(0, len(images), 256):
            features = extractor(images[start:start + 256].to(DEVICE))
            if isinstance(features, dict):
                features = {k: v.to(DEVICE).float() for k, v in features.items()}
            else:
                features = features.to(DEVICE).float()
            scores.append(detector(features).cpu())
        return float(torch.cat(scores).quantile(quantile))
    finally:
        detector.train(detector_training)


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a detector under adaptive white-box PGD.")
    parser.add_argument("--model", choices=list(MODELS), default="attention_early_hybrid")
    parser.add_argument("--weights", type=Path, default=None,
                        help="detector state dict; defaults to the model's standard weights")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--limit", type=int, default=256,
                        help="number of held-out images to attack")
    parser.add_argument("--steps", type=int, default=config.ADAPTIVE_STEPS)
    parser.add_argument("--alpha", type=float, default=config.ADAPTIVE_ALPHA)
    parser.add_argument("--detector-weight", type=float, default=config.ADAPTIVE_DETECTOR_WEIGHT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.limit < 1:
        raise SystemExit("limit must be positive")
    if args.steps < 0 or args.alpha < 0 or args.detector_weight < 0:
        raise SystemExit("steps, alpha, and detector-weight must be non-negative")

    seed_everything(args.seed)
    extractor_factory, detector_factory, default_weights, _ = MODELS[args.model]
    weights = args.weights or default_weights
    if not weights.exists():
        raise SystemExit(f"detector weights not found: {weights}")

    target = load_target().to(DEVICE)
    extractor = extractor_factory(target)
    detector = detector_factory().to(DEVICE)
    detector.load_state_dict(torch.load(weights, map_location=DEVICE, weights_only=True))
    detector.eval()

    clean_pool, _, _ = load_caches()
    clean_cache = torch.load(config.ADV_CACHE_DIR / "clean.pt", weights_only=True)
    labels = clean_cache.get("labels")
    if labels is None:
        labels = _labels_from_target(target, clean_pool)

    train_ids, test_ids = split_image_ids(clean_pool.shape[0], seed=config.SEED)
    threshold = _score_quantile(extractor, detector, clean_pool[train_ids])
    test_ids = test_ids[:args.limit]
    images = clean_pool[test_ids]
    test_labels = labels[test_ids]

    output = args.output or (config.ARTIFACTS_DIR / f"adaptive_{args.model}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[adaptive] device = {DEVICE}")
    print(f"[adaptive] model = {args.model}; images = {len(images)}; threshold = {threshold:.4f}")
    print(f"{'eps':>8} {'target_success':>15} {'detector_detection':>19} {'firewall_bypass':>16} {'mean_linf':>12}")

    for eps in config.ADAPTIVE_EPSILONS:
        successes = detections = bypasses = 0
        distances = []
        for start in range(0, len(images), config.MINIMAX_BATCH_SIZE):
            clean = images[start:start + config.MINIMAX_BATCH_SIZE].to(DEVICE)
            y = test_labels[start:start + config.MINIMAX_BATCH_SIZE].to(DEVICE)
            result = adaptive_pgd(
                target,
                extractor,
                detector,
                clean,
                y,
                eps=eps,
                alpha=args.alpha,
                steps=args.steps,
                detector_weight=args.detector_weight,
                random_start=True,
                detector_threshold=threshold,
            )
            successes += int(result.target_success.sum())
            detections += int((result.target_success & ~result.joint_success).sum())
            bypasses += int(result.joint_success.sum())
            distances.append(linf_distances(clean, result.adversarial).cpu())

        n = len(images)
        rows.append({
            "model": args.model,
            "epsilon": eps,
            "threshold": threshold,
            "target_success_rate": successes / n,
            "detector_detection_rate_on_successes": detections / max(successes, 1),
            "firewall_bypass_rate": bypasses / n,
            "mean_linf": float(torch.cat(distances).mean()),
        })
        row = rows[-1]
        print(
            f"{eps:8.5f} {row['target_success_rate']:15.3f} "
            f"{row['detector_detection_rate_on_successes']:19.3f} "
            f"{row['firewall_bypass_rate']:16.3f} {row['mean_linf']:12.5f}"
        )

    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[adaptive] saved curve -> {output}")


if __name__ == "__main__":
    main()
