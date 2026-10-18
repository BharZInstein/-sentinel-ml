"""Alternating minimax training against a white-box adaptive attacker.

Each round generates fresh L-infinity examples with ``adaptive_pgd`` using the
current detector, then updates the detector on clean, benign-corrupted, and
adaptive examples.  The target classifier remains frozen throughout.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import config
from sentinelml.device import DEVICE, seed_everything
from sentinelml.target.train_target import load_target
from sentinelml.detector.data import load_caches, split_image_ids
from sentinelml.detector.train import MODELS
from sentinelml.attacks.adaptive import adaptive_pgd


def _concat(features):
    if isinstance(features[0], dict):
        return {k: torch.cat([f[k] for f in features]) for k in features[0]}
    return torch.cat(features)


def _on_device(features):
    if isinstance(features, dict):
        return {k: v.to(DEVICE).float() for k, v in features.items()}
    return features.to(DEVICE).float()


def _detector_scores(detector, features):
    return detector(_on_device(features))


@torch.no_grad()
def _labels_from_target(target, images):
    batches = []
    for start in range(0, len(images), config.MINIMAX_BATCH_SIZE):
        batches.append(target(images[start:start + config.MINIMAX_BATCH_SIZE].to(DEVICE)).argmax(1).cpu())
    return torch.cat(batches)


@torch.no_grad()
def _clean_score_quantile(extractor, detector, images, quantile: float = 0.95):
    detector_training = detector.training
    detector.eval()
    scores = []
    try:
        for start in range(0, len(images), config.MINIMAX_BATCH_SIZE):
            features = extractor(images[start:start + config.MINIMAX_BATCH_SIZE].to(DEVICE))
            scores.append(_detector_scores(detector, features).cpu())
        return float(torch.cat(scores).quantile(quantile))
    finally:
        detector.train(detector_training)


@torch.no_grad()
def _adaptive_evasion_rate(extractor, detector, adversarial_images, threshold):
    detector_training = detector.training
    detector.eval()
    evaded = total = 0
    try:
        for images in adversarial_images:
            for start in range(0, len(images), config.MINIMAX_BATCH_SIZE):
                x = images[start:start + config.MINIMAX_BATCH_SIZE].to(DEVICE)
                scores = _detector_scores(detector, extractor(x))
                evaded += int((scores < threshold).sum())
                total += len(x)
        return evaded / max(total, 1)
    finally:
        detector.train(detector_training)


def _parse_args():
    parser = argparse.ArgumentParser(description="Minimax detector training against adaptive white-box PGD.")
    parser.add_argument("--model", choices=list(MODELS), default="attention_early_hybrid")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help="training randomness; the train/test image split stays fixed")
    parser.add_argument("--rounds", type=int, default=config.MINIMAX_ROUNDS)
    parser.add_argument("--batch-size", type=int, default=config.MINIMAX_BATCH_SIZE)
    parser.add_argument("--attack-steps", type=int, default=config.MINIMAX_ATTACK_STEPS)
    parser.add_argument("--eps", type=float, default=config.ADAPTIVE_EPSILONS[1])
    parser.add_argument("--alpha", type=float, default=config.MINIMAX_ATTACK_ALPHA)
    parser.add_argument("--detector-weight", type=float, default=config.MINIMAX_DETECTOR_WEIGHT)
    parser.add_argument("--init", type=Path, default=None,
                        help="optional detector state dict to warm-start")
    parser.add_argument("--output", type=Path, default=None,
                        help="output state-dict path (defaults to artifacts/detector_<model>_minimax.pt)")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.rounds < 1 or args.batch_size < 1:
        raise SystemExit("rounds and batch-size must be positive")
    if args.attack_steps < 0 or args.eps < 0 or args.alpha < 0 or args.detector_weight < 0:
        raise SystemExit("attack-steps, eps, alpha, and detector-weight must be non-negative")

    seed_everything(args.seed)
    extractor_factory, detector_factory, _, _ = MODELS[args.model]
    target = load_target().to(DEVICE)
    extractor = extractor_factory(target)
    detector = detector_factory().to(DEVICE)

    if args.init is not None:
        detector.load_state_dict(torch.load(args.init, map_location=DEVICE, weights_only=True))
        print(f"[minimax] initialized detector from {args.init}")

    clean_pool, families, corruptions = load_caches()
    clean_cache = torch.load(config.ADV_CACHE_DIR / "clean.pt", weights_only=True)
    labels = clean_cache.get("labels")
    if labels is None:
        labels = _labels_from_target(target, clean_pool)

    # Keep the same held-out image IDs as the baseline and adaptive evaluator;
    # --seed controls optimization randomness only.
    train_ids, _ = split_image_ids(clean_pool.shape[0])
    train_images = clean_pool[train_ids]
    train_labels = labels[train_ids]
    train_set = set(train_ids.tolist())
    replay_parts = []
    for name in config.TRAIN_ATTACKS:
        if name not in families:
            continue
        adv, ids = families[name]
        keep = torch.tensor([int(i) in train_set for i in ids])
        if keep.any():
            replay_parts.append(adv[keep])
    static_replay = torch.cat(replay_parts) if replay_parts else None
    output = args.output or (config.ARTIFACTS_DIR / f"detector_{args.model}_minimax.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(detector.parameters(), lr=config.DETECTOR_LR)
    print(f"[minimax] device = {DEVICE}")
    print(f"[minimax] model = {args.model}; examples = {len(train_images)}; rounds = {args.rounds}")
    replay_families = [name for name in config.TRAIN_ATTACKS if name in families]
    missing_families = [name for name in config.TRAIN_ATTACKS if name not in families]
    print(f"[minimax] static replay families: {replay_families}")
    if missing_families:
        print(f"[minimax] WARNING: missing static attack caches: {missing_families}")
    print(f"[minimax] static replay: {len(static_replay) if static_replay is not None else 0} examples")
    print(f"[minimax] eps = {args.eps:.6f}; attack steps = {args.attack_steps}; detector weight = {args.detector_weight}")

    for round_idx in range(args.rounds):
        round_start = time.time()
        detector.train()
        perm = torch.randperm(len(train_images))
        total_loss = 0.0
        seen = 0
        target_successes = 0
        adaptive_images = []

        for start in range(0, len(perm), args.batch_size):
            batch_ids = perm[start:start + args.batch_size]
            clean = train_images[batch_ids].to(DEVICE)
            y = train_labels[batch_ids].to(DEVICE)

            adaptive = adaptive_pgd(
                target,
                extractor,
                detector,
                clean,
                y,
                eps=args.eps,
                alpha=args.alpha,
                steps=args.attack_steps,
                detector_weight=args.detector_weight,
                random_start=True,
            )

            detector.train()
            feature_batches = [extractor(clean), extractor(adaptive.adversarial.detach())]
            labels_batches = [
                torch.zeros(clean.shape[0], device=DEVICE),
                torch.ones(clean.shape[0], device=DEVICE),
            ]

            if corruptions is not None:
                corr = corruptions[train_ids[batch_ids]].to(DEVICE)
                feature_batches.append(extractor(corr))
                labels_batches.append(torch.zeros(corr.shape[0], device=DEVICE))

            # Keep the detector's original attack-family coverage while it
            # learns the adaptive best response.  Without replay, a short
            # minimax run can forget useful static-family decision boundaries.
            if static_replay is not None:
                replay_ids = torch.randint(
                    len(static_replay), (clean.shape[0],), generator=torch.Generator().manual_seed(
                        args.seed + round_idx + start
                    )
                )
                feature_batches.append(extractor(static_replay[replay_ids].to(DEVICE)))
                labels_batches.append(torch.ones(clean.shape[0], device=DEVICE))

            features = _concat(feature_batches)
            y_det = torch.cat(labels_batches)
            loss = F.binary_cross_entropy_with_logits(
                _detector_scores(detector, features), y_det
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = clean.shape[0]
            total_loss += loss.item() * batch_n
            seen += batch_n
            target_successes += int(adaptive.target_success.sum())
            adaptive_images.append(adaptive.adversarial.detach().cpu())

            batch_number = start // args.batch_size + 1
            if batch_number % 10 == 0 or batch_number == (len(perm) + args.batch_size - 1) // args.batch_size:
                print(
                    f"[minimax] round {round_idx + 1}/{args.rounds} "
                    f"batch {batch_number}/{(len(perm) + args.batch_size - 1) // args.batch_size} "
                    f"loss={loss.item():.4f} target_success={target_successes / seen:.3f}",
                    flush=True,
                )

        threshold = _clean_score_quantile(extractor, detector, train_images)
        # Re-score the round's attacks after all detector updates. Using scores
        # collected during the attack phase would mix several detector versions
        # and make this rate incomparable to the post-round clean threshold.
        evasion_rate = _adaptive_evasion_rate(extractor, detector, adaptive_images, threshold)
        print(
            f"[minimax] round {round_idx + 1:2d}/{args.rounds} "
            f"loss={total_loss / seen:.4f} "
            f"target_success={target_successes / seen:.3f} "
            f"detector_evasion={evasion_rate:.3f} "
            f"clean_threshold={threshold:.3f} "
            f"seconds={time.time() - round_start:.1f}",
            flush=True,
        )
        # Keep the most recent complete round on disk so long runs can be
        # resumed with --init if interrupted.
        torch.save(detector.state_dict(), output)

    print(f"[minimax] saved detector -> {output}")


if __name__ == "__main__":
    main()
