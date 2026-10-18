"""White-box attacks against the target model and the detector together.

The attacker solves the inner problem used by the minimax trainer:

    maximize_x  CE(target(x), y) - lambda * detector(target_features(x))
    subject to  ||x - x0||_inf <= epsilon and x in [0, 1].

The first term seeks a target-model error.  The second term makes the same
example look clean to the detector.  The target and detector weights are never
updated by this module; gradients are taken only with respect to the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

import config


@dataclass
class AdaptiveAttackResult:
    adversarial: torch.Tensor
    target_success: torch.Tensor
    detector_scores: torch.Tensor
    joint_success: torch.Tensor


def _forward_target_and_detector(target, extractor, detector, x):
    """Return target logits and detector logits from one target-model pass."""

    features = extractor(x)
    detector_logits = detector(features)
    if isinstance(features, dict) and "logits" in features:
        # TokenExtractor keeps the raw target logits before the three summary stats.
        target_logits = features["logits"][..., :config.NUM_CLASSES]
    else:
        target_logits = extractor.last_logits
    return target_logits, detector_logits


def _set_requires_grad(module, value: bool):
    previous = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(value)
    return previous


def _restore_requires_grad(module, previous):
    for p, value in zip(module.parameters(), previous):
        p.requires_grad_(value)


def adaptive_pgd(
    target,
    extractor,
    detector,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    eps: float = config.ADAPTIVE_EPSILONS[1],
    alpha: float = config.ADAPTIVE_ALPHA,
    steps: int = config.ADAPTIVE_STEPS,
    detector_weight: float = config.ADAPTIVE_DETECTOR_WEIGHT,
    random_start: bool = True,
    detector_threshold: Optional[float] = None,
) -> AdaptiveAttackResult:
    """Craft an untargeted L-infinity attack that also evades ``detector``.

    ``detector`` must output a scalar logit per image, with larger values meaning
    more adversarial as in ``MLPDetector`` and ``AttentionDetector``.  The
    returned ``joint_success`` is the event required to defeat the runtime
    firewall: the target prediction changes and the detector score is below the
    supplied threshold.  If no threshold is supplied it equals
    ``target_success``.
    """

    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same batch dimension")
    if eps < 0 or alpha < 0 or steps < 0 or detector_weight < 0:
        raise ValueError("eps, alpha, steps, and detector_weight must be non-negative")

    target_training = target.training
    detector_training = detector.training
    target_requires_grad = _set_requires_grad(target, False)
    detector_requires_grad = _set_requires_grad(detector, False)
    target.eval()
    detector.eval()

    clean = images.detach()
    if random_start and eps > 0:
        adv = clean + torch.empty_like(clean).uniform_(-eps, eps)
        adv = adv.clamp(0, 1)
    else:
        adv = clean.clone()

    best_adv = adv.detach().clone()
    best_objective = torch.full((len(clean),), -torch.inf, device=clean.device)

    try:
        for _ in range(steps):
            adv.requires_grad_(True)
            target_logits, detector_logits = _forward_target_and_detector(
                target, extractor, detector, adv
            )
            target_losses = F.cross_entropy(target_logits, labels, reduction="none")
            # Gradient ascent on this objective attacks the target while lowering
            # the detector's adversarial logit.
            objective = target_losses - detector_weight * detector_logits
            improve = objective.detach() > best_objective
            best_adv[improve] = adv.detach()[improve]
            best_objective = torch.maximum(best_objective, objective.detach())
            gradient, = torch.autograd.grad(objective.mean(), adv)
            adv = adv.detach() + alpha * gradient.sign()
            delta = (adv - clean).clamp(-eps, eps)
            adv = (clean + delta).clamp(0, 1).detach()

        with torch.no_grad():
            target_logits, detector_scores = _forward_target_and_detector(
                target, extractor, detector, adv
            )
            final_objective = F.cross_entropy(target_logits, labels, reduction="none") - detector_weight * detector_scores
            improve = final_objective > best_objective
            best_adv[improve] = adv[improve]
            best_objective = torch.maximum(best_objective, final_objective)
            adv = best_adv
            target_logits, detector_scores = _forward_target_and_detector(
                target, extractor, detector, adv
            )
            target_success = target_logits.argmax(1).ne(labels)
            if detector_threshold is None:
                joint_success = target_success.clone()
            else:
                joint_success = target_success & (detector_scores < detector_threshold)
    finally:
        _restore_requires_grad(target, target_requires_grad)
        _restore_requires_grad(detector, detector_requires_grad)
        target.train(target_training)
        detector.train(detector_training)

    return AdaptiveAttackResult(
        adversarial=adv,
        target_success=target_success,
        detector_scores=detector_scores,
        joint_success=joint_success,
    )


def linf_distances(clean: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
    """Per-image L-infinity perturbation size."""

    return (adversarial - clean).flatten(1).abs().amax(1)
