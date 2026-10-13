# SentinelML

**A runtime adversarial-attack detection layer for production ML models.**

SentinelML is an "AI firewall" that sits in front of a protected model, screens every inbound
inference request, and flags inputs that look adversarially manipulated *before* they reach the
real model — an intrusion-detection system for ML pipelines instead of network traffic.

This repository is the **Phase 1 vision research core**. The full design is in
[`SentinelML_TDD.md`](./SentinelML_TDD.md).

## Research question

Most published adversarial-detection work is *attack-specific* and reports high accuracy against the
one attack it trained on. SentinelML instead measures the honest question:

> How well does a **single** detector generalize to attack families it was **never trained on**,
> and where does that generalization break?

We build a train-attack × test-attack **generalization matrix** (leave-one-attack-out), report
threshold-independent ROC-AUC, and measure the false-positive rate on realistic clean traffic.

## Phase 1 scope

- **Track:** vision — evasion attacks on a CIFAR-10 classifier. The detector trains on FGSM, PGD,
  Carlini-Wagner and patch attacks; Square, APGD (CE/DLR), DeepFool, OnePixel and a spatial
  (rotation/translation) attack are held out for out-of-distribution testing. Benign corruptions
  (noise, blur, JPEG, brightness, contrast) are used as extra clean examples.
- **Core:** attack generation → detector training → generalization + FPR evaluation.
- **Deferred:** the FastAPI gateway, the SIEM dashboard, the LLM track, and adaptive white-box
  red-teaming (see the TDD).

## Stack

Python 3.11 · PyTorch (Apple-silicon MPS) · torchattacks · scikit-learn · matplotlib

## Project layout

```
config.py                 # all tunable knobs (paths, seeds, epsilons, sizes)
sentinelml/
  device.py               # device selection + global seeding
  target/                 # the protected CIFAR-10 classifier
  attacks/                # FGSM/PGD/CW/Square/APGD/DeepFool/OnePixel/patch/spatial + clean-side corruptions
  detector/               # MLP baseline on pooled target activations (features, data, model, train)
  demo.py                 # renders clean vs adversarial examples to artifacts/adversarial_demo.png
```

Planned, not yet built: `eval/` (generalization matrix), `scripts/run_all.py`, an attention-based
detector, and an adaptive attacker.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run everything from the repository root, in this order:

```bash
# 1. Train the protected CIFAR-10 classifier  -> artifacts/target.pt
python -m sentinelml.target.train_target

# 2. Generate adversarial caches + the clean-side corruptions cache -> data/adv_cache/
python -m sentinelml.attacks.generate                 # every attack family (slow on CPU)
python -m sentinelml.attacks.generate fgsm_4 pgd_8    # or only the families you name
python -m sentinelml.attacks.generate --no-corruptions apgd_ce   # skip rewriting corruptions.pt

# 3. Train and evaluate the detector          -> artifacts/detector_mlp.pt
python -m sentinelml.detector.train

# Optional: render example adversarial images -> artifacts/adversarial_demo.png
python -m sentinelml.demo
```

Notes:

- Step 2 must include the training families (`fgsm_4 fgsm_8 pgd_4 pgd_8 cw`, and `patch` if you
  want it) before step 3; step 3 uses whichever cache files exist and reports the rest as held out.
- Step 2 rewrites `clean.pt` each run. It is deterministic for a given target model, but regenerate
  all caches if you retrain the target.
- `cw`, `onepixel`, `spatial` and `apgd_*` are slow on CPU (`apgd_ce` took about 40 minutes).

## Results so far (MLP baseline)

AUC / detection rate at 5% false positives, on held-out images:

| Family | Role | AUC | TPR @ 5% FPR |
|---|---|---|---|
| fgsm_4 | train | 0.918 | 0.573 |
| fgsm_8 | train | 0.970 | 0.806 |
| pgd_4 | train | 0.939 | 0.692 |
| pgd_8 | train | 0.978 | 0.891 |
| cw | train | 0.849 | 0.355 |
| square | held-out | 0.925 | 0.622 |
| apgd_ce | held-out | 0.978 | 0.892 |

False-positive rate on the corrupted clean images: 0.043. Full log: `artifacts/detector_run.log`.

## Status

Under active construction — see commit history. Phase 1 is being built module by module.
