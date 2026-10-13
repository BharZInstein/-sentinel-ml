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

- **Track:** vision — FGSM / PGD evasion on a CIFAR-10 classifier, with Carlini-Wagner and a
  spatial attack held out for out-of-distribution testing.
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
  detector/               # feature-space + input-space adversarial detectors
  eval/                   # generalization matrix, ROC-AUC, clean FPR
scripts/run_all.py        # end-to-end pipeline
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

Under active construction — see commit history. Phase 1 is being built module by module.
