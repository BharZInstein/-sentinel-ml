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

Python 3.11 · PyTorch (CUDA, Apple-silicon MPS, or CPU — picked automatically) · torchattacks ·
scikit-learn · matplotlib

On an NVIDIA GPU, install a CUDA build of PyTorch (see pytorch.org); the default `pip` wheel on
Windows is CPU-only and every step below becomes much slower.

## Project layout

```
config.py                 # all tunable knobs (paths, seeds, epsilons, sizes)
sentinelml/
  device.py               # device selection + global seeding
  target/                 # the protected CIFAR-10 classifier
  attacks/                # FGSM/PGD/CW/Square/APGD/DeepFool/OnePixel/patch/spatial + clean-side corruptions
  detector/
    features.py           # forward hooks on the target: pooled vectors (MLP) or tokens (attention)
    data.py               # loads caches, splits train/test by image id
    model.py              # MLPDetector and AttentionDetector (token layouts, optional hybrid head)
    train.py              # train + evaluate any detector variant (--model, --seed)
    attention_map.py      # attention-weight figure for the best attention variant
  demo.py                 # renders clean vs adversarial examples to artifacts/adversarial_demo.png
```

Planned, not yet built: `eval/` (full leave-one-attack-out matrix), `scripts/run_all.py`, an
adaptive attacker, and minimax (attacker vs detector) training.

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

# 3. Train and evaluate a detector            -> artifacts/detector_<model>.pt
python -m sentinelml.detector.train --model mlp                      # baseline
python -m sentinelml.detector.train --model attention_early_hybrid   # best attention variant
python -m sentinelml.detector.train --model mlp --seed 1             # extra seed (split stays fixed)

# 4. Attention-weight figure for the best variant (needs step 3 with the default seed)
python -m sentinelml.detector.attention_map                          # -> artifacts/attention_c.png

# Optional: render example adversarial images -> artifacts/adversarial_demo.png
python -m sentinelml.demo
```

Detector variants (`--model`):

| Name | Input |
|---|---|
| `mlp` | Each stage average-pooled to one vector, plus 3 logit statistics (963 numbers) |
| `attention` | `[CLS]` + layer1/layer2 pooled, layer3 (8×8) and layer4 (4×4) positions, logit token (84 tokens) |
| `attention_early` | `[CLS]` + layer1/layer2/layer3 on 8×8 grids, layer4 pooled, logit token (195 tokens) |
| `attention_hybrid` | `attention`, plus the logit vector fed directly to the output layer |
| `attention_early_hybrid` | `attention_early` plus the direct logit path ("model C", the best variant) |

All attention variants: tokens projected to width 128, learned stage and position embeddings,
2 transformer encoder layers with 4 heads, ~0.4M parameters.

Notes:

- Step 2 must include the training families (`fgsm_4 fgsm_8 pgd_4 pgd_8 cw`, and `patch` if you
  want it) before step 3; step 3 uses whichever cache files exist and reports the rest as held out.
- Step 2 rewrites `clean.pt` each run. It is deterministic for a given target model, but regenerate
  all caches if you retrain the target.
- Slow attacks: on CPU, `apgd_ce` took about 40 minutes; on an RTX 4050 GPU, `patch` took about
  4 minutes and each detector trains in 1–4 minutes.
- `deepfool` is registered but its cache has not been generated yet, so it does not appear below.

## Results so far

Detectors trained on `fgsm_4 fgsm_8 pgd_4 pgd_8 cw patch`; every row is evaluated on held-out
*images* (25% split by image id). Held-out *families* were never seen in training. Mean ± std over
3 training seeds (1337, 1, 2). Metric: ROC-AUC / detection rate (TPR) at 5% false-positive rate.

| Family | Role | MLP AUC | Model C AUC | MLP TPR | Model C TPR |
|---|---|---|---|---|---|
| fgsm_4 | train | 0.904 ± 0.004 | **0.961 ± 0.004** | 0.531 ± 0.023 | **0.789 ± 0.016** |
| fgsm_8 | train | 0.967 ± 0.004 | **0.985 ± 0.004** | 0.811 ± 0.031 | **0.931 ± 0.026** |
| pgd_4 | train | 0.923 ± 0.002 | **0.968 ± 0.004** | 0.641 ± 0.003 | **0.844 ± 0.013** |
| pgd_8 | train | 0.973 ± 0.004 | **0.983 ± 0.003** | 0.876 ± 0.018 | **0.926 ± 0.012** |
| cw | train | 0.800 ± 0.005 | 0.798 ± 0.011 | 0.240 ± 0.005 | **0.279 ± 0.012** |
| patch | train | 0.936 ± 0.002 | **0.987 ± 0.005** | 0.692 ± 0.017 | **0.935 ± 0.024** |
| square | held-out | 0.891 ± 0.003 | **0.909 ± 0.006** | 0.467 ± 0.012 | **0.572 ± 0.006** |
| apgd_ce | held-out | 0.978 ± 0.003 | 0.982 ± 0.003 | 0.900 ± 0.003 | **0.924 ± 0.005** |
| apgd_dlr | held-out | 0.975 ± 0.004 | **0.991 ± 0.002** | 0.864 ± 0.023 | **0.953 ± 0.011** |
| onepixel | held-out | **0.783 ± 0.011** | 0.723 ± 0.030 | 0.122 ± 0.040 | 0.106 ± 0.011 |
| spatial | held-out | **0.643 ± 0.009** | 0.584 ± 0.030 | **0.111 ± 0.023** | 0.022 ± 0.009 |

False-positive rate on corrupted clean images (noise, blur, JPEG, brightness, contrast):
MLP 0.045 ± 0.003, model C 0.047 ± 0.014.

**Findings**

- Model C (`attention_early_hybrid`) beats the MLP on every gradient-based attack, on `patch`, and on
  the held-out black-box `square` attack, at the same clean false-positive rate.
- The gain came from keeping early layers spatial. An attention model with only late-layer spatial
  tokens (`attention`) did not beat the MLP; the direct logit path alone (`attention_hybrid`) did
  not help either.
- Neither detector handles `spatial` (rotation/translation, no added noise) or `onepixel`; the MLP
  is better on both. `onepixel` has only 41 held-out images and `spatial` 108, so those rows are
  the noisiest.

**Attention weights** (`artifacts/attention_c.png`, from `attention_map.py`): for noise attacks
(FGSM, PGD, APGD, Square) the `[CLS]` token puts 50–78% of its last-layer attention on the
layer1 grid, against 20% for clean images; for `patch` it shifts to layer2 and, in the example
image, lands on the patch itself. For `cw`, `onepixel` and `spatial` the attention pattern is close
to the clean one — the same families the detectors miss. Attention shows where the detector looks,
not a complete explanation of its decision.

Per-run logs: `artifacts/detector_*_run.log`.

## Status

Under active construction — see commit history. Phase 1 is being built module by module.
