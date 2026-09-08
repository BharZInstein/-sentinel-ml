"""Central configuration for SentinelML (Phase 1: vision research core).

All tunable knobs live here so experiments are reproducible and easy to scale
up or down. Keep dataset sizes modest by default so the full pipeline runs in
minutes on an Apple-silicon laptop; raise them for a more thorough benchmark.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ADV_CACHE_DIR = DATA_DIR / "adv_cache"
ARTIFACTS_DIR = ROOT / "artifacts"

TARGET_WEIGHTS = ARTIFACTS_DIR / "target.pt"
DETECTOR_WEIGHTS = ARTIFACTS_DIR / "detector_mlp.pt"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"
MATRIX_CSV = ARTIFACTS_DIR / "generalization_matrix.csv"
MATRIX_PNG = ARTIFACTS_DIR / "generalization_matrix.png"
ROC_PNG = ARTIFACTS_DIR / "roc_curves.png"

for _d in (DATA_DIR, ADV_CACHE_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 1337

# ---------------------------------------------------------------------------
# CIFAR-10 normalization (channel means / stds)
# ---------------------------------------------------------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10

# ---------------------------------------------------------------------------
# Target model training
# ---------------------------------------------------------------------------
TARGET_EPOCHS = 12
TARGET_BATCH_SIZE = 128
TARGET_LR = 0.1
TARGET_WEIGHT_DECAY = 5e-4
TARGET_MOMENTUM = 0.9

# ---------------------------------------------------------------------------
# Attack / dataset generation
# ---------------------------------------------------------------------------
# Number of clean CIFAR-10 *test* images to draw from when crafting attacks.
# Each attack family perturbs (a slice of) this pool; keep it modest for speed.
ATTACK_POOL_SIZE = 4000
# Carlini-Wagner is iterative and slow — cap how many images it attacks.
CW_POOL_SIZE = 800
ATTACK_BATCH_SIZE = 128

# Epsilons are expressed in the 0-1 pixel space (converted to normalized space
# inside the attack wrappers). A varied grid prevents single-config overfitting.
FGSM_EPSILONS = [4 / 255, 8 / 255]
PGD_EPSILONS = [4 / 255, 8 / 255]
PGD_STEPS = 10
PGD_ALPHA = 2 / 255

# ---------------------------------------------------------------------------
# Attack taxonomy: which families train the detector vs. are held out for the
# out-of-distribution generalization test. This split is the research core.
# ---------------------------------------------------------------------------
TRAIN_ATTACKS = ["fgsm_4", "fgsm_8", "pgd_4", "pgd_8"]
HELDOUT_ATTACKS = ["cw", "spatial"]
ALL_ATTACKS = TRAIN_ATTACKS + HELDOUT_ATTACKS

# ---------------------------------------------------------------------------
# Detector training
# ---------------------------------------------------------------------------
DETECTOR_EPOCHS = 20
DETECTOR_BATCH_SIZE = 128
DETECTOR_LR = 1e-3
TEST_SPLIT = 0.25  # fraction of each family's examples reserved for evaluation

# Operating threshold on detector score for reporting detection rate / FPR.
DECISION_THRESHOLD = 0.5
