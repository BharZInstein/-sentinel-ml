from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import config
from sentinelml.device import DEVICE, seed_everything
from sentinelml.target import data as target_data
from sentinelml.target.train_target import load_target
from sentinelml.attacks.registry import build_attack

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

N_EXAMPLES = 5
ATTACK = "fgsm_8"


@torch.no_grad()
def _predict(model, x):

    probs = F.softmax(model(x), dim=1)
    conf, idx = probs.max(1)
    return idx.item(), conf.item()


def find_flipped_examples(model, attack, dataset, n):

    examples = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        x = img.unsqueeze(0).to(DEVICE)

        clean_idx, clean_conf = _predict(model, x)
        if clean_idx != label:
            continue

        adv = attack(x, torch.tensor([label], device=DEVICE))
        adv_idx, adv_conf = _predict(model, adv)
        if adv_idx == label:
            continue

        examples.append({
            "clean": img.cpu(),
            "adv": adv.squeeze(0).cpu(),
            "true": label,
            "clean_pred": (clean_idx, clean_conf),
            "adv_pred": (adv_idx, adv_conf),
        })
        if len(examples) >= n:
            break
    return examples


def render(examples, out_path):
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.15 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    col_titles = ["Original", "Perturbation (×10)", "Adversarial"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=12, fontweight="bold")

    for r, ex in enumerate(examples):
        clean = ex["clean"].permute(1, 2, 0).numpy()
        adv = ex["adv"].permute(1, 2, 0).numpy()
        pert = (ex["adv"] - ex["clean"])
        pert_vis = (pert * 10 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

        ci, cc = ex["clean_pred"]
        ai, ac = ex["adv_pred"]

        axes[r, 0].imshow(clean)
        axes[r, 0].set_xlabel(f"'{CIFAR_CLASSES[ci]}'  ({cc*100:.0f}%)",
                              color="#1a7f37", fontsize=11, fontweight="bold")

        axes[r, 1].imshow(pert_vis)
        axes[r, 1].set_xlabel("added noise\n(invisible at real scale)",
                              color="#555", fontsize=9)

        axes[r, 2].imshow(adv)
        axes[r, 2].set_xlabel(f"'{CIFAR_CLASSES[ai]}'  ({ac*100:.0f}%)",
                              color="#cf222e", fontsize=11, fontweight="bold")

        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])

    fig.suptitle("SentinelML — Adversarial Attack Demo (FGSM, ε=8/255)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005,
             "Same image to a human → the classifier flips its answer. "
             "This is the threat SentinelML detects.",
             ha="center", fontsize=10, color="#333")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[demo] saved figure -> {out_path}")


def main():
    seed_everything()
    print(f"[demo] device = {DEVICE}")
    model = load_target()
    attack = build_attack(ATTACK, model)
    dataset = target_data.test_dataset()

    examples = find_flipped_examples(model, attack, dataset, N_EXAMPLES)
    print(f"[demo] found {len(examples)} successfully-attacked examples")
    for ex in examples:
        ci, cc = ex["clean_pred"]
        ai, ac = ex["adv_pred"]
        print(f"  {CIFAR_CLASSES[ci]:>10} ({cc*100:.0f}%)  ->  "
              f"{CIFAR_CLASSES[ai]:<10} ({ac*100:.0f}%)")

    out_path = config.ARTIFACTS_DIR / "adversarial_demo.png"
    render(examples, out_path)


if __name__ == "__main__":
    main()
