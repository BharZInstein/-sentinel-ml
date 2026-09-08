"""Train the target CIFAR-10 classifier and save its weights.

Run as a module from the project root:

    python -m sentinelml.target.train_target

The trained weights land at `artifacts/target.pt`. Accuracy only needs to be
reasonable (~85%+) — adversarial attacks succeed regardless of the exact score.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn

import config
from sentinelml.device import DEVICE, seed_everything
from sentinelml.target import data
from sentinelml.target.model import build_target


def train() -> float:
    seed_everything()
    print(f"[target] device = {DEVICE}")

    model = build_target().to(DEVICE)
    train_dl = data.train_loader()
    test_dl = data.test_loader()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.TARGET_LR,
        momentum=config.TARGET_MOMENTUM,
        weight_decay=config.TARGET_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TARGET_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, config.TARGET_EPOCHS + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        acc = data.evaluate(model, test_dl, DEVICE)
        best_acc = max(best_acc, acc)
        print(
            f"[target] epoch {epoch:2d}/{config.TARGET_EPOCHS} "
            f"loss={running / len(train_dl):.3f} test_acc={acc * 100:.2f}% "
            f"({time.time() - t0:.1f}s)"
        )

    config.TARGET_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.TARGET_WEIGHTS)
    print(f"[target] saved weights -> {config.TARGET_WEIGHTS} (best acc {best_acc * 100:.2f}%)")
    return best_acc


def load_target(weights_path=config.TARGET_WEIGHTS, device=DEVICE):
    """Load a trained target model in eval mode."""
    model = build_target().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


if __name__ == "__main__":
    train()
