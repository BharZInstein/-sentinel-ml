from __future__ import annotations

import time

import torch

import config
from sentinelml.device import DEVICE, seed_everything
from sentinelml.target import data as target_data
from sentinelml.target.train_target import load_target
from sentinelml.attacks.registry import ATTACK_FACTORIES, POOL_SIZE, build_attack


@torch.no_grad()
def _correctly_classified_pool(model, dataset, n: int):

    imgs, labels = [], []
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.ATTACK_BATCH_SIZE)
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        keep = model(x).argmax(1) == y
        imgs.append(x[keep].cpu())
        labels.append(y[keep].cpu())
        if sum(t.shape[0] for t in imgs) >= n:
            break
    imgs = torch.cat(imgs)[:n]
    labels = torch.cat(labels)[:n]
    return imgs, labels


def _run_attack(model, attack, images, labels):

    adv_batches = []
    for i in range(0, images.shape[0], config.ATTACK_BATCH_SIZE):
        xb = images[i:i + config.ATTACK_BATCH_SIZE].to(DEVICE)
        yb = labels[i:i + config.ATTACK_BATCH_SIZE].to(DEVICE)
        adv_batches.append(attack(xb, yb).detach().cpu())
    return torch.cat(adv_batches)


@torch.no_grad()
def _keep_successful(model, adv, labels):

    preds = []
    for i in range(0, adv.shape[0], config.ATTACK_BATCH_SIZE):
        xb = adv[i:i + config.ATTACK_BATCH_SIZE].to(DEVICE)
        preds.append(model(xb).argmax(1).cpu())
    preds = torch.cat(preds)
    return preds != labels


def generate_family(model, name: str, clean_imgs, clean_labels) -> dict:

    attack = build_attack(name, model)
    t0 = time.time()
    adv = _run_attack(model, attack, clean_imgs, clean_labels)
    success = _keep_successful(model, adv, clean_labels)

    cache = {
        "family": name,
        "adv": adv[success],
        "clean": clean_imgs[success],
        "labels": clean_labels[success],
        "meta": {
            "attempted": int(clean_imgs.shape[0]),
            "successful": int(success.sum()),
            "success_rate": float(success.float().mean()),
            "seconds": round(time.time() - t0, 1),
        },
    }
    m = cache["meta"]
    print(f"[attack] {name:8} success {m['successful']}/{m['attempted']} "
          f"({m['success_rate'] * 100:.1f}%) in {m['seconds']}s")
    return cache


def main():
    seed_everything()
    model = load_target()
    print(f"[attack] device = {DEVICE}")


    pool_n = max(POOL_SIZE.values())
    dataset = target_data.test_dataset()
    clean_imgs, clean_labels = _correctly_classified_pool(model, dataset, pool_n)
    print(f"[attack] clean pool: {clean_imgs.shape[0]} correctly-classified images")


    torch.save({"clean": clean_imgs, "labels": clean_labels},
               config.ADV_CACHE_DIR / "clean.pt")

    for name in ATTACK_FACTORIES:
        n = POOL_SIZE[name]
        cache = generate_family(model, name, clean_imgs[:n], clean_labels[:n])
        torch.save(cache, config.ADV_CACHE_DIR / f"{name}.pt")

    print(f"[attack] done -> {config.ADV_CACHE_DIR}")


if __name__ == "__main__":
    main()
