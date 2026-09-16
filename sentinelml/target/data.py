from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


def _train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def _eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor()])


def train_loader(batch_size: int = config.TARGET_BATCH_SIZE) -> DataLoader:
    ds = datasets.CIFAR10(config.DATA_DIR, train=True, download=True, transform=_train_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)


def test_loader(batch_size: int = config.TARGET_BATCH_SIZE) -> DataLoader:
    ds = datasets.CIFAR10(config.DATA_DIR, train=False, download=True, transform=_eval_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def test_dataset():

    return datasets.CIFAR10(config.DATA_DIR, train=False, download=True, transform=_eval_transform())


@torch.no_grad()
def evaluate(model, loader, device) -> float:

    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total
