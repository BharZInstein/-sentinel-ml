"""The protected CIFAR-10 classifier.

A compact ResNet (ResNet-18 style, adapted for 32x32 inputs) built from
torchvision-free primitives so the architecture is fully explicit and easy to
hook for feature-space detection later.

The model expects *normalized* inputs. To keep adversarial attacks operating in
the natural 0-1 pixel space, normalization is folded into the model as a
non-trainable first layer (`NormalizeLayer`). This means attacks perturb raw
pixels while the network still sees standardized features.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class NormalizeLayer(nn.Module):
    """Applies per-channel normalization inside the model.

    Folding normalization into the graph lets attack libraries craft
    perturbations directly in [0, 1] pixel space (the perceptually meaningful
    space) while the convolutional backbone still receives standardized input.
    """

    def __init__(self, mean=config.CIFAR_MEAN, std=config.CIFAR_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    """Standard ResNet residual block."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """ResNet-18 adapted for 32x32 CIFAR-10 (3x3 stem, no max-pool)."""

    def __init__(self, block=BasicBlock, num_blocks=(2, 2, 2, 2), num_classes=config.NUM_CLASSES):
        super().__init__()
        self.normalize = NormalizeLayer()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer activations (the 512-d pooled feature vector).

        This is the representation SentinelML's feature-space detector reads —
        exposed as a first-class method so the detector never has to guess at
        internal layer names.
        """
        out = self.normalize(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(x))


def build_target() -> ResNet:
    """Factory for the target classifier."""
    return ResNet()
