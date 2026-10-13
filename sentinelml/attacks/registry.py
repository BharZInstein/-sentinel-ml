from __future__ import annotations

from typing import Callable, Dict

import torchattacks

import config
from sentinelml.attacks.custom import PatchAttack, SpatialAttack


def _fgsm(model, eps):
    return torchattacks.FGSM(model, eps=eps)


def _pgd(model, eps):
    return torchattacks.PGD(model, eps=eps, alpha=config.PGD_ALPHA, steps=config.PGD_STEPS)


def _cw(model):


    return torchattacks.CW(model, c=1.0, steps=100, lr=0.01)


def _square(model):


    return torchattacks.Square(model, norm="Linf", eps=8 / 255, n_queries=1000, seed=config.SEED)


def _apgd(model, loss):
    return torchattacks.APGD(model, norm="Linf", eps=config.APGD_EPS, steps=config.APGD_STEPS,
                             loss=loss, seed=config.SEED)


def _deepfool(model):
    return torchattacks.DeepFool(model, steps=config.DEEPFOOL_STEPS, overshoot=config.DEEPFOOL_OVERSHOOT)


def _onepixel(model):
    return torchattacks.OnePixel(model, pixels=config.ONEPIXEL_PIXELS, steps=config.ONEPIXEL_STEPS,
                                 popsize=config.ONEPIXEL_POPSIZE)


ATTACK_FACTORIES: Dict[str, Callable] = {
    "fgsm_4": lambda m: _fgsm(m, config.FGSM_EPSILONS[0]),
    "fgsm_8": lambda m: _fgsm(m, config.FGSM_EPSILONS[1]),
    "pgd_4": lambda m: _pgd(m, config.PGD_EPSILONS[0]),
    "pgd_8": lambda m: _pgd(m, config.PGD_EPSILONS[1]),
    "cw": _cw,
    "square": _square,
    "apgd_ce": lambda m: _apgd(m, "ce"),
    "apgd_dlr": lambda m: _apgd(m, "dlr"),
    "deepfool": _deepfool,
    "onepixel": _onepixel,
    "patch": PatchAttack,
    "spatial": SpatialAttack,
}


POOL_SIZE = {name: config.ATTACK_POOL_SIZE for name in ATTACK_FACTORIES}
POOL_SIZE["cw"] = config.CW_POOL_SIZE
POOL_SIZE["square"] = config.SQUARE_POOL_SIZE
POOL_SIZE["deepfool"] = config.DEEPFOOL_POOL_SIZE
POOL_SIZE["onepixel"] = config.ONEPIXEL_POOL_SIZE
POOL_SIZE["spatial"] = config.SPATIAL_POOL_SIZE


def build_attack(name: str, model):

    if name not in ATTACK_FACTORIES:
        raise KeyError(f"unknown attack '{name}'; known: {list(ATTACK_FACTORIES)}")
    return ATTACK_FACTORIES[name](model)
