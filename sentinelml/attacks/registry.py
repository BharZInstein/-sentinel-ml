from __future__ import annotations

from typing import Callable, Dict

import torchattacks

import config


def _fgsm(model, eps):
    return torchattacks.FGSM(model, eps=eps)


def _pgd(model, eps):
    return torchattacks.PGD(model, eps=eps, alpha=config.PGD_ALPHA, steps=config.PGD_STEPS)


def _cw(model):


    return torchattacks.CW(model, c=1.0, steps=100, lr=0.01)


def _square(model):


    return torchattacks.Square(model, norm="Linf", eps=8 / 255, n_queries=1000, seed=config.SEED)


ATTACK_FACTORIES: Dict[str, Callable] = {
    "fgsm_4": lambda m: _fgsm(m, config.FGSM_EPSILONS[0]),
    "fgsm_8": lambda m: _fgsm(m, config.FGSM_EPSILONS[1]),
    "pgd_4": lambda m: _pgd(m, config.PGD_EPSILONS[0]),
    "pgd_8": lambda m: _pgd(m, config.PGD_EPSILONS[1]),
    "cw": _cw,
    "square": _square,
}


POOL_SIZE = {name: config.ATTACK_POOL_SIZE for name in ATTACK_FACTORIES}
POOL_SIZE["cw"] = config.CW_POOL_SIZE
POOL_SIZE["square"] = config.SQUARE_POOL_SIZE


def build_attack(name: str, model):

    if name not in ATTACK_FACTORIES:
        raise KeyError(f"unknown attack '{name}'; known: {list(ATTACK_FACTORIES)}")
    return ATTACK_FACTORIES[name](model)
