from __future__ import annotations

import math

import torch

from .clifford_math import clifford_geometric_product


def clifford_commutator(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return clifford_geometric_product(a, b) - clifford_geometric_product(b, a)


def clifford_identity_like(x: torch.Tensor) -> torch.Tensor:
    ident = torch.zeros_like(x)
    ident[..., 0] = 1.0
    return ident


def clifford_exp(x: torch.Tensor, terms: int = 10) -> torch.Tensor:
    result = clifford_identity_like(x)
    running = clifford_identity_like(x)
    factorial = 1.0
    for n in range(1, max(int(terms), 1) + 1):
        running = clifford_geometric_product(running, x)
        factorial *= float(n)
        result = result + running / factorial
    return result


def dexp_inv(u: torch.Tensor, w: torch.Tensor, order: int = 4) -> torch.Tensor:
    order = int(order)
    out = w
    if order >= 1:
        uw = clifford_commutator(u, w)
        out = out - 0.5 * uw
    if order >= 2:
        uuw = clifford_commutator(u, uw)
        out = out + (1.0 / 12.0) * uuw
    if order >= 4:
        u4w = clifford_commutator(u, clifford_commutator(u, clifford_commutator(u, uuw)))
        out = out - (1.0 / 720.0) * u4w
    return out


__all__ = ["clifford_commutator", "clifford_exp", "clifford_identity_like", "dexp_inv"]
