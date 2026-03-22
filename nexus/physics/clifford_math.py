from __future__ import annotations

from functools import lru_cache

import torch


# Public multivector layout:
# [1, e1, e2, e3, e12, e23, e31, e123]
# Internal layout used for the algebra:
# [1, e1, e2, e3, e12, e23, e13, e123]

_INTERNAL_MASKS = (0, 1, 2, 4, 3, 6, 5, 7)


def _public_to_internal(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 6] = -y[..., 6]
    return y


def _internal_to_public(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 6] = -y[..., 6]
    return y


def _blade_sign(mask_a: int, mask_b: int) -> int:
    sign = 1
    for bit in range(3):
        if (mask_a >> bit) & 1:
            lower = mask_b & ((1 << bit) - 1)
            if bin(lower).count("1") % 2 == 1:
                sign = -sign
    return sign


@lru_cache(maxsize=1)
def _multiplication_table() -> tuple[list[list[int]], list[list[int]]]:
    sign = [[0 for _ in range(8)] for _ in range(8)]
    index = [[0 for _ in range(8)] for _ in range(8)]
    mask_to_idx = {mask: idx for idx, mask in enumerate(_INTERNAL_MASKS)}
    for i, mask_a in enumerate(_INTERNAL_MASKS):
        for j, mask_b in enumerate(_INTERNAL_MASKS):
            result_mask = mask_a ^ mask_b
            index[i][j] = mask_to_idx[result_mask]
            sign[i][j] = _blade_sign(mask_a, mask_b)
    return sign, index


def _multiplication_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sign, index = _multiplication_table()
    table = torch.zeros(8, 8, 8, dtype=dtype, device=device)
    for i in range(8):
        for j in range(8):
            table[index[i][j], i, j] = sign[i][j]
    return table


def clifford_geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.size(-1) != 8 or b.size(-1) != 8:
        raise ValueError("Clifford multivectors must have trailing dimension 8")
    a_internal = _public_to_internal(a)
    b_internal = _public_to_internal(b)
    dtype = torch.promote_types(a_internal.dtype, b_internal.dtype)
    table = _multiplication_tensor(a_internal.device, dtype)
    out = torch.einsum(
        "...i,...j,kij->...k",
        a_internal.to(dtype=dtype),
        b_internal.to(dtype=dtype),
        table,
    )
    return _internal_to_public(out)


def embed_coordinates(r: torch.Tensor) -> torch.Tensor:
    if r.size(-1) != 3:
        raise ValueError("Coordinates must have trailing dimension 3")
    mv = torch.zeros(r.shape[:-1] + (8,), dtype=r.dtype, device=r.device)
    mv[..., 1:4] = r
    return mv


# Lie algebra operators (clifford_commutator, clifford_exp, dexp_inv) live in
# nexus/physics/lie_algebra.py — that is the single canonical source.
# Import them from nexus.physics directly; do not import from this module.

__all__ = [
    "clifford_geometric_product",
    "embed_coordinates",
]
