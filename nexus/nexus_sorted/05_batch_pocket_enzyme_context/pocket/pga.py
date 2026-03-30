from __future__ import annotations

from typing import Optional

import torch

from nexus.physics.pga_math import pga_inner_product as _pga_inner_product


# Public multivector layout for G(3,0,1) — canonical basis shared with
# nexus/physics/pga_math.py:
# [1, e1, e2, e3, e0, e12, e13, e23, e10, e20, e30, e123, e120, e130, e230, e1230]
# Idx  0   1   2   3   4    5    6    7    8    9   10    11    12    13    14    15
#
# Grade-1 (vectors):    indices 1-4  (e1, e2, e3, e0)
# Grade-2 (bivectors):  indices 5-10 (e12, e13, e23 = Euclidean; e10, e20, e30 = ideal)
# Grade-3 (trivectors): indices 11-14
# Grade-4 (pseudo):     index  15
PGA_DIM = 16

VECTOR_SLICE = slice(1, 5)
BIVECTOR_SLICE = slice(5, 11)
TRIVECTOR_SLICE = slice(11, 15)


def _normalize(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)


def embed_point(coords: torch.Tensor) -> torch.Tensor:
    if coords.size(-1) != 3:
        raise ValueError("coords must have trailing dimension 3")
    mv = torch.zeros(coords.shape[:-1] + (PGA_DIM,), dtype=coords.dtype, device=coords.device)
    mv[..., 1:4] = coords   # e1, e2, e3  ←  x, y, z
    mv[..., 4] = 1.0        # e0          ←  homogeneous weight
    return mv


def embed_plane(normal: torch.Tensor, offset: Optional[torch.Tensor] = None) -> torch.Tensor:
    if normal.size(-1) != 3:
        raise ValueError("normal must have trailing dimension 3")
    n = _normalize(normal)
    mv = torch.zeros(normal.shape[:-1] + (PGA_DIM,), dtype=normal.dtype, device=normal.device)
    # Euclidean bivectors encode orientation: n₁·e23 − n₂·e13 + n₃·e12
    mv[..., 5] = n[..., 2]    # e12  ←  nz
    mv[..., 6] = -n[..., 1]   # e13  ←  -ny
    mv[..., 7] = n[..., 0]    # e23  ←  nx
    if offset is not None:
        mv[..., 8] = offset    # e10  ←  plane offset (ideal bivector)
    return mv


def embed_volume(direction: torch.Tensor, magnitude: Optional[torch.Tensor] = None) -> torch.Tensor:
    if direction.size(-1) != 3:
        raise ValueError("direction must have trailing dimension 3")
    d = _normalize(direction)
    mv = torch.zeros(direction.shape[:-1] + (PGA_DIM,), dtype=direction.dtype, device=direction.device)
    # e123 carries the scalar magnitude; e120/e130/e230 carry the direction
    if magnitude is None:
        mv[..., 11] = 1.0
    else:
        mag = magnitude.to(dtype=direction.dtype, device=direction.device)
        expected = mv.shape[:-1]   # leading dims of the multivector
        if mag.shape != expected:
            raise ValueError(
                f"magnitude shape {tuple(mag.shape)} does not match the leading "
                f"dimensions of the multivector {tuple(expected)}. "
                f"Pass magnitude with shape {tuple(expected)}."
            )
        mv[..., 11] = mag
    mv[..., 12] = d[..., 0]   # e120
    mv[..., 13] = d[..., 1]   # e130
    mv[..., 14] = d[..., 2]   # e230
    return mv


def embed_residue_anchor(
    coords: torch.Tensor,
    aromatic_normal: Optional[torch.Tensor] = None,
    cavity_direction: Optional[torch.Tensor] = None,
    cavity_density: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    anchor = embed_point(coords)
    if aromatic_normal is not None:
        anchor = anchor + embed_plane(aromatic_normal)
    if cavity_direction is not None:
        anchor = anchor + embed_volume(cavity_direction, magnitude=cavity_density)
    return anchor


def grade_project(mv: torch.Tensor, grade: int) -> torch.Tensor:
    out = torch.zeros_like(mv)
    if grade == 0:
        out[..., 0] = mv[..., 0]
    elif grade == 1:
        out[..., VECTOR_SLICE] = mv[..., VECTOR_SLICE]
    elif grade == 2:
        out[..., BIVECTOR_SLICE] = mv[..., BIVECTOR_SLICE]
    elif grade == 3:
        out[..., TRIVECTOR_SLICE] = mv[..., TRIVECTOR_SLICE]
    elif grade == 4:
        out[..., 15] = mv[..., 15]
    else:
        raise ValueError("grade must be in {0,1,2,3,4}")
    return out


def geometric_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """True G(3,0,1) inner product: scalar_part(a × reverse(b)).

    Delegates to the Cayley-table implementation in nexus.physics.pga_math
    so that both subsystems use an identical, physically correct algebra.
    """
    if a.size(-1) != PGA_DIM or b.size(-1) != PGA_DIM:
        raise ValueError(f"Expected trailing dimension {PGA_DIM}")
    return _pga_inner_product(a, b)


def pga_norm(mv: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return torch.sqrt((mv * mv).sum(dim=-1).clamp_min(eps))


__all__ = [
    "PGA_DIM",
    "VECTOR_SLICE",
    "BIVECTOR_SLICE",
    "TRIVECTOR_SLICE",
    "embed_point",
    "embed_plane",
    "embed_volume",
    "embed_residue_anchor",
    "grade_project",
    "geometric_inner_product",
    "pga_norm",
]
