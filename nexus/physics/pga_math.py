"""
G(3,0,1) Projective Geometric Algebra
=======================================
16-dimensional algebra for unified rigid-body geometry.

Metric signature:
  e1² = e2² = e3² = +1   (Euclidean directions)
  e0² =  0               (projective / ideal dimension — encodes translations)

User basis layout (16 components):
  ┌──────────────────────────────────────────────────────────────────┐
  │ Idx │ Basis  │ Grade │ Geometric meaning                         │
  ├─────┼────────┼───────┼───────────────────────────────────────────┤
  │  0  │ 1      │  0    │ scalar                                    │
  │  1  │ e1     │  1    │ atomic position x / Heme Fe x             │
  │  2  │ e2     │  1    │ atomic position y                         │
  │  3  │ e3     │  1    │ atomic position z                         │
  │  4  │ e0     │  1    │ homogeneous / projective coordinate        │
  │  5  │ e12    │  2    │ rotation in x-y plane / aromatic ring     │
  │  6  │ e13    │  2    │ rotation in x-z plane                     │
  │  7  │ e23    │  2    │ rotation in y-z plane                     │
  │  8  │ e10    │  2    │ translation along x (ideal line)          │
  │  9  │ e20    │  2    │ translation along y                       │
  │ 10  │ e30    │  2    │ translation along z                       │
  │ 11  │ e123   │  3    │ Euclidean pseudoscalar / volume element   │
  │ 12  │ e120   │  3    │ pocket-volume density component            │
  │ 13  │ e130   │  3    │ pocket-volume density component            │
  │ 14  │ e230   │  3    │ pocket-volume density component            │
  │ 15  │ e1230  │  4    │ projective pseudoscalar                   │
  └─────┴────────┴───────┴───────────────────────────────────────────┘

Non-canonical elements and their canonical equivalents:
  e10 = e1·e0 = -e0·e1 = -e01       → stored with sign_correction = -1
  e20 = e2·e0 = -e02                 → sign_correction = -1
  e30 = e3·e0 = -e03                 → sign_correction = -1
  e120 = e1·e2·e0 = +e012            → sign_correction = +1
  e130 = e1·e3·e0 = +e013            → sign_correction = +1
  e230 = e2·e3·e0 = +e023            → sign_correction = +1
  e1230 = e1·e2·e3·e0 = -e0123       → sign_correction = -1

Spin Group (motors):
  Rotation motor : M_rot = cos(θ/2) - sin(θ/2)·(n1·e23 - n2·e13 + n3·e12)
  Translation    : T     = 1 + (1/2)·(dx·e10 + dy·e20 + dz·e30)
  Combined motor : M = T · M_rot   (translate after rotating)
  Sandwich action: X' = M · X · M̃   where M̃ = reverse(M)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Basis definition
# ---------------------------------------------------------------------------

# Each entry: (canonical_bitmask, sign_correction)
# Canonical ordering: e0 < e1 < e2 < e3  (bits 0,1,2,3)
# sign_correction s means: user_basis_i = s × canonical_blade_with_same_mask
_USER_BASIS: list[tuple[int, int]] = [
    (0b0000,  1),   #  0: 1
    (0b0010,  1),   #  1: e1
    (0b0100,  1),   #  2: e2
    (0b1000,  1),   #  3: e3
    (0b0001,  1),   #  4: e0
    (0b0110,  1),   #  5: e12   = +e12  (canonical)
    (0b1010,  1),   #  6: e13   = +e13
    (0b1100,  1),   #  7: e23   = +e23
    (0b0011, -1),   #  8: e10   = -e01  (= e1·e0)
    (0b0101, -1),   #  9: e20   = -e02
    (0b1001, -1),   # 10: e30   = -e03
    (0b1110,  1),   # 11: e123  = +e123
    (0b0111,  1),   # 12: e120  = +e012  (e1·e2·e0 → 2 swaps → +)
    (0b1011,  1),   # 13: e130  = +e013
    (0b1101,  1),   # 14: e230  = +e023
    (0b1111, -1),   # 15: e1230 = -e0123 (e1·e2·e3·e0 → 3 swaps → -)
]

# Reverse lookup: canonical bitmask → user basis index
_MASK_TO_IDX: dict[int, int] = {mask: idx for idx, (mask, _) in enumerate(_USER_BASIS)}

# Grade of each user basis element (for reverse and grade-projection utilities)
_GRADES = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4]

# Sign for the reverse operation: reverse(grade-r blade) = (-1)^(r(r-1)/2)
# r=0→+1, r=1→+1, r=2→-1, r=3→-1, r=4→+1
_REVERSE_SIGN_BY_GRADE = {0: 1, 1: 1, 2: -1, 3: -1, 4: 1}
_REVERSE_SIGNS_LIST: list[float] = [
    float(_REVERSE_SIGN_BY_GRADE[g]) for g in _GRADES
]


# ---------------------------------------------------------------------------
# Cayley table construction
# ---------------------------------------------------------------------------

def _blade_sign_pga(mask_a: int, mask_b: int) -> int:
    """Anticommutation sign of canonical_blade(mask_a) × canonical_blade(mask_b).

    Counts, for each basis vector in mask_a, how many lower-indexed
    basis vectors from mask_b must be passed over (each swap → sign flip).
    The metric for G(3,0,1) contributes +1 for shared e1/e2/e3 bits and 0
    for shared e0 bits (null); the null case is handled before calling this.
    """
    sign = 1
    for bit in range(4):
        if (mask_a >> bit) & 1:
            lower = mask_b & ((1 << bit) - 1)
            if bin(lower).count("1") % 2 == 1:
                sign = -sign
    return sign


@lru_cache(maxsize=1)
def _build_pga_cayley_table() -> list[list[list[float]]]:
    """Build the 16×16×16 Cayley table for G(3,0,1) in user basis layout.

    table[k][i][j] = coefficient of user_basis_k in (user_basis_i × user_basis_j).
    All values are in {-1, 0, +1}.
    """
    table: list[list[list[float]]] = [
        [[0.0] * 16 for _ in range(16)] for _ in range(16)
    ]
    for i, (mask_i, sign_i) in enumerate(_USER_BASIS):
        for j, (mask_j, sign_j) in enumerate(_USER_BASIS):
            shared = mask_i & mask_j
            # e0 appears in both → null product (e0² = 0)
            if shared & 0b0001:
                continue
            result_mask = mask_i ^ mask_j
            anticomm = _blade_sign_pga(mask_i, mask_j)
            k = _MASK_TO_IDX[result_mask]
            _, sign_k = _USER_BASIS[k]
            # user_i × user_j = sign_i·canon_i × sign_j·canon_j
            #                 = sign_i·sign_j·anticomm · canon_k
            #                 = sign_i·sign_j·anticomm·sign_k · user_k
            table[k][i][j] = float(sign_i * sign_j * anticomm * sign_k)
    return table


# Cache table tensors by (str(device), str(dtype)) so .to(device) works correctly
_TABLE_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def _pga_multiplication_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), str(dtype))
    if key not in _TABLE_CACHE:
        raw = _build_pga_cayley_table()
        t = torch.tensor(raw, dtype=dtype, device=device)
        _TABLE_CACHE[key] = t
    return _TABLE_CACHE[key]


# ---------------------------------------------------------------------------
# Core algebraic operations
# ---------------------------------------------------------------------------

def pga_geometric_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geometric product C = A × B in G(3,0,1).

    Args:
        a, b: (..., 16) multivectors in user basis layout.

    Returns:
        (..., 16) product in the same layout.
    """
    if a.size(-1) != 16 or b.size(-1) != 16:
        raise ValueError("PGA multivectors must have trailing dimension 16")
    dtype = torch.promote_types(a.dtype, b.dtype)
    table = _pga_multiplication_tensor(a.device, dtype)
    return torch.einsum(
        "...i,...j,kij->...k",
        a.to(dtype=dtype),
        b.to(dtype=dtype),
        table,
    )


def pga_reverse(mv: torch.Tensor) -> torch.Tensor:
    """Reverse of a multivector: negate grade-2 and grade-3 components.

    reverse(grade-r blade) = (-1)^(r(r-1)/2) × blade
      r=0,1 → +1  |  r=2,3 → -1  |  r=4 → +1
    """
    signs = torch.tensor(
        _REVERSE_SIGNS_LIST, dtype=mv.dtype, device=mv.device
    )
    return mv * signs


def pga_sandwich(motor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Apply rigid transform: motor × x × reverse(motor).

    Used to rotate/translate any geometric object x (point, line, plane).

    Args:
        motor: (..., 16) — even-grade multivector from the Spin group.
        x:     (..., 16) — object to transform (must broadcast with motor).

    Returns:
        (..., 16) transformed object.
    """
    return pga_geometric_product(
        pga_geometric_product(motor, x),
        pga_reverse(motor),
    )


def pga_inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Geometric inner product: scalar part of A × reverse(B).

    ⟨A, B⟩_G = (A × B̃)[0]

    This is the attention score used by PGAEnzymePocketEncoder: it is sensitive
    to both the magnitude of the multivectors *and* their relative orientation
    (unlike the Euclidean dot product which ignores grade interactions).

    Args:
        a, b: (..., 16) multivectors.

    Returns:
        (...,) scalar — one score per leading-dimension element.
    """
    b_rev = pga_reverse(b)
    product = pga_geometric_product(a, b_rev)
    return product[..., 0]   # index 0 = scalar grade


# ---------------------------------------------------------------------------
# Grade-specific embedding utilities
# ---------------------------------------------------------------------------

def embed_point(xyz: torch.Tensor, w: float = 1.0) -> torch.Tensor:
    """Embed 3D coordinates as a grade-1 PGA vector (homogeneous point).

    In PGA a finite point P = xe1 + ye2 + ze3 + we0  (w ≠ 0 for finite points).
    Indices: e1=1, e2=2, e3=3, e0=4.

    Args:
        xyz: (..., 3) Cartesian coordinates.
        w:   homogeneous weight (default 1.0).

    Returns:
        (..., 16) multivector with only grade-1 components set.
    """
    if xyz.size(-1) != 3:
        raise ValueError("xyz must have trailing dimension 3")
    mv = torch.zeros(xyz.shape[:-1] + (16,), dtype=xyz.dtype, device=xyz.device)
    mv[..., 1:4] = xyz          # e1, e2, e3 ← x, y, z
    mv[..., 4] = w              # e0 ← homogeneous weight
    return mv


def embed_bivector(data: torch.Tensor) -> torch.Tensor:
    """Embed a 6-dim feature vector into the grade-2 PGA bivector components.

    Grade-2 indices:  e12=5, e13=6, e23=7, e10=8, e20=9, e30=10.
    The first 3 (Euclidean bivectors) encode rotational orientation;
    the last 3 (ideal bivectors) encode translational direction.

    Args:
        data: (..., 6) features.

    Returns:
        (..., 16) multivector with only grade-2 components set.
    """
    if data.size(-1) != 6:
        raise ValueError("bivector data must have trailing dimension 6")
    mv = torch.zeros(data.shape[:-1] + (16,), dtype=data.dtype, device=data.device)
    mv[..., 5:11] = data        # indices 5-10
    return mv


def embed_trivector(data: torch.Tensor) -> torch.Tensor:
    """Embed a 4-dim feature vector into the grade-3 PGA trivector components.

    Grade-3 indices: e123=11, e120=12, e130=13, e230=14.
    Trivectors represent volumetric (plane-like) elements in PGA.

    Args:
        data: (..., 4) features.

    Returns:
        (..., 16) multivector with only grade-3 components set.
    """
    if data.size(-1) != 4:
        raise ValueError("trivector data must have trailing dimension 4")
    mv = torch.zeros(data.shape[:-1] + (16,), dtype=data.dtype, device=data.device)
    mv[..., 11:15] = data       # indices 11-14
    return mv


# ---------------------------------------------------------------------------
# Spin group motor constructors
# ---------------------------------------------------------------------------

def make_rotor(
    axis: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Construct a pure rotation motor for G(3,0,1).

    M_rot = cos(θ/2) - sin(θ/2)·(n1·e23 - n2·e13 + n3·e12)

    The sandwich M_rot × P × M̃_rot rotates point P by angle θ
    counter-clockwise (right-hand rule) around the axis n through the origin.

    Args:
        axis:  (..., 3) unit rotation axis (n1, n2, n3).
        angle: (...,) rotation angles in radians.

    Returns:
        (..., 16) even-grade motor.
    """
    half = angle / 2.0
    c = torch.cos(half)          # (...,)
    s = torch.sin(half)          # (...,)
    motor = torch.zeros(axis.shape[:-1] + (16,), dtype=axis.dtype, device=axis.device)
    motor[..., 0]  = c                    # scalar
    motor[..., 7]  = -s * axis[..., 0]   # e23 ← -n1  (CCW right-hand rule)
    motor[..., 6]  =  s * axis[..., 1]   # e13 ← +n2
    motor[..., 5]  = -s * axis[..., 2]   # e12 ← -n3
    return motor


def make_translator(displacement: torch.Tensor) -> torch.Tensor:
    """Construct a pure translation motor for G(3,0,1).

    T = 1 + (1/2)·(dx·e10 + dy·e20 + dz·e30)

    The sandwich T × P × T̃ translates point P by vector (dx, dy, dz).

    Note on the origin: a point embedded as a pure e0 vector (i.e., with
    x=y=z=0 and w=1 via embed_point([0,0,0])) is a mathematical fixed point
    of every translator in G(3,0,1) because e0² = 0.  This is geometrically
    correct — the ideal element at infinity is invariant under translation.
    All finite atomic positions (non-zero x,y,z) translate correctly.

    Args:
        displacement: (..., 3) translation vector (dx, dy, dz).

    Returns:
        (..., 16) even-grade motor.
    """
    motor = torch.zeros(
        displacement.shape[:-1] + (16,),
        dtype=displacement.dtype,
        device=displacement.device,
    )
    motor[..., 0]  = 1.0
    motor[..., 8]  = displacement[..., 0] / 2.0   # e10 ← dx/2
    motor[..., 9]  = displacement[..., 1] / 2.0   # e20 ← dy/2
    motor[..., 10] = displacement[..., 2] / 2.0   # e30 ← dz/2
    return motor


# ---------------------------------------------------------------------------
# GeometricProductLayer — nn.Module wrapper
# ---------------------------------------------------------------------------

class GeometricProductLayer(nn.Module):
    """Learnable G(3,0,1) geometric product layer.

    Registers the Cayley table as a persistent buffer so it migrates
    correctly with .to(device) / .to(dtype) calls.

    Exposes:
      forward(mv_a, mv_b)         → geometric product
      geometric_inner_product(Q, K) → attention score ⟨Q, K⟩_G
      reverse_multivector(mv)    → grade-flipped reverse
    """

    def __init__(self) -> None:
        super().__init__()
        raw = _build_pga_cayley_table()
        cayley = torch.tensor(raw, dtype=torch.float32)   # [16, 16, 16]
        self.register_buffer("cayley_table", cayley)

    def forward(self, mv_a: torch.Tensor, mv_b: torch.Tensor) -> torch.Tensor:
        """Geometric product C = mv_a × mv_b.

        Args:
            mv_a, mv_b: (..., 16) multivectors.

        Returns:
            (..., 16) product.
        """
        dtype = torch.promote_types(mv_a.dtype, mv_b.dtype)
        table = self.cayley_table.to(dtype=dtype)
        return torch.einsum(
            "...i,...j,kij->...k",
            mv_a.to(dtype=dtype),
            mv_b.to(dtype=dtype),
            table,
        )

    def reverse_multivector(self, mv: torch.Tensor) -> torch.Tensor:
        """Reverse: negate grade-2 and grade-3 components."""
        signs = torch.tensor(
            _REVERSE_SIGNS_LIST, dtype=mv.dtype, device=mv.device
        )
        return mv * signs

    def geometric_inner_product(
        self,
        q_drug: torch.Tensor,
        k_prot: torch.Tensor,
    ) -> torch.Tensor:
        """Attention score ⟨Q_drug, K_prot⟩_G = scalar_part(Q × reverse(K)).

        Args:
            q_drug: (..., 16) drug query multivectors.
            k_prot: (..., 16) protein key multivectors.

        Returns:
            (...,) scalar attention logits.
        """
        k_rev = self.reverse_multivector(k_prot)
        return self.forward(q_drug, k_rev)[..., 0]


__all__ = [
    "pga_geometric_product",
    "pga_reverse",
    "pga_sandwich",
    "pga_inner_product",
    "embed_point",
    "embed_bivector",
    "embed_trivector",
    "make_rotor",
    "make_translator",
    "GeometricProductLayer",
]
