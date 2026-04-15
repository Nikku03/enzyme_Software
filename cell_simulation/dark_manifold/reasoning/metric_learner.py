"""
nexus/reasoning/metric_learner.py

Mechanism-aware embedding utilities for analogical retrieval.

Two encoder families live here:

1. MechanismEncoder
   Legacy Euclidean encoder over Morgan fingerprints.  It is kept as a
   reversible fallback while the projected hyperbolic bank is rolled out.

2. HGNNProjection
   Learned projection from continuous multivector features into the Poincare
   ball.  This is the intended bridge between the module-1 continuous field and
   the analogical memory bank.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus.pocket.pga import PGA_DIM


class MechanismEncoder(nn.Module):
    """
    3-layer MLP: 2048 -> 512 -> 256 -> 128, L2-normalised output.

    Dropout(0.10) after the first hidden layer prevents the encoder from
    memorising fingerprint bit positions and forces it to learn functional
    motifs instead.
    """

    def __init__(self, fp_bits: int = 2048, embed_dim: int = 128) -> None:
        super().__init__()
        self.fp_bits = fp_bits
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(fp_bits, 512),
            nn.SiLU(),
            nn.Dropout(p=0.10),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, fp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fp: float/bool tensor [..., fp_bits]
        Returns:
            L2-normalised embedding [..., embed_dim]
        """
        return F.normalize(self.net(fp.float()), p=2, dim=-1)


class PoincareMath:
    """Numerically safe Poincare-ball operations used by HGNNProjection."""

    def __init__(self, c: float = 1.0, eps: float = 1.0e-15) -> None:
        self.c = float(max(c, 1.0e-8))
        self.eps = float(max(eps, 1.0e-15))

    def _sqrt_c(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_tensor(math.sqrt(self.c))

    @staticmethod
    def _safe_norm(x: torch.Tensor, eps: float = 1.0e-15) -> torch.Tensor:
        """||x|| with defined gradient at x=0: sqrt(sum(x^2) + eps^2)."""
        return (x.pow(2).sum(dim=-1, keepdim=True) + eps * eps).sqrt()

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        max_norm = (1.0 - 1.0e-5) / math.sqrt(self.c)
        norm = self._safe_norm(x, self.eps)
        safe = torch.where(norm > max_norm, x * (max_norm / norm), x)
        return torch.nan_to_num(safe, nan=0.0, posinf=max_norm, neginf=-max_norm)

    def exp_map_0(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        v_norm = self._safe_norm(v, self.eps)
        sqrt_c = self._sqrt_c(v)
        scale = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return self._project(scale * v)

    def log_map_0(self, y: torch.Tensor) -> torch.Tensor:
        y = self._project(y.float())
        y_norm = self._safe_norm(y, self.eps)
        sqrt_c = self._sqrt_c(y)
        clipped = (sqrt_c * y_norm).clamp_max(1.0 - 1.0e-6)
        scale = torch.atanh(clipped) / (sqrt_c * y_norm)
        return scale * y

    def mobius_add(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u = self._project(u.float())
        v = self._project(v.float())
        u2 = torch.sum(u * u, dim=-1, keepdim=True)
        v2 = torch.sum(v * v, dim=-1, keepdim=True)
        uv = torch.sum(u * v, dim=-1, keepdim=True)
        c = u.new_tensor(self.c)
        num = (1 + 2 * c * uv + c * v2) * u + (1 - c * u2) * v
        denom = 1 + 2 * c * uv + (c * c) * u2 * v2
        return self._project(num / denom.clamp_min(self.eps))

    def distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        u = self._project(u.float())
        v = self._project(v.float())
        sqdist = (u - v).pow(2).sum(dim=-1, keepdim=True)
        squnorm = u.pow(2).sum(dim=-1, keepdim=True)
        sqvnorm = v.pow(2).sum(dim=-1, keepdim=True)
        c = u.new_tensor(self.c)
        arg = 1.0 + 2.0 * c * sqdist / (
            (1.0 - c * squnorm).clamp_min(self.eps) * (1.0 - c * sqvnorm).clamp_min(self.eps)
        )
        return torch.acosh(arg.clamp_min(1.0 + self.eps)).squeeze(-1)


class HGNNProjection(nn.Module):
    """
    Project continuous multivector features into a graph-level Poincare embedding.

    Input shape:
      - [N, F]     single molecule with N nodes
      - [B, N, F]  batch of molecules

    Output shape:
      - [D] or [B, D] graph embedding strictly inside the Poincare ball
    """

    def __init__(
        self,
        in_channels_16d: int | None = None,
        hidden_dim: int = 256,
        poincare_dim: int = 128,
        c: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.poincare_dim = int(poincare_dim)
        self.math = PoincareMath(c=c)
        first = nn.Linear(int(in_channels_16d), self.hidden_dim) if in_channels_16d is not None else nn.LazyLinear(self.hidden_dim)
        self.tangent_mlp = nn.Sequential(
            first,
            nn.GELU(),
            nn.Dropout(p=float(max(dropout, 0.0))),
            nn.Linear(self.hidden_dim, self.poincare_dim),
        )

    def tangent_encode(self, multivectors: torch.Tensor) -> torch.Tensor:
        x = multivectors.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.tangent_mlp(x)

    def forward(self, q_fp_multivectors: torch.Tensor) -> torch.Tensor:
        tangent = self.tangent_encode(q_fp_multivectors)
        if tangent.ndim == 2:
            pooled = tangent.mean(dim=0)
        elif tangent.ndim >= 3:
            pooled = tangent.mean(dim=-2)
        else:
            pooled = tangent
        return self.math.exp_map_0(pooled)


def _som_class(som_idx: int, mol) -> int:
    """
    Coarse SoM class: returns the atomic number of the SoM atom,
    collapsing atom identity to a rough mechanistic proxy.
    Carbon (6) → aliphatic vs aromatic distinguished by ring membership.
    Falls back to atom index when mol is unavailable.
    """
    try:
        atom = mol.GetAtomWithIdx(som_idx)
        an = atom.GetAtomicNum()
        if an == 6 and atom.GetIsAromatic():
            return 60  # aromatic carbon (separate class)
        return an
    except Exception:
        return -1


def encoder_supervision_loss(
    query_embed: torch.Tensor,
    retrieved_embed: torch.Tensor,
    same_som_class: bool,
    margin_neg: float = 0.10,
) -> torch.Tensor:
    """
    Cosine-margin loss that attracts same-class pairs and repels different-class ones.

    For same-class pairs:   loss = max(0, 1 - cos_sim)  (pull toward cos=1)
    For different-class:    loss = max(0, cos_sim - margin_neg) (push below margin)

    The retrieved embedding is detached so only the query branch receives gradients
    during the analogical retrieval phase (retrieval index is not differentiable).

    Args:
        query_embed:      [embed_dim] L2-normalised query embedding (grad enabled)
        retrieved_embed:  [embed_dim] L2-normalised retrieved embedding
        same_som_class:   True if query and retrieved share the same SoM atom class
        margin_neg:       cosine similarity margin for negative pairs (default 0.10)

    Returns:
        scalar loss tensor
    """
    r = retrieved_embed.detach().float()
    q = query_embed.float()
    cos_sim = (q * r).sum()
    if same_som_class:
        return (1.0 - cos_sim).clamp_min(0.0)
    else:
        return F.relu(cos_sim - margin_neg)


def hyperbolic_supervision_loss(
    query_embed: torch.Tensor,
    retrieved_embed: torch.Tensor,
    same_som_class: bool,
    *,
    margin_neg: float = 1.0,
    curvature: float = 1.0,
) -> torch.Tensor:
    """
    Pairwise supervision directly in hyperbolic space.

    Same-class pairs are pulled together by minimizing geodesic distance.
    Different-class pairs are pushed apart beyond a margin.
    """
    math = PoincareMath(c=curvature)
    q = query_embed.float()
    r = retrieved_embed.detach().float()
    dist = math.distance(q, r)
    if same_som_class:
        return dist.mean()
    return F.relu(margin_neg - dist).mean()


def mechanism_contrastive_loss(
    query_multivectors: torch.Tensor,
    retrieved_multivectors: torch.Tensor,
    query_targets: torch.Tensor,
    retrieved_targets: torch.Tensor,
    query_mask: torch.Tensor,
    retrieved_mask: torch.Tensor,
    *,
    margin: float = 1.25,
    query_confidence: float = 1.0,
    retrieved_confidence: float = 1.0,
) -> torch.Tensor:
    """
    Node-level mechanism contrastive loss on continuous multivectors.

    Each labeled query atom is compared against each labeled retrieved atom.
    Pairs that share at least one mechanism class are pulled together; pairs
    with disjoint mechanism labels are pushed apart up to a margin.

    The retrieved multivectors are detached because retrieval is non-differentiable
    and only the live query encoder should receive gradients during training.
    """
    def _canonicalize_multivectors(multivectors: torch.Tensor) -> torch.Tensor:
        mv = torch.as_tensor(multivectors, dtype=torch.float32)
        if mv.ndim != 2:
            raise ValueError("mechanism_contrastive_loss expects [N, D] multivector tensors")
        if mv.size(-1) == PGA_DIM:
            return mv
        if mv.size(-1) != 8:
            raise ValueError(
                f"mechanism_contrastive_loss expects trailing multivector dimension 8 or {PGA_DIM}, got {mv.size(-1)}"
            )
        out = torch.zeros(mv.shape[:-1] + (PGA_DIM,), dtype=mv.dtype, device=mv.device)
        out[..., 0] = mv[..., 0]
        out[..., 1:4] = mv[..., 1:4]
        out[..., 5] = mv[..., 4]
        out[..., 6] = mv[..., 6]
        out[..., 7] = mv[..., 5]
        out[..., 11] = mv[..., 7]
        return out

    q_mv = _canonicalize_multivectors(query_multivectors)
    r_mv = _canonicalize_multivectors(retrieved_multivectors).detach()
    q_target = torch.as_tensor(query_targets, dtype=torch.float32, device=q_mv.device)
    r_target = torch.as_tensor(retrieved_targets, dtype=torch.float32, device=q_mv.device)
    q_mask = torch.as_tensor(query_mask, dtype=torch.float32, device=q_mv.device)
    r_mask = torch.as_tensor(retrieved_mask, dtype=torch.float32, device=q_mv.device)

    if q_target.ndim != 2 or r_target.ndim != 2:
        raise ValueError("mechanism_contrastive_loss expects [N, C] target tensors")

    q_valid = (q_mask.sum(dim=-1) > 0.0)
    r_valid = (r_mask.sum(dim=-1) > 0.0)
    if not bool(q_valid.any().item()) or not bool(r_valid.any().item()):
        return q_mv.new_zeros(())

    q_vec = F.normalize(q_mv[q_valid], p=2, dim=-1)
    r_vec = F.normalize(r_mv[r_valid], p=2, dim=-1)
    q_lbl = q_target[q_valid]
    r_lbl = r_target[r_valid]

    overlap = torch.matmul(q_lbl, r_lbl.transpose(0, 1))
    positive_pairs = overlap > 0.0
    negative_pairs = ~positive_pairs
    distances = torch.cdist(q_vec, r_vec, p=2)

    losses = []
    if bool(positive_pairs.any().item()):
        pull_loss = distances[positive_pairs].pow(2).mean()
        losses.append(pull_loss)
    if bool(negative_pairs.any().item()):
        push_loss = F.relu(float(margin) - distances[negative_pairs]).pow(2).mean()
        losses.append(0.5 * push_loss)
    if not losses:
        return q_mv.new_zeros(())

    confidence_scale = math.sqrt(
        max(float(query_confidence), 0.0) * max(float(retrieved_confidence), 0.0)
    )
    return sum(losses) * q_mv.new_tensor(confidence_scale, dtype=torch.float32)
