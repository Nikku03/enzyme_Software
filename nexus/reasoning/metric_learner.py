"""
nexus/reasoning/metric_learner.py

Mechanism-aware embedding for analogical retrieval.

MechanismEncoder maps ECFP4 fingerprints (2048-bit) to a 128-dim L2-normalised
embedding space that is trained to pull molecules with the same SoM atom class
together and push mechanistically-different ones apart.

This addresses the "magic methyl" failure mode: two molecules can share a scaffold
(high Tanimoto) but differ in their metabolic site because one has a methyl group
blocking the reactive carbon.  ECFP4 Tanimoto cannot see this; the learned
MechanismEncoder can, because it is supervised by SoM class co-occurrence.

Usage
-----
    enc = MechanismEncoder().to(device)
    q_embed = enc(q_fp)           # [128] L2-normalised
    r_embed = enc(r_fp)           # [128] L2-normalised
    loss = encoder_supervision_loss(q_embed, r_embed, same_som_class=True)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
