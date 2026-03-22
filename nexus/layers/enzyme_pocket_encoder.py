"""
PGAEnzymePocketEncoder  +  GatedLoss  —  Layer 5 Core
======================================================
Implements the G(3,0,1) attention-based encoder for the CYP450 active
site and the unified GatedLoss function.

Architecture overview
---------------------
1. Grade-specific feature embedding
     atomic_feats   → grade-1 vectors  (e1, e2, e3, e0)
     side_chain_feats → grade-2 bivectors (e12…e30)
     volume_feats   → grade-3 trivectors (e123…e230)
   yielding drug and protein anchor multivectors in G(3,0,1).

2. Reversed geometric attention
     score(d, p) = ⟨Q_drug[d], K_prot[p]⟩_G
                 = scalar_part(Q_drug[d] × reverse(K_prot[p]))
   The geometric inner product is sensitive to both magnitude *and*
   grade-level alignment (shape-matching), unlike a plain dot product.

3. A_field computation
     A_field = field_net(attention_context)   ← per-drug-atom scalar
     A_field > 0  →  accessible steric region
     A_field < 0  →  hard steric overlap (penalised by GatedLoss)

4. Hamiltonian gating
     gated_mv = drug_mv × sigmoid(A_field)
   When A_field is strongly negative (steric clash) sigmoid ≈ 0 and
   the drug multivector is suppressed; when positive it passes through.

GatedLoss
---------
  L_total = L_affinity + 0.5 · L_causal + 0.1 · L_steric

  L_affinity  MSE(pred_ΔΔG, true_ΔΔG)   — binding affinity regression
  L_causal    BCE(σ(W_pred), W_true)      — metabolic DAG edge prediction
  L_steric    mean(relu(-A_field))        — steric constraint penalty
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nexus.physics.pga_math import GeometricProductLayer


# ---------------------------------------------------------------------------
# GatedLoss
# ---------------------------------------------------------------------------

@dataclass
class GatedLossOutput:
    total: torch.Tensor          # L_total = L_affinity + 0.5·L_causal + 0.1·L_steric
    affinity_loss: torch.Tensor  # MSE between predicted and true ΔΔG
    causal_loss: torch.Tensor    # BCE between predicted W and ground-truth DAG
    steric_loss: torch.Tensor    # mean(relu(-A_field)) steric overlap penalty


class GatedLoss(nn.Module):
    """Unified loss for the PGAEnzymePocketEncoder.

    Weights follow the 2026-standard:
      L_total = L_affinity + 0.5·L_causal + 0.1·L_steric

    Args:
        causal_weight:  Weight on the causal adjacency loss (default 0.5).
        steric_weight:  Weight on the steric constraint loss (default 0.1).
    """

    def __init__(
        self,
        causal_weight: float = 0.5,
        steric_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.causal_weight = float(causal_weight)
        self.steric_weight = float(steric_weight)

    def forward(
        self,
        pred_ddg: torch.Tensor,
        true_ddg: torch.Tensor,
        w_pred: torch.Tensor,
        w_true: torch.Tensor,
        a_field: torch.Tensor,
    ) -> GatedLossOutput:
        """Compute the three-term gated loss.

        Args:
            pred_ddg: (B,) or (B, 1) predicted binding affinity (ΔΔG).
            true_ddg: (B,) or (B, 1) ground-truth binding affinity.
            w_pred:   (B, N, N) predicted causal adjacency weights (raw logits
                      — sigmoid is applied internally for numerical stability).
            w_true:   (B, N, N) ground-truth DAG adjacency (0 or 1).
            a_field:  (...) steric accessibility field values (any shape).

        Returns:
            GatedLossOutput with individual and total losses.
        """
        # L_affinity: binding affinity regression
        l_aff = F.mse_loss(pred_ddg.view(-1), true_ddg.view(-1))

        # L_causal: causal DAG edge prediction
        # binary_cross_entropy_with_logits treats w_pred as raw logits (numerically stable)
        l_cau = F.binary_cross_entropy_with_logits(
            w_pred,
            w_true.to(dtype=w_pred.dtype),
        )

        # L_steric: penalise atoms that have entered hard steric regions (A_field < 0)
        l_ste = F.relu(-a_field).mean()

        total = l_aff + self.causal_weight * l_cau + self.steric_weight * l_ste
        return GatedLossOutput(
            total=total,
            affinity_loss=l_aff,
            causal_loss=l_cau,
            steric_loss=l_ste,
        )


# ---------------------------------------------------------------------------
# EnzymePocketEncoder
# ---------------------------------------------------------------------------

class PGAEnzymePocketEncoder(nn.Module):
    """G(3,0,1) geometric attention encoder for the CYP450 active-site pocket.

    Inputs
    ------
    drug_feat:    (B, N_d, atom_feat_dim)   atomic features of the drug
    drug_sc_feat: (B, N_d, sc_feat_dim)     side-chain / ring features
    drug_vol_feat:(B, N_d, vol_feat_dim)    volumetric density features
    prot_feat:    (B, N_p, atom_feat_dim)   protein anchor atomic features
    prot_sc_feat: (B, N_p, sc_feat_dim)     protein side-chain features
    prot_vol_feat:(B, N_p, vol_feat_dim)    protein volumetric features

    Outputs
    -------
    gated_mv:  (B, N_d, 16)   Hamiltonian-gated drug multivector field
    a_field:   (B, N_d)       accessibility field (A_field) for GatedLoss

    Args:
        atom_feat_dim:   Dimension of input atomic feature vectors (default 64).
        sc_feat_dim:     Side-chain feature dimension (default 32).
        vol_feat_dim:    Volumetric feature dimension (default 16).
        n_heads:         Number of parallel geometric attention heads (default 4).
        attn_dropout:    Attention weight dropout (default 0.1).
        hidden_dim:      Width of the A_field MLP (default 64).
    """

    def __init__(
        self,
        atom_feat_dim: int = 64,
        sc_feat_dim: int = 32,
        vol_feat_dim: int = 16,
        n_heads: int = 4,
        attn_dropout: float = 0.1,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_heads = int(n_heads)

        # ------------------------------------------------------------------
        # Grade-specific feature embedders
        # ------------------------------------------------------------------
        # Grade 1 (vectors): 4 components — e1, e2, e3, e0
        # We embed atomic features into the e0 (projective) slot; the e1-e3
        # slots are typically filled with actual 3D coordinates via embed_point.
        self.grade1_embed = nn.Linear(atom_feat_dim, 4)

        # Grade 2 (bivectors): 6 components — e12, e13, e23, e10, e20, e30
        self.grade2_embed = nn.Linear(sc_feat_dim, 6)

        # Grade 3 (trivectors): 4 components — e123, e120, e130, e230
        self.grade3_embed = nn.Linear(vol_feat_dim, 4)

        # ------------------------------------------------------------------
        # Multi-head geometric attention
        # Each head has its own Q, K, V projections (16 → 16).
        # ------------------------------------------------------------------
        self.q_projs = nn.ModuleList([nn.Linear(16, 16, bias=False) for _ in range(n_heads)])
        self.k_projs = nn.ModuleList([nn.Linear(16, 16, bias=False) for _ in range(n_heads)])
        self.v_projs = nn.ModuleList([nn.Linear(16, 16, bias=False) for _ in range(n_heads)])

        self.geo_product = GeometricProductLayer()

        self.attn_dropout = nn.Dropout(p=attn_dropout)

        # ------------------------------------------------------------------
        # A_field: per-drug-atom scalar accessibility from attention context
        # ------------------------------------------------------------------
        # Input: concatenated head contexts (n_heads × 16)
        self.field_net = nn.Sequential(
            nn.Linear(16 * n_heads, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Output projection back to 16D
        self.out_proj = nn.Linear(16, 16)

    # ------------------------------------------------------------------
    # Grade embedding
    # ------------------------------------------------------------------

    def embed_multivector(
        self,
        atomic_feats: torch.Tensor,    # (B, N, atom_feat_dim)
        sc_feats: torch.Tensor,         # (B, N, sc_feat_dim)
        vol_feats: torch.Tensor,        # (B, N, vol_feat_dim)
        coords: Optional[torch.Tensor] = None,  # (B, N, 3) optional 3D coords
    ) -> torch.Tensor:
        """Embed features into a 16D G(3,0,1) multivector.

        Grade 1 slots [1-4]: learned from atomic_feats; optionally override
        e1/e2/e3 (slots 1-3) with actual Cartesian coordinates.
        Grade 2 slots [5-10]: from side-chain / ring features.
        Grade 3 slots [11-14]: from volumetric density features.
        All other slots (scalar 0, pseudoscalar 15) remain zero.

        Returns (B, N, 16).
        """
        B, N, _ = atomic_feats.shape
        mv = torch.zeros(B, N, 16, dtype=atomic_feats.dtype, device=atomic_feats.device)

        # Grade 1: embed atomic features into all four vector slots
        g1 = self.grade1_embed(atomic_feats)            # (B, N, 4)
        mv[..., 1:5] = g1

        # Override e1/e2/e3 with actual coordinates when available
        if coords is not None:
            mv[..., 1:4] = coords

        # Grade 2: side-chain / aromatic-ring orientations
        mv[..., 5:11] = self.grade2_embed(sc_feats)     # (B, N, 6)

        # Grade 3: pocket volumetric density
        mv[..., 11:15] = self.grade3_embed(vol_feats)   # (B, N, 4)

        return mv

    # ------------------------------------------------------------------
    # Geometric attention
    # ------------------------------------------------------------------

    def _geometric_attn_head(
        self,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        drug_mv: torch.Tensor,   # (B, N_d, 16)
        prot_mv: torch.Tensor,   # (B, N_p, 16)
    ) -> torch.Tensor:
        """One head of reversed geometric attention → (B, N_d, 16)."""
        Q = q_proj(drug_mv)      # (B, N_d, 16)
        K = k_proj(prot_mv)      # (B, N_p, 16)
        V = v_proj(prot_mv)      # (B, N_p, 16)

        # Reverse the key multivectors: reverse(K)
        K_rev = self.geo_product.reverse_multivector(K)   # (B, N_p, 16)

        # Geometric inner product for each (drug, prot) pair:
        # score[b, d, p] = scalar_part(Q[b,d] × K_rev[b,p])
        # Expand for pairwise computation: (B, N_d, N_p, 16) → scalar
        Q_exp = Q.unsqueeze(2)       # (B, N_d,  1, 16)
        K_exp = K_rev.unsqueeze(1)   # (B,  1, N_p, 16)
        scores = self.geo_product(Q_exp, K_exp)[..., 0]   # (B, N_d, N_p)
        scores = scores / (16.0 ** 0.5)   # scale by sqrt(PGA dim)

        weights = torch.softmax(scores, dim=-1)           # (B, N_d, N_p)
        weights = self.attn_dropout(weights)

        # Weighted sum of value multivectors
        return torch.bmm(weights, V)                      # (B, N_d, 16)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        drug_feats: torch.Tensor,       # (B, N_d, atom_feat_dim)
        drug_sc_feats: torch.Tensor,    # (B, N_d, sc_feat_dim)
        drug_vol_feats: torch.Tensor,   # (B, N_d, vol_feat_dim)
        prot_feats: torch.Tensor,       # (B, N_p, atom_feat_dim)
        prot_sc_feats: torch.Tensor,    # (B, N_p, sc_feat_dim)
        prot_vol_feats: torch.Tensor,   # (B, N_p, vol_feat_dim)
        drug_coords: Optional[torch.Tensor] = None,  # (B, N_d, 3)
        prot_coords: Optional[torch.Tensor] = None,  # (B, N_p, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode drug-protein pocket interaction.

        Returns:
            gated_mv: (B, N_d, 16)  Hamiltonian-gated drug multivector field.
            a_field:  (B, N_d)      Accessibility field for GatedLoss.
        """
        # Build 16D multivectors for drug and protein
        drug_mv = self.embed_multivector(
            drug_feats, drug_sc_feats, drug_vol_feats, coords=drug_coords
        )
        prot_mv = self.embed_multivector(
            prot_feats, prot_sc_feats, prot_vol_feats, coords=prot_coords
        )

        # Multi-head geometric attention: each head attends with a different
        # projection, then all head contexts are concatenated.
        head_contexts = [
            self._geometric_attn_head(q, k, v, drug_mv, prot_mv)
            for q, k, v in zip(self.q_projs, self.k_projs, self.v_projs)
        ]                                         # n_heads × (B, N_d, 16)
        concat_ctx = torch.cat(head_contexts, dim=-1)   # (B, N_d, 16·n_heads)

        # A_field: per-drug-atom accessibility scalar
        a_field = self.field_net(concat_ctx).squeeze(-1)    # (B, N_d)

        # Hamiltonian gating: suppress drug multivector in steric-clash regions
        gate = torch.sigmoid(a_field).unsqueeze(-1)         # (B, N_d, 1)
        gated_mv = drug_mv * gate                           # (B, N_d, 16)

        return self.out_proj(gated_mv), a_field


__all__ = ["PGAEnzymePocketEncoder", "GatedLoss", "GatedLossOutput"]
