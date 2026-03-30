from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .encoder import PocketEncoderOutput
from ..physics.pga_math import pga_geometric_product, pga_reverse
from .pga import PGA_DIM


@dataclass
class ReversedGeometricAttentionOutput:
    attended_drug: torch.Tensor
    pocket_context: torch.Tensor
    attention_weights: torch.Tensor
    residue_importance: torch.Tensor
    accessibility_mask: torch.Tensor
    logits: torch.Tensor


class ReversedGeometricAttention(nn.Module):
    def __init__(
        self,
        drug_dim: int = 8,
        pocket_dim: int = PGA_DIM,
        heads: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.drug_dim = int(drug_dim)
        self.pocket_dim = int(pocket_dim)
        self.heads = int(heads)
        self.hidden_dim = int(hidden_dim)

        self.drug_proj = nn.Linear(self.drug_dim, self.pocket_dim)
        self.query_proj = nn.Linear(self.pocket_dim, self.heads * self.pocket_dim)
        self.key_proj = nn.Linear(self.pocket_dim, self.heads * self.pocket_dim)
        self.value_proj = nn.Linear(self.pocket_dim, self.heads * self.hidden_dim)
        self.out_proj = nn.Linear(self.heads * self.hidden_dim, self.pocket_dim)

    def _ensure_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 2:
            return x.unsqueeze(0), True
        return x, False

    def forward(
        self,
        drug_multivectors: torch.Tensor,
        pocket: PocketEncoderOutput,
        drug_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> ReversedGeometricAttentionOutput:
        drug_mv, squeeze = self._ensure_batch(drug_multivectors)
        pocket_mv, _ = self._ensure_batch(pocket.attention_anchors)

        drug_mv = self.drug_proj(drug_mv)
        if drug_mask is None:
            drug_mask = torch.ones(drug_mv.shape[:2], dtype=torch.bool, device=drug_mv.device)
        if residue_mask is None:
            residue_mask = torch.ones(pocket_mv.shape[:2], dtype=torch.bool, device=pocket_mv.device)

        # Q and K are per-head 16D PGA multivectors — each head learns a distinct
        # geometric projection of the drug/pocket into G(3,0,1) space.
        # V stays in hidden_dim for rich aggregation (out_proj maps it back to pocket_dim).
        q = self.query_proj(drug_mv).view(drug_mv.size(0), drug_mv.size(1), self.heads, self.pocket_dim)
        k = self.key_proj(pocket_mv).view(pocket_mv.size(0), pocket_mv.size(1), self.heads, self.pocket_dim)
        v = self.value_proj(pocket_mv).view(pocket_mv.size(0), pocket_mv.size(1), self.heads, self.hidden_dim)

        # True reversed geometric attention: score = scalar_part(Q × reverse(K))
        # k_rev: [B, N_p, H, 16];  expand to [B, N_d, N_p, H, 16] for pairwise product
        N_d, N_p = drug_mv.size(1), pocket_mv.size(1)
        k_rev = pga_reverse(k)
        logits = pga_geometric_product(
            q.unsqueeze(2).expand(-1, -1, N_p, -1, -1),
            k_rev.unsqueeze(1).expand(-1, N_d, -1, -1, -1),
        )[..., 0] / (self.pocket_dim ** 0.5)

        mask = drug_mask.unsqueeze(-1).unsqueeze(-1) & residue_mask.unsqueeze(1).unsqueeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(logits, dim=2)
        attended = torch.einsum("bdph,bphf->bdhf", attention_weights, v)
        attended_flat = attended.reshape(drug_mv.size(0), drug_mv.size(1), self.heads * self.hidden_dim)
        attended_drug = self.out_proj(attended_flat)

        residue_importance = attention_weights.mean(dim=(1, 3))
        accessibility_mask = attention_weights.mean(dim=-1).max(dim=2).values
        # Global pocket context: importance-weighted pool of protein value features.
        # residue_importance: [B, N_prot] — global, drug-atom-agnostic importance score.
        # Weighted sum across residues → [B, heads*hidden_dim] → project to [B, pocket_dim].
        v_flat = v.reshape(v.size(0), v.size(1), self.heads * self.hidden_dim)
        pocket_context = self.out_proj(
            torch.einsum("bp,bpf->bf", residue_importance, v_flat)
        )

        if squeeze:
            attended_drug = attended_drug.squeeze(0)
            pocket_context = pocket_context.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
            residue_importance = residue_importance.squeeze(0)
            accessibility_mask = accessibility_mask.squeeze(0)
            logits = logits.squeeze(0)
        return ReversedGeometricAttentionOutput(
            attended_drug=attended_drug,
            pocket_context=pocket_context,
            attention_weights=attention_weights,
            residue_importance=residue_importance,
            accessibility_mask=accessibility_mask,
            logits=logits,
        )


__all__ = ["ReversedGeometricAttention", "ReversedGeometricAttentionOutput"]
