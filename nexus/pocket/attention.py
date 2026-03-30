from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import PocketEncoderOutput
from ..physics.pga_math import pga_geometric_product
from .pga import PGA_DIM, geometric_inner_product


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
        drug_dim: int = PGA_DIM,
        pocket_dim: int = PGA_DIM,
        heads: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.drug_dim = int(drug_dim)
        self.pocket_dim = int(pocket_dim)
        self.heads = int(heads)
        self.hidden_dim = int(hidden_dim)

        self.q_transform = nn.Parameter(torch.empty(self.heads, self.pocket_dim))
        self.k_transform = nn.Parameter(torch.empty(self.heads, self.pocket_dim))
        self.value_proj = nn.Sequential(
            nn.Linear(5, self.heads * self.hidden_dim),
            nn.SiLU(),
        )
        self.out_proj = nn.Linear(self.heads * self.hidden_dim, self.pocket_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.q_transform, std=0.02)
        nn.init.normal_(self.k_transform, std=0.02)
        with torch.no_grad():
            self.q_transform[:, 0] += 1.0
            self.k_transform[:, 0] += 1.0

    def _ensure_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 2:
            return x.unsqueeze(0), True
        return x, False

    def _pad_to_pga(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == self.pocket_dim:
            return x
        out = x.new_zeros(x.shape[:-1] + (self.pocket_dim,))
        take = min(x.size(-1), self.pocket_dim)
        out[..., :take] = x[..., :take]
        return out

    def _extract_invariants(self, mv: torch.Tensor) -> torch.Tensor:
        g0 = mv[..., 0:1].abs()
        g1 = mv[..., 1:5].norm(dim=-1, keepdim=True)
        g2 = mv[..., 5:11].norm(dim=-1, keepdim=True)
        g3 = mv[..., 11:15].norm(dim=-1, keepdim=True)
        g4 = mv[..., 15:16].abs()
        return torch.cat([g0, g1, g2, g3, g4], dim=-1)

    def forward(
        self,
        drug_multivectors: torch.Tensor,
        pocket: PocketEncoderOutput,
        drug_mask: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> ReversedGeometricAttentionOutput:
        drug_mv, squeeze = self._ensure_batch(drug_multivectors)
        pocket_mv, _ = self._ensure_batch(pocket.attention_anchors)
        drug_mv = self._pad_to_pga(drug_mv)
        pocket_mv = self._pad_to_pga(pocket_mv)

        if drug_mask is None:
            drug_mask = torch.ones(drug_mv.shape[:2], dtype=torch.bool, device=drug_mv.device)
        if residue_mask is None:
            residue_mask = torch.ones(pocket_mv.shape[:2], dtype=torch.bool, device=pocket_mv.device)

        batch_size, n_drug, _ = drug_mv.shape
        _, n_prot, _ = pocket_mv.shape

        q = pga_geometric_product(
            drug_mv.unsqueeze(2).expand(batch_size, n_drug, self.heads, self.pocket_dim),
            self.q_transform.view(1, 1, self.heads, self.pocket_dim).expand(batch_size, n_drug, self.heads, self.pocket_dim),
        )
        k = pga_geometric_product(
            pocket_mv.unsqueeze(2).expand(batch_size, n_prot, self.heads, self.pocket_dim),
            self.k_transform.view(1, 1, self.heads, self.pocket_dim).expand(batch_size, n_prot, self.heads, self.pocket_dim),
        )

        inv_pocket = self._extract_invariants(pocket_mv)
        v = self.value_proj(inv_pocket).view(batch_size, n_prot, self.heads, self.hidden_dim)

        logits = geometric_inner_product(
            q.unsqueeze(2),
            k.unsqueeze(1),
        ).permute(0, 3, 1, 2) / math.sqrt(float(self.pocket_dim))

        mask = drug_mask.unsqueeze(1).unsqueeze(-1) & residue_mask.unsqueeze(1).unsqueeze(1)
        logits = logits.masked_fill(~mask, float("-inf"))
        attention_weights = F.softmax(logits, dim=-1)

        v_heads = v.permute(0, 2, 1, 3)
        attended = torch.matmul(attention_weights, v_heads)
        attended_flat = attended.permute(0, 2, 1, 3).reshape(batch_size, n_drug, self.heads * self.hidden_dim)
        attended_drug = self.out_proj(attended_flat)

        residue_importance = attention_weights.mean(dim=(1, 2))
        accessibility_mask = attention_weights.mean(dim=1).max(dim=-1).values
        v_flat = v.reshape(batch_size, n_prot, self.heads * self.hidden_dim)
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
