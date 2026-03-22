from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .pga import embed_residue_anchor


@dataclass
class PocketEncoderOutput:
    scalar_features: torch.Tensor
    vector_features: torch.Tensor
    anchor_multivectors: torch.Tensor
    attention_anchors: torch.Tensor
    edge_weights: torch.Tensor
    pair_mask: torch.Tensor
    pocket_context: torch.Tensor


class SteerablePocketBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scalar_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.vector_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        scalar_features: torch.Tensor,
        vector_features: torch.Tensor,
        positions: torch.Tensor,
        pair_mask: torch.Tensor,
        conservation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # unsqueeze(-2)/(-3) is dimension-agnostic:
        #   [N, 3]    → [N, 1, 3] and [1, N, 3]    → [N, N, 3]
        #   [B, N, 3] → [B, N, 1, 3] and [B, 1, N, 3] → [B, N, N, 3]
        rel = positions.unsqueeze(-2) - positions.unsqueeze(-3)
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        direction = rel / dist

        # Broadcast node features to all (i, j) pairs in a dimension-agnostic way.
        # unsqueeze(-2): source i repeats across j  →  [..., N, 1, F]
        # unsqueeze(-3): target j repeats across i  →  [..., 1, N, F]
        feat_i = scalar_features.unsqueeze(-2).expand(*scalar_features.shape[:-1], scalar_features.size(-2), scalar_features.size(-1))
        feat_j = scalar_features.unsqueeze(-3).expand(*scalar_features.shape[:-1], scalar_features.size(-2), scalar_features.size(-1))
        cons_i = conservation.unsqueeze(-2).expand(*conservation.shape[:-1], conservation.size(-2), conservation.size(-1))
        cons_j = conservation.unsqueeze(-3).expand(*conservation.shape[:-1], conservation.size(-2), conservation.size(-1))
        edge_input = torch.cat(
            [
                feat_i,
                feat_j,
                dist,
                cons_i,
                cons_j,
                pair_mask.unsqueeze(-1).to(dtype=scalar_features.dtype),
            ],
            dim=-1,
        )
        edge_hidden = self.edge_mlp(edge_input)
        edge_weights = torch.sigmoid(edge_hidden.mean(dim=-1)) * pair_mask.to(dtype=scalar_features.dtype)

        # sum(dim=-2) aggregates over the source-atom (j) axis for any leading batch dims.
        message_scalar = (edge_hidden * edge_weights.unsqueeze(-1)).sum(dim=-2)
        scalar_out = self.norm(scalar_features + self.scalar_update(torch.cat([scalar_features, message_scalar], dim=-1)))

        vector_gate = self.vector_gate(edge_hidden).squeeze(-1) * edge_weights
        vector_message = (vector_gate.unsqueeze(-1) * direction).sum(dim=-2)
        vector_out = vector_features + vector_message
        return scalar_out, vector_out, edge_weights


class SEGNNPocketEncoder(nn.Module):
    def __init__(
        self,
        residue_vocab: int = 32,
        hidden_dim: int = 128,
        layers: int = 3,
        cutoff: float = 12.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cutoff = float(cutoff)
        self.residue_embedding = nn.Embedding(residue_vocab, hidden_dim)
        self.conservation_proj = nn.Linear(1, hidden_dim)
        self.anchor_proj = nn.Sequential(
            nn.Linear(hidden_dim + 16 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 16),
        )
        self.layers = nn.ModuleList([SteerablePocketBlock(hidden_dim) for _ in range(int(layers))])

    def forward(
        self,
        residue_positions: torch.Tensor,
        residue_types: torch.Tensor,
        aromatic_normals: Optional[torch.Tensor] = None,
        cavity_vectors: Optional[torch.Tensor] = None,
        cavity_density: Optional[torch.Tensor] = None,
        conservation_scores: Optional[torch.Tensor] = None,
        residue_mask: Optional[torch.Tensor] = None,
    ) -> PocketEncoderOutput:
        if residue_positions.ndim != 2 or residue_positions.size(-1) != 3:
            raise ValueError("residue_positions must have shape [N, 3]")

        device = residue_positions.device
        dtype = residue_positions.dtype
        n_residues = residue_positions.size(0)
        if conservation_scores is None:
            conservation_scores = torch.zeros(n_residues, 1, dtype=dtype, device=device)
        else:
            conservation_scores = conservation_scores.reshape(n_residues, 1).to(dtype=dtype, device=device)
        if residue_mask is None:
            residue_mask = torch.ones(n_residues, dtype=torch.bool, device=device)

        if aromatic_normals is None:
            aromatic_normals = torch.zeros_like(residue_positions)
        if cavity_vectors is None:
            cavity_vectors = residue_positions - residue_positions.mean(dim=0, keepdim=True)

        base_scalar = self.residue_embedding(residue_types.to(device=device))
        scalar_features = base_scalar + self.conservation_proj(conservation_scores)
        vector_features = torch.zeros_like(residue_positions)

        dist = torch.cdist(residue_positions, residue_positions)
        pair_mask = (dist <= self.cutoff) & residue_mask.unsqueeze(0) & residue_mask.unsqueeze(1)
        pair_mask = pair_mask & ~torch.eye(n_residues, device=device, dtype=torch.bool)

        edge_weights = residue_positions.new_zeros((n_residues, n_residues))
        for block in self.layers:
            scalar_features, vector_features, edge_weights = block(
                scalar_features,
                vector_features,
                residue_positions,
                pair_mask,
                conservation_scores,
            )

        anchors = embed_residue_anchor(
            residue_positions,
            aromatic_normal=aromatic_normals,
            cavity_direction=cavity_vectors,
            cavity_density=cavity_density,
        )
        anchor_delta = self.anchor_proj(torch.cat([scalar_features, anchors, vector_features], dim=-1))
        attention_anchors = anchors + anchor_delta
        pocket_context = scalar_features.mean(dim=0)
        return PocketEncoderOutput(
            scalar_features=scalar_features,
            vector_features=vector_features,
            anchor_multivectors=anchors,
            attention_anchors=attention_anchors,
            edge_weights=edge_weights,
            pair_mask=pair_mask,
            pocket_context=pocket_context,
        )


__all__ = ["PocketEncoderOutput", "SEGNNPocketEncoder"]
