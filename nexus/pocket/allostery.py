from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class AllostericEncoderOutput:
    node_features: torch.Tensor
    vector_features: torch.Tensor
    edge_weights: torch.Tensor
    pair_mask: torch.Tensor
    global_embedding: torch.Tensor
    pocket_embedding: torch.Tensor
    membrane_embedding: torch.Tensor
    catalytic_signal: torch.Tensor


class AllostericSEGNNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 6, hidden_dim),
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
        node_features: torch.Tensor,
        vector_features: torch.Tensor,
        positions: torch.Tensor,
        pair_mask: torch.Tensor,
        conservation_scores: torch.Tensor,
        membrane_mask: torch.Tensor,
        catalytic_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel = positions.unsqueeze(1) - positions.unsqueeze(0)
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        direction = rel / dist

        cons_i = conservation_scores.unsqueeze(1).expand(-1, node_features.size(0), -1)
        cons_j = conservation_scores.unsqueeze(0).expand(node_features.size(0), -1, -1)
        mem_i = membrane_mask.unsqueeze(1).unsqueeze(-1).expand(-1, node_features.size(0), -1)
        cat_j = catalytic_mask.unsqueeze(0).unsqueeze(-1).expand(node_features.size(0), -1, -1)
        edge_input = torch.cat(
            [
                node_features.unsqueeze(1).expand(-1, node_features.size(0), -1),
                node_features.unsqueeze(0).expand(node_features.size(0), -1, -1),
                dist,
                cons_i,
                cons_j,
                mem_i,
                cat_j,
                pair_mask.unsqueeze(-1).to(dtype=node_features.dtype),
            ],
            dim=-1,
        )
        edge_hidden = self.edge_mlp(edge_input)
        edge_weights = torch.sigmoid(edge_hidden.mean(dim=-1)) * pair_mask.to(dtype=node_features.dtype)

        message_scalar = (edge_hidden * edge_weights.unsqueeze(-1)).sum(dim=1)
        scalar_out = self.norm(node_features + self.scalar_update(torch.cat([node_features, message_scalar], dim=-1)))

        vector_gate = self.vector_gate(edge_hidden).squeeze(-1) * edge_weights
        vector_message = (vector_gate.unsqueeze(-1) * direction).sum(dim=1)
        vector_out = vector_features + vector_message
        return scalar_out, vector_out, edge_weights


class AllostericMessagePassingEncoder(nn.Module):
    def __init__(
        self,
        residue_vocab: int = 32,
        hidden_dim: int = 128,
        layers: int = 4,
        cutoff: float = 18.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.cutoff = float(cutoff)
        self.residue_embedding = nn.Embedding(residue_vocab, hidden_dim)
        self.conservation_proj = nn.Linear(1, hidden_dim)
        self.state_proj = nn.Linear(3, hidden_dim)
        self.layers = nn.ModuleList([AllostericSEGNNBlock(hidden_dim) for _ in range(int(layers))])
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.catalytic_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        residue_positions: torch.Tensor,
        residue_types: torch.Tensor,
        *,
        pocket_mask: torch.Tensor,
        membrane_mask: Optional[torch.Tensor] = None,
        catalytic_mask: Optional[torch.Tensor] = None,
        conservation_scores: Optional[torch.Tensor] = None,
        functional_state: Optional[torch.Tensor] = None,
    ) -> AllostericEncoderOutput:
        if residue_positions.ndim != 2 or residue_positions.size(-1) != 3:
            raise ValueError("residue_positions must have shape [N, 3]")
        device = residue_positions.device
        dtype = residue_positions.dtype
        n_res = residue_positions.size(0)
        if conservation_scores is None:
            conservation_scores = torch.zeros(n_res, 1, dtype=dtype, device=device)
        else:
            conservation_scores = conservation_scores.reshape(n_res, 1).to(device=device, dtype=dtype)
        if membrane_mask is None:
            membrane_mask = torch.zeros(n_res, dtype=torch.bool, device=device)
        if catalytic_mask is None:
            catalytic_mask = pocket_mask.to(device=device)
        if functional_state is None:
            functional_state = torch.zeros(3, dtype=dtype, device=device)
        functional_state = functional_state.reshape(1, -1).expand(n_res, -1)

        node_features = self.residue_embedding(residue_types.to(device=device))
        node_features = node_features + self.conservation_proj(conservation_scores) + self.state_proj(functional_state)
        vector_features = torch.zeros_like(residue_positions)

        dist = torch.cdist(residue_positions, residue_positions)
        pair_mask = (dist <= self.cutoff) & ~torch.eye(n_res, dtype=torch.bool, device=device)

        edge_weights = residue_positions.new_zeros((n_res, n_res))
        for block in self.layers:
            node_features, vector_features, edge_weights = block(
                node_features,
                vector_features,
                residue_positions,
                pair_mask,
                conservation_scores,
                membrane_mask.to(device=device),
                catalytic_mask.to(device=device),
            )

        pocket_embed = node_features[pocket_mask.to(device=device)].mean(dim=0) if bool(pocket_mask.any().item()) else node_features.mean(dim=0)
        membrane_embed = node_features[membrane_mask].mean(dim=0) if bool(membrane_mask.any().item()) else node_features.new_zeros(self.hidden_dim)
        global_embed = self.global_head(torch.cat([node_features.mean(dim=0), pocket_embed, membrane_embed], dim=0))

        catalytic_signal = torch.sigmoid(
            self.catalytic_head(
                torch.cat(
                    [
                        node_features,
                        global_embed.unsqueeze(0).expand(n_res, -1),
                        torch.stack(
                            [
                                pocket_mask.to(dtype=dtype, device=device),
                                membrane_mask.to(dtype=dtype, device=device),
                                catalytic_mask.to(dtype=dtype, device=device),
                            ],
                            dim=-1,
                        ),
                    ],
                    dim=-1,
                )
            )
        ).squeeze(-1)

        return AllostericEncoderOutput(
            node_features=node_features,
            vector_features=vector_features,
            edge_weights=edge_weights,
            pair_mask=pair_mask,
            global_embedding=global_embed,
            pocket_embedding=pocket_embed,
            membrane_embedding=membrane_embed,
            catalytic_signal=catalytic_signal,
        )


__all__ = ["AllostericEncoderOutput", "AllostericMessagePassingEncoder"]
