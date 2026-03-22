from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def topology_entropy_regularizer(scale_weights: torch.Tensor, scale_embeddings: torch.Tensor) -> torch.Tensor:
    probs = scale_weights.clamp_min(1.0e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    normed = F.normalize(scale_embeddings, dim=-1)
    similarity = torch.matmul(normed, normed.transpose(-1, -2))
    eye = torch.eye(similarity.size(-1), device=similarity.device, dtype=similarity.dtype)
    redundancy = ((similarity - eye) ** 2).mean()
    return redundancy - 0.05 * entropy


@dataclass
class MSGPreAttentionOutput:
    atomic_features: torch.Tensor
    group_features: torch.Tensor
    conformer_features: torch.Tensor
    scale_weights: torch.Tensor
    entropy_loss: torch.Tensor
    pooled_scale_embeddings: torch.Tensor


class MSGPreAttention(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.scale_gate = nn.Sequential(
            nn.Linear(feature_dim * 3 + 4, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )
        self.atomic_refine = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim))
        self.group_refine = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim))
        self.conformer_refine = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.SiLU(), nn.Linear(feature_dim, feature_dim))

    def forward(
        self,
        atomic_features: torch.Tensor,
        group_features: torch.Tensor,
        conformer_features: torch.Tensor,
        distance_matrix: torch.Tensor,
        local_mask: torch.Tensor,
        global_mask: torch.Tensor,
        assignments: torch.Tensor,
    ) -> MSGPreAttentionOutput:
        atom_pool = atomic_features.mean(dim=0)
        group_pool = group_features.mean(dim=0) if group_features.numel() else atomic_features.new_zeros((self.feature_dim,))
        conf_pool = conformer_features.mean(dim=0)
        scale_embeddings = torch.stack([atom_pool, group_pool, conf_pool], dim=0)

        local_density = local_mask.to(dtype=atomic_features.dtype).mean()
        global_density = global_mask.to(dtype=atomic_features.dtype).mean()
        off_diag = distance_matrix[~torch.eye(distance_matrix.size(0), device=distance_matrix.device, dtype=torch.bool)]
        mean_distance = off_diag.mean() if off_diag.numel() > 0 else distance_matrix.new_zeros(())
        n_groups = atomic_features.new_tensor(float(int(assignments.max().item()) + 1) if assignments.numel() else 0.0)
        stats = torch.stack(
            [
                local_density,
                global_density,
                mean_distance,
                n_groups / max(float(atomic_features.size(0)), 1.0),
            ],
            dim=0,
        )
        gate_logits = self.scale_gate(torch.cat([atom_pool, group_pool, conf_pool, stats], dim=0))
        scale_weights = torch.softmax(gate_logits, dim=-1)

        atomic_out = self.atomic_refine(torch.cat([atomic_features, scale_weights[0] * conf_pool.unsqueeze(0).expand_as(atomic_features)], dim=-1))
        if group_features.numel():
            group_out = self.group_refine(torch.cat([group_features, scale_weights[1] * atom_pool.unsqueeze(0).expand_as(group_features)], dim=-1))
        else:
            group_out = group_features
        conformer_out = self.conformer_refine(torch.cat([conformer_features, scale_weights[2] * atom_pool.unsqueeze(0).expand_as(conformer_features)], dim=-1))

        entropy_loss = topology_entropy_regularizer(scale_weights.unsqueeze(0), scale_embeddings.unsqueeze(0))
        return MSGPreAttentionOutput(
            atomic_features=atomic_out,
            group_features=group_out,
            conformer_features=conformer_out,
            scale_weights=scale_weights,
            entropy_loss=entropy_loss,
            pooled_scale_embeddings=scale_embeddings,
        )
