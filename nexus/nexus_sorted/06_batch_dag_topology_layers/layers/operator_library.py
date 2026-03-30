from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from nexus.core.generative_agency import NEXUS_Seed
from nexus.models.mol_llama_wrapper import LatentBlueprint
from nexus.physics.clifford_math import clifford_geometric_product, embed_coordinates


DEFAULT_OPERATOR_NAMES: tuple[str, ...] = (
    "aliphatic_hydroxylation",
    "aromatic_hydroxylation",
    "n_dealkylation",
    "o_dealkylation",
    "deamination",
    "epoxidation",
    "sulfoxidation",
    "glucuronidation",
)


@dataclass
class OperatorApplication:
    operator_index: int
    operator_name: str
    transformed_seed: NEXUS_Seed
    transformed_multivector: torch.Tensor
    rotor: torch.Tensor
    field_perturbation: torch.Tensor


class DifferentiableGeometricOperatorLibrary(nn.Module):
    def __init__(
        self,
        operator_names: Sequence[str] = DEFAULT_OPERATOR_NAMES,
        hidden_dim: int = 64,
        latent_dim: int = 256,
    ) -> None:
        super().__init__()
        self.operator_names = tuple(operator_names)
        self.n_operators = len(self.operator_names)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

        self.operator_embedding = nn.Embedding(self.n_operators, self.hidden_dim)
        self.geometry_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 6, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 4),
        )
        self.latent_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.local_field_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 8),
        )

        plane_lookup = torch.tensor([4, 5, 6, 4, 5, 6, 4, 5], dtype=torch.long)
        self.register_buffer("plane_lookup", plane_lookup, persistent=False)

    def operator_name(self, operator_index: int) -> str:
        return self.operator_names[int(operator_index) % self.n_operators]

    def _rotor_reverse(self, rotor: torch.Tensor) -> torch.Tensor:
        rev = rotor.clone()
        rev[..., 4:7] = -rev[..., 4:7]
        rev[..., 7] = -rev[..., 7]
        return rev

    def _apply_rotor(self, rel_coords: torch.Tensor, rotor: torch.Tensor) -> torch.Tensor:
        rel_mv = embed_coordinates(rel_coords)
        rotated = clifford_geometric_product(
            clifford_geometric_product(rotor, rel_mv),
            self._rotor_reverse(rotor),
        )
        return rotated[..., 1:4]

    def _latent_shift(
        self,
        latent: LatentBlueprint,
        op_embed: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> LatentBlueprint:
        dtype = latent.pooled.dtype
        device = latent.pooled.device
        control = torch.cat(
            [
                op_embed.to(dtype=dtype, device=device),
                edge_weight.reshape(1).to(dtype=dtype, device=device),
                latent.chirality_signature.mean().reshape(1).to(dtype=dtype, device=device),
            ],
            dim=0,
        )
        pooled_delta = 0.05 * self.latent_head(control)
        seq_delta = pooled_delta.unsqueeze(0).expand_as(latent.sequence) * 0.1
        return LatentBlueprint(
            sequence=latent.sequence + seq_delta,
            pooled=latent.pooled + pooled_delta,
            token_ids=latent.token_ids,
            attention_mask=latent.attention_mask,
            smiles=latent.smiles,
            source=latent.source,
            chirality_signature=latent.chirality_signature,
        )

    def apply(
        self,
        seed: NEXUS_Seed,
        *,
        target_atom_index: int,
        operator_index: int,
        edge_weight: torch.Tensor,
        approach_vector: torch.Tensor | None = None,
    ) -> OperatorApplication:
        pos = seed.pos
        dtype = pos.dtype
        device = pos.device
        op_idx = int(operator_index) % self.n_operators
        op_embed = self.operator_embedding(torch.as_tensor(op_idx, dtype=torch.long, device=device)).to(dtype=dtype)

        target_pos = pos[target_atom_index]
        rel = pos - target_pos.unsqueeze(0)
        dist = rel.norm(dim=-1, keepdim=True)
        locality = torch.exp(-dist)
        z_scaled = (seed.z.to(dtype=dtype, device=device).unsqueeze(-1) / 53.0).clamp(0.0, 1.0)
        edge_expand = edge_weight.reshape(1, 1).to(dtype=dtype, device=device).expand(pos.size(0), 1)
        op_expand = op_embed.unsqueeze(0).expand(pos.size(0), -1)
        geom_feat = torch.cat([rel, dist, z_scaled, edge_expand, op_expand], dim=-1)
        geom_params = self.geometry_head(geom_feat)

        angle = 0.35 * torch.tanh(geom_params[..., 0:1]) * locality * edge_expand
        radial_shift = 0.10 * torch.tanh(geom_params[..., 1:2]) * locality * edge_expand
        approach_shift = 0.20 * torch.tanh(geom_params[..., 2:3]) * locality * edge_expand
        field_scale = 0.15 * torch.tanh(geom_params[..., 3:4]) * locality * edge_expand

        plane_idx = int(self.plane_lookup[op_idx].detach().cpu().item())
        rotor = torch.zeros(pos.size(0), 8, dtype=dtype, device=device)
        rotor[..., 0] = torch.cos(0.5 * angle.squeeze(-1))
        rotor[..., plane_idx] = torch.sin(0.5 * angle.squeeze(-1))

        rotated_rel = self._apply_rotor(rel, rotor)
        radial_dir = rel / dist.clamp_min(1.0e-8)
        if approach_vector is None:
            approach = torch.zeros_like(rel)
        else:
            approach = approach_vector.to(dtype=dtype, device=device).reshape(1, 3).expand_as(rel)
        child_pos = target_pos.unsqueeze(0) + rotated_rel + radial_shift * radial_dir + approach_shift * approach
        child_pos = child_pos.requires_grad_(True)

        child_latent = self._latent_shift(seed.latent_blueprint, op_embed, edge_weight)

        field_feat = torch.cat([rel, edge_expand, op_expand], dim=-1)
        field_delta = field_scale * self.local_field_head(field_feat)
        transformed_multivector = embed_coordinates(child_pos) + field_delta

        metadata = dict(seed.metadata)
        metadata.update(
            {
                "operator_index": op_idx,
                "operator_name": self.operator_name(op_idx),
                "edge_weight": float(edge_weight.detach().cpu().item()),
                "target_atom_index": int(target_atom_index),
                "spawned_child": True,
            }
        )
        transformed_seed = NEXUS_Seed(
            pos=child_pos,
            z=seed.z,
            latent_blueprint=child_latent,
            smiles=seed.smiles,
            atom_symbols=list(seed.atom_symbols),
            chirality_codes=seed.chirality_codes,
            jacobian_hook=seed.jacobian_hook,
            metadata=metadata,
        )
        return OperatorApplication(
            operator_index=op_idx,
            operator_name=self.operator_name(op_idx),
            transformed_seed=transformed_seed,
            transformed_multivector=transformed_multivector,
            rotor=rotor,
            field_perturbation=field_delta,
        )


__all__ = [
    "DEFAULT_OPERATOR_NAMES",
    "DifferentiableGeometricOperatorLibrary",
    "OperatorApplication",
]
