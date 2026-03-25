from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from nexus.physics.constants import ATOMIC_MASSES

from .siren_base import DynamicLayerParams


def _atomic_masses(z: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mass_table = torch.ones(128, dtype=dtype, device=device) * 12.0
    for atomic_number, mass in ATOMIC_MASSES.items():
        mass_table[int(atomic_number)] = float(mass)
    idx = z.to(dtype=torch.long).clamp(min=0, max=127)
    return mass_table[idx]


def get_canonical_coordinates(coords: torch.Tensor, masses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Canonicalize 3-D coordinates with the principal axes of inertia.

    Returns:
    - canonical coordinates in a right-handed frame
    - center of mass
    - canonical basis matrix whose columns are the principal axes
    """
    if coords.ndim != 2 or coords.size(-1) != 3:
        raise ValueError("coords must have shape [N, 3]")
    if masses.ndim != 1 or masses.size(0) != coords.size(0):
        raise ValueError("masses must have shape [N]")

    work_dtype = torch.float64
    coords64 = coords.to(dtype=work_dtype)
    masses64 = masses.to(dtype=work_dtype).clamp_min(1.0e-8)
    total_mass = masses64.sum().clamp_min(1.0e-8)
    center_of_mass = (coords64 * masses64.unsqueeze(-1)).sum(dim=0) / total_mass
    centered = coords64 - center_of_mass

    weighted = centered * masses64.unsqueeze(-1)
    inertia_tensor = centered.transpose(0, 1) @ weighted
    _, eigenvectors = torch.linalg.eigh(inertia_tensor)
    # Detach eigenvectors: the canonical frame is a coordinate-system choice
    # determined by physics (principal axes of inertia), not a learnable quantity.
    # eigh backward = 1/(λᵢ−λⱼ) → NaN/inf for degenerate eigenvalues (linear
    # molecules, symmetric rings).  Gradients still flow through pos directly
    # via centered_pos = (pos − centroid) @ frame.
    eigenvectors = eigenvectors.detach()

    aligned = centered @ eigenvectors
    skewness = (aligned.pow(3) * masses64.unsqueeze(-1)).sum(dim=0)
    signs = torch.where(
        skewness < 0,
        -torch.ones_like(skewness),
        torch.ones_like(skewness),
    )
    canonical_basis = eigenvectors * signs.unsqueeze(0)

    if torch.linalg.det(canonical_basis) < 0:
        canonical_basis[:, 0] = -canonical_basis[:, 0]

    canonical_coords = centered @ canonical_basis
    return canonical_coords, center_of_mass, canonical_basis


def get_canonical_frame(pos: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if pos.ndim != 2 or pos.size(-1) != 3:
        raise ValueError("pos must have shape [N, 3]")
    if z.ndim != 1 or z.size(0) != pos.size(0):
        raise ValueError("z must have shape [N]")
    masses = _atomic_masses(z, dtype=torch.float64, device=pos.device)
    _, centroid, frame = get_canonical_coordinates(pos, masses)
    return centroid, frame


@dataclass
class FieldConditioning:
    molecular_context: torch.Tensor
    hidden_params: List[DynamicLayerParams]
    output_row_scale: torch.Tensor
    output_col_scale: torch.Tensor
    output_bias_shift: torch.Tensor
    canonical_centroid: torch.Tensor
    canonical_frame: torch.Tensor
    attention_weights: torch.Tensor


class ReactivityHyperNetwork(nn.Module):
    def __init__(
        self,
        context_dim: int = 640,
        hidden_dim: int = 512,
        hidden_layers: int = 5,
    ) -> None:
        super().__init__()
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.token_dim = 512

        self.proj_0e = nn.Linear(128, self.token_dim)
        self.proj_0o = nn.Linear(128, self.token_dim)
        self.proj_1o = nn.Linear(128, self.token_dim)
        self.proj_1e = nn.Linear(128, self.token_dim)
        self.proj_2e = nn.Linear(64, self.token_dim)
        self.proj_2o = nn.Linear(64, self.token_dim)
        self.coord_proj = nn.Linear(2, self.token_dim)
        self.species_embedding = nn.Embedding(128, self.token_dim)
        self.pre_attn_norm = nn.LayerNorm(self.token_dim)
        self.context_mha = nn.MultiheadAttention(self.token_dim, num_heads=8, batch_first=True)
        self.context_query = nn.Parameter(torch.randn(1, 1, self.token_dim))
        self.context_gate = nn.Sequential(nn.Linear(self.token_dim, self.token_dim), nn.Sigmoid())
        self.context_proj = nn.Sequential(
            nn.Linear(self.token_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
        )
        self.hidden_row_heads = nn.ModuleList([nn.Linear(512, self.hidden_dim) for _ in range(self.hidden_layers)])
        self.hidden_col_heads = nn.ModuleList(
            [nn.Linear(512, 1 if idx == 0 else self.hidden_dim) for idx in range(self.hidden_layers)]
        )
        self.hidden_bias_heads = nn.ModuleList([nn.Linear(512, self.hidden_dim) for _ in range(self.hidden_layers)])
        self.output_row_head = nn.Linear(512, 1)
        self.output_col_head = nn.Linear(512, self.hidden_dim)
        self.output_bias_head = nn.Linear(512, 1)

    def _build_atom_tokens(self, features: Dict[str, torch.Tensor], pos: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centroid, frame = get_canonical_frame(pos, z)
        centered_pos = (pos.to(dtype=torch.float64) - centroid.unsqueeze(0)) @ frame
        centered_pos = centered_pos.to(dtype=pos.dtype)
        radial = centered_pos.norm(dim=-1, keepdim=True)
        dist_mat = torch.cdist(centered_pos, centered_pos)
        diag_mask = torch.eye(dist_mat.size(0), dtype=torch.bool, device=dist_mat.device)
        dist_mat = dist_mat.masked_fill(diag_mask, float('inf'))
        # Two-step guard for the 1/r Coulomb sum:
        # 1. nan_to_num: torch.cdist uses |a|²+|b|²-2<a,b> for N>25 atoms; float32
        #    rounding can make d²<0, yielding sqrt(negative)=NaN for near-close pairs.
        # 2. clamp_min: prevents 1/r→+inf for coincident atoms; the masked diagonal
        #    (inf) is unaffected (inf > 1e-6), so 1/inf=0 still zeros self-terms.
        dist_mat = torch.nan_to_num(dist_mat, nan=0.0).clamp_min(1.0e-6)
        inv_dist_sum = (1.0 / dist_mat).sum(dim=-1, keepdim=True)
        geom = torch.cat([radial, inv_dist_sum], dim=-1)

        token = (
            self.proj_0e(features["0e"])
            + self.proj_0o(features["0o"])
            + self.proj_1o(features["1o"].norm(dim=-1))
            + self.proj_1e(features["1e"].norm(dim=-1))
            + self.proj_2e(features["2e"].norm(dim=-1))
            + self.proj_2o(features["2o"].norm(dim=-1))
            + self.coord_proj(geom)
            + self.species_embedding(z.to(dtype=torch.long).clamp(min=0, max=127))
        )
        if "topology_atomic" in features:
            token = token + self.proj_0e(features["topology_atomic"])
        elif "0e_topology" in features:
            token = token + self.proj_0e(features["0e_topology"])
        return self.pre_attn_norm(token), centroid, frame

    def _aggregate_context(self, atom_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = atom_tokens.unsqueeze(0)
        attn_output, _ = self.context_mha(sequence, sequence, sequence, need_weights=False)
        attn_output = attn_output.squeeze(0)
        pooled = attn_output.mean(dim=0)
        query = self.context_query.expand(1, -1, -1)
        summary, attn_weights = self.context_mha(query, sequence, sequence, need_weights=True, average_attn_weights=False)
        summary = summary.squeeze(0).squeeze(0)
        gate = self.context_gate(summary)
        context = self.context_proj(summary + gate * pooled)
        return context, attn_weights.mean(dim=1).squeeze(0)

    def forward(self, features: Dict[str, torch.Tensor], pos: torch.Tensor, z: torch.Tensor) -> FieldConditioning:
        atom_tokens, centroid, frame = self._build_atom_tokens(features, pos, z)
        context, attention_weights = self._aggregate_context(atom_tokens)

        hidden_params: List[DynamicLayerParams] = []
        for row_head, col_head, bias_head in zip(self.hidden_row_heads, self.hidden_col_heads, self.hidden_bias_heads):
            row_scale = 1.0 + 0.1 * torch.tanh(row_head(context))
            col_scale = 1.0 + 0.1 * torch.tanh(col_head(context))
            bias_shift = 0.05 * torch.tanh(bias_head(context))
            hidden_params.append(
                DynamicLayerParams(
                    row_scale=row_scale,
                    col_scale=col_scale,
                    bias_shift=bias_shift,
                )
            )

        output_row_scale = 1.0 + 0.1 * torch.tanh(self.output_row_head(context))
        output_col_scale = 1.0 + 0.1 * torch.tanh(self.output_col_head(context))
        output_bias_shift = 0.05 * torch.tanh(self.output_bias_head(context))
        return FieldConditioning(
            molecular_context=context,
            hidden_params=hidden_params,
            output_row_scale=output_row_scale.view(-1),
            output_col_scale=output_col_scale.view(-1),
            output_bias_shift=output_bias_shift.view(-1),
            canonical_centroid=centroid,
            canonical_frame=frame,
            attention_weights=attention_weights,
        )
