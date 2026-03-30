from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DDIOccupancyState:
    inhibitor_positions: torch.Tensor
    inhibitor_radii: torch.Tensor
    strength: torch.Tensor
    field: "DDIOccupancyLayer"

    def occupancy(self, coords: torch.Tensor) -> torch.Tensor:
        return self.field.inhibitor_field(coords, self)


@dataclass
class DDIAccessibilityOutput:
    base_accessibility: torch.Tensor
    inhibitor_field: torch.Tensor
    total_accessibility: torch.Tensor
    occupancy_penalty: torch.Tensor


@dataclass
class InhibitorMaskOutput:
    combined_accessibility: torch.Tensor
    inhibitor_mask_3d: torch.Tensor


class DDIOccupancyLayer(nn.Module):
    def __init__(
        self,
        inhibitor_vocab: int = 128,
        hidden_dim: int = 64,
        default_radius: float = 1.7,
        occupancy_sharpness: float = 6.0,
    ) -> None:
        super().__init__()
        self.default_radius = float(default_radius)
        self.occupancy_sharpness = float(occupancy_sharpness)
        self.radius_embedding = nn.Embedding(inhibitor_vocab, 1)
        with torch.no_grad():
            self.radius_embedding.weight.fill_(self.default_radius)
        self.occupancy_refine = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def build_state(
        self,
        inhibitor_positions: torch.Tensor,
        inhibitor_species: Optional[torch.Tensor] = None,
        *,
        inhibitor_radii: Optional[torch.Tensor] = None,
        inhibitor_strength: float | torch.Tensor = 1.0,
    ) -> DDIOccupancyState:
        if inhibitor_radii is None:
            if inhibitor_species is None:
                radii = inhibitor_positions.new_full((inhibitor_positions.size(0),), self.default_radius)
            else:
                radii = self.radius_embedding(
                    inhibitor_species.to(device=inhibitor_positions.device, dtype=torch.long).clamp_min(0)
                ).squeeze(-1).to(dtype=inhibitor_positions.dtype)
        else:
            radii = inhibitor_radii.to(device=inhibitor_positions.device, dtype=inhibitor_positions.dtype)
        return DDIOccupancyState(
            inhibitor_positions=inhibitor_positions,
            inhibitor_radii=radii,
            strength=torch.as_tensor(inhibitor_strength, device=inhibitor_positions.device, dtype=inhibitor_positions.dtype),
            field=self,
        )

    def inhibitor_field(
        self,
        coords: torch.Tensor,
        state: DDIOccupancyState,
    ) -> torch.Tensor:
        original_shape = coords.shape[:-1]
        points = coords.reshape(-1, 3)
        inhibitor_positions = state.inhibitor_positions.to(device=points.device, dtype=points.dtype)
        inhibitor_radii = state.inhibitor_radii.to(device=points.device, dtype=points.dtype)

        dist = torch.cdist(points.unsqueeze(0), inhibitor_positions.unsqueeze(0)).squeeze(0)
        surface_delta = inhibitor_radii.unsqueeze(0) - dist
        radial_gate = torch.sigmoid(self.occupancy_sharpness * surface_delta)
        refine = self.occupancy_refine(
            torch.stack(
                [
                    dist,
                    inhibitor_radii.unsqueeze(0).expand_as(dist),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        occupancy = (radial_gate * torch.sigmoid(refine)).amax(dim=-1)
        occupancy = occupancy * state.strength.to(device=points.device, dtype=points.dtype)
        return occupancy.reshape(original_shape)

    def update_accessibility_for_ddi(
        self,
        base_a_field: torch.Tensor,
        inhibitor_mv_field: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(base_a_field - inhibitor_mv_field)

    def forward(
        self,
        base_accessibility: torch.Tensor,
        coords: torch.Tensor,
        state: DDIOccupancyState,
    ) -> DDIAccessibilityOutput:
        inhibitor_field = self.inhibitor_field(coords, state)
        total_accessibility = self.update_accessibility_for_ddi(base_accessibility, inhibitor_field)
        occupancy_penalty = inhibitor_field.mean()
        return DDIAccessibilityOutput(
            base_accessibility=base_accessibility,
            inhibitor_field=inhibitor_field,
            total_accessibility=total_accessibility,
            occupancy_penalty=occupancy_penalty,
        )


class InhibitorMaskGenerator(nn.Module):
    def __init__(self, threshold: float = 0.2) -> None:
        super().__init__()
        self.threshold = float(threshold)

    def compute_combined_accessibility(
        self,
        pocket_a_field: torch.Tensor,
        inhibitor_occupancy_field: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(pocket_a_field - inhibitor_occupancy_field)

    def forward(
        self,
        pocket_a_field: torch.Tensor,
        inhibitor_occupancy_field: torch.Tensor,
    ) -> InhibitorMaskOutput:
        combined_accessibility = self.compute_combined_accessibility(
            pocket_a_field,
            inhibitor_occupancy_field,
        )
        inhibitor_mask_3d = (combined_accessibility > self.threshold).to(dtype=combined_accessibility.dtype)
        return InhibitorMaskOutput(
            combined_accessibility=combined_accessibility,
            inhibitor_mask_3d=inhibitor_mask_3d,
        )


def _broadcast_mask_to_trajectories(
    inhibitor_mask_3d: torch.Tensor,
    reaction_trajectories: torch.Tensor,
) -> torch.Tensor:
    mask = inhibitor_mask_3d
    while mask.ndim < reaction_trajectories.ndim:
        mask = mask.unsqueeze(1)
    return mask.expand_as(reaction_trajectories)


def sample_path_accessibility(
    inhibitor_mask_3d: torch.Tensor,
    reaction_trajectories: torch.Tensor,
) -> torch.Tensor:
    if reaction_trajectories.ndim < 4:
        raise ValueError("reaction_trajectories must include spatial dimensions")
    mask = _broadcast_mask_to_trajectories(
        inhibitor_mask_3d.to(dtype=reaction_trajectories.dtype, device=reaction_trajectories.device),
        reaction_trajectories,
    )
    spatial_dims = tuple(range(3, reaction_trajectories.ndim))
    return (mask * reaction_trajectories).mean(dim=spatial_dims)


def apply_digital_inhibition(
    W_original: torch.Tensor,
    inhibitor_mask_3d: torch.Tensor,
    reaction_trajectories: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    path_accessibility = sample_path_accessibility(inhibitor_mask_3d, reaction_trajectories)
    hard_gate = (path_accessibility > float(threshold)).to(dtype=W_original.dtype, device=W_original.device)
    if W_original.ndim == 2:
        if path_accessibility.ndim == 3:
            path_accessibility = path_accessibility.squeeze(0)
            hard_gate = hard_gate.squeeze(0)
    return W_original * hard_gate, path_accessibility


__all__ = [
    "DDIOccupancyLayer",
    "DDIOccupancyState",
    "DDIAccessibilityOutput",
    "InhibitorMaskOutput",
    "InhibitorMaskGenerator",
    "sample_path_accessibility",
    "apply_digital_inhibition",
]
