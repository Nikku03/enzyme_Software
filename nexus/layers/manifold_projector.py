"""
Phase 3 – G(3,0,1) Admissible Manifold Projection
==================================================

This projector validates multivectors in invariant space, but unlike the
older autoencoder it does not try to reconstruct raw coordinates from
invariants. Instead it predicts grade-wise scaling factors and applies them
to the original multivector, preserving geometric orientation while nudging
thermodynamic magnitudes toward the learned admissible manifold.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ManifoldProjectionOutput:
    projected_invariants: torch.Tensor
    projected_multivectors: torch.Tensor
    recon_loss: torch.Tensor
    surface_energy: torch.Tensor
    density_penalty: torch.Tensor
    stability_mask: torch.Tensor


class MultivectorManifoldProjector(nn.Module):
    """
    G(3,0,1)-native manifold projector with orientation-preserving grade scaling.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        surface_energy_threshold: float = 5.0,
        radical_ratio_threshold: float = 2.0,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.invariant_dim = 8
        self.surface_energy_threshold = float(surface_energy_threshold)
        self.radical_ratio_threshold = float(radical_ratio_threshold)

        self.encoder = nn.Sequential(
            nn.Linear(self.invariant_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.invariant_dim),
        )
        self.grade_scaler = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 5),
        )

    @staticmethod
    def _require_pga16(multivectors: torch.Tensor) -> torch.Tensor:
        if multivectors.size(-1) != 16:
            raise ValueError(
                f"MultivectorManifoldProjector expects trailing multivector dimension 16, got {multivectors.size(-1)}"
            )
        return multivectors

    def _invariant_features(self, multivectors: torch.Tensor) -> torch.Tensor:
        mv = self._require_pga16(multivectors)
        inv0 = mv[..., 0].abs()
        inv1 = mv[..., 1:4].norm(dim=-1)
        inv2 = mv[..., 4].abs()
        inv3 = mv[..., 5:8].norm(dim=-1)
        inv4 = mv[..., 8:11].norm(dim=-1)
        inv5 = mv[..., 11].abs()
        inv6 = mv[..., 12:15].norm(dim=-1)
        inv7 = mv[..., 15].abs()
        return torch.stack([inv0, inv1, inv2, inv3, inv4, inv5, inv6, inv7], dim=-1)

    def surface_area_energy(self, multivectors: torch.Tensor) -> torch.Tensor:
        mv = self._require_pga16(multivectors)
        biv_norm = mv[..., 5:11].norm(dim=-1)
        scalar_mag = mv[..., 0].abs()
        return biv_norm / (scalar_mag + 1.0e-6)

    def electronic_density_penalty(self, multivectors: torch.Tensor) -> torch.Tensor:
        mv = self._require_pga16(multivectors)
        scalar = mv[..., 0]
        neg_penalty = F.relu(-scalar)
        vec_norm = mv[..., 1:5].norm(dim=-1)
        radical_char = vec_norm / (scalar.abs() + 1.0e-6)
        radical_penalty = F.relu(radical_char - self.radical_ratio_threshold)
        return (neg_penalty + radical_penalty).mean()

    def stability_mask_for(self, multivectors: torch.Tensor) -> torch.Tensor:
        mv = self._require_pga16(multivectors)
        surf = self.surface_area_energy(mv)
        dens = F.relu(-mv[..., 0]) + F.relu((mv[..., 1:5].norm(dim=-1) / (mv[..., 0].abs() + 1.0e-6)) - self.radical_ratio_threshold)
        return (surf < self.surface_energy_threshold) & (dens < 0.1)

    def project(self, multivectors: torch.Tensor) -> torch.Tensor:
        mv = self._require_pga16(multivectors)
        inv_input = self._invariant_features(mv)
        latent = self.encoder(inv_input)
        scales = F.softplus(self.grade_scaler(latent))

        projected_mv = mv.clone()
        projected_mv[..., 0] *= scales[..., 0]
        projected_mv[..., 1:5] *= scales[..., 1].unsqueeze(-1)
        projected_mv[..., 5:11] *= scales[..., 2].unsqueeze(-1)
        projected_mv[..., 11:15] *= scales[..., 3].unsqueeze(-1)
        projected_mv[..., 15] *= scales[..., 4]
        return projected_mv

    def forward(self, multivectors: torch.Tensor) -> ManifoldProjectionOutput:
        mv = self._require_pga16(multivectors)
        inv_input = self._invariant_features(mv)
        latent = self.encoder(inv_input)
        inv_proj = self.decoder(latent)
        recon_loss = F.mse_loss(inv_proj, inv_input, reduction="mean")

        scales = F.softplus(self.grade_scaler(latent))
        projected_mv = mv.clone()
        projected_mv[..., 0] *= scales[..., 0]
        projected_mv[..., 1:5] *= scales[..., 1].unsqueeze(-1)
        projected_mv[..., 5:11] *= scales[..., 2].unsqueeze(-1)
        projected_mv[..., 11:15] *= scales[..., 3].unsqueeze(-1)
        projected_mv[..., 15] *= scales[..., 4]

        surf_energy = self.surface_area_energy(projected_mv)
        density_penalty = self.electronic_density_penalty(projected_mv)
        stab_mask = self.stability_mask_for(projected_mv)

        return ManifoldProjectionOutput(
            projected_invariants=inv_proj,
            projected_multivectors=projected_mv,
            recon_loss=recon_loss,
            surface_energy=surf_energy,
            density_penalty=density_penalty,
            stability_mask=stab_mask,
        )


__all__ = ["MultivectorManifoldProjector", "ManifoldProjectionOutput"]
