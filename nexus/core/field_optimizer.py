from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from nexus.field.siren_base import SIREN_OMEGA_0


@dataclass
class FieldGradientOptimizationReport:
    gradient_loss: torch.Tensor
    spectral_penalty: torch.Tensor
    alpha_calibration_loss: torch.Tensor
    total_loss: torch.Tensor
    atomic_gradients: torch.Tensor
    vacuum_values: torch.Tensor
    vacuum_gradients: torch.Tensor


class Field_Gradient_Optimizer(nn.Module):
    def __init__(
        self,
        gradient_weight: float = 1.0,
        spectral_weight: float = 0.05,
        alpha_weight: float = 0.05,
        vacuum_scale: float = 1.35,
    ) -> None:
        super().__init__()
        self.gradient_weight = float(gradient_weight)
        self.spectral_weight = float(spectral_weight)
        self.alpha_weight = float(alpha_weight)
        self.vacuum_scale = float(vacuum_scale)

    def _vacuum_points(self, manifold, field, n_points: int = 16) -> torch.Tensor:
        work_dtype = manifold.pos.dtype
        pos = manifold.pos.to(dtype=work_dtype)
        center = pos.mean(dim=0)
        rel = pos - center.unsqueeze(0)
        radius = rel.norm(dim=-1).max().clamp_min(2.0) * self.vacuum_scale
        angles = torch.linspace(0.0, 2.0 * torch.pi, n_points + 1, device=manifold.pos.device, dtype=work_dtype)[:-1]
        z = torch.linspace(-0.5, 0.5, n_points, device=manifold.pos.device, dtype=work_dtype)
        x = torch.cos(angles) * radius
        y = torch.sin(angles) * radius
        points = torch.stack([x, y, z * radius], dim=-1)
        return center.unsqueeze(0) + points

    def gradient_match_loss(self, field, manifold) -> tuple[torch.Tensor, torch.Tensor]:
        fallback = manifold.pos - manifold.pos.mean(dim=0, keepdim=True)
        direction = manifold.forces + 0.1 * fallback
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        coords = (manifold.pos + 1.0e-2 * direction).clone().requires_grad_(True)
        psi = field.query(coords)
        gradients = torch.autograd.grad(
            outputs=psi.sum(),
            inputs=coords,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        gradients = torch.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)
        target = manifold.forces.to(dtype=coords.dtype, device=coords.device)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        loss = (gradients - target).pow(2).mean()
        return loss, gradients

    def spectral_penalty(self, field, manifold) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vacuum = self._vacuum_points(manifold, field).clone().requires_grad_(True)
        psi = field.query(vacuum)
        gradients = torch.autograd.grad(
            outputs=psi.sum(),
            inputs=vacuum,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]
        grad_norm = gradients.norm(dim=-1)
        laplacian_terms = []
        for axis in range(3):
            partial = gradients[:, axis]
            second = torch.autograd.grad(
                outputs=partial.sum(),
                inputs=vacuum,
                retain_graph=True,
                create_graph=True,
                allow_unused=False,
            )[0][:, axis]
            laplacian_terms.append(second)
        laplacian = torch.stack(laplacian_terms, dim=-1).sum(dim=-1)
        penalty = (psi.pow(2) + 0.5 * grad_norm.pow(2) + 0.25 * laplacian.pow(2)).mean()
        return penalty, psi, gradients

    def alpha_calibration_loss(self, field) -> torch.Tensor:
        state = field.splatter_state
        return (state.alpha - state.alpha_target).pow(2).mean()

    def forward(self, field, manifold) -> FieldGradientOptimizationReport:
        if abs(float(field.engine.omega_0) - SIREN_OMEGA_0) > 1.0e-8:
            raise ValueError(f"Field engine omega_0 must remain locked at {SIREN_OMEGA_0}")
        gradient_loss, gradients = self.gradient_match_loss(field, manifold)
        spectral_penalty, vacuum_values, vacuum_gradients = self.spectral_penalty(field, manifold)
        alpha_loss = self.alpha_calibration_loss(field)
        total = (
            self.gradient_weight * gradient_loss
            + self.spectral_weight * spectral_penalty
            + self.alpha_weight * alpha_loss
        )
        return FieldGradientOptimizationReport(
            gradient_loss=gradient_loss,
            spectral_penalty=spectral_penalty,
            alpha_calibration_loss=alpha_loss,
            total_loss=total,
            atomic_gradients=gradients,
            vacuum_values=vacuum_values,
            vacuum_gradients=vacuum_gradients,
        )
