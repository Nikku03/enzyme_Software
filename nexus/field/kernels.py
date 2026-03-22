from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GaussianKernelState:
    alpha: torch.Tensor
    weights: torch.Tensor
    normalized_weights: torch.Tensor
    weight_sum: torch.Tensor
    distances: torch.Tensor


class LearnedGaussianSplatKernel(nn.Module):
    def __init__(self, init_alpha: float = 0.35) -> None:
        super().__init__()
        init_alpha = max(float(init_alpha), 1.0e-4)
        init_raw = torch.log(torch.expm1(torch.tensor(init_alpha)))
        self.alpha_raw = nn.Parameter(init_raw)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha_raw).clamp(min=1.0e-4, max=50.0)

    def forward(
        self,
        query_coords: torch.Tensor,
        atom_coords: torch.Tensor,
        alpha_override: torch.Tensor | None = None,
    ) -> GaussianKernelState:
        rel = query_coords.unsqueeze(1) - atom_coords.unsqueeze(0)
        dist2 = rel.square().sum(dim=-1)
        if alpha_override is None:
            alpha = self.alpha.to(device=query_coords.device, dtype=query_coords.dtype)
        else:
            alpha = alpha_override.to(device=query_coords.device, dtype=query_coords.dtype)
        weights = torch.exp(-alpha * dist2)
        weight_sum = weights.sum(dim=1, keepdim=True)
        normalized_weights = weights / weight_sum.clamp_min(1.0e-8)
        return GaussianKernelState(
            alpha=alpha,
            weights=weights,
            normalized_weights=normalized_weights,
            weight_sum=weight_sum,
            distances=dist2.sqrt(),
        )
