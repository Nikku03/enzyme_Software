from __future__ import annotations

import torch
import torch.nn as nn


class FlowMatchTSGenerator(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 3,
    ) -> None:
        super().__init__()
        input_dim = 16
        layers = []
        dim = input_dim
        for _ in range(max(int(depth), 1)):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 8))
        self.mlp = nn.Sequential(*layers)
        self.time_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 8),
            nn.Tanh(),
        )

    def velocity_field(
        self,
        y_reactant: torch.Tensor,
        y_product: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1)
        while t.ndim < y_reactant.ndim - 1:
            t = t.unsqueeze(-1)
        midpoint = 0.5 * (y_reactant + y_product)
        delta = y_product - y_reactant
        inputs = torch.cat([midpoint, delta], dim=-1)
        base = self.mlp(inputs)
        gate = self.time_gate(t.to(dtype=inputs.dtype, device=inputs.device))
        return base + gate * delta

    def forward(
        self,
        y_reactant: torch.Tensor,
        y_product: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t is None:
            t = torch.full((), 0.5, dtype=y_reactant.dtype, device=y_reactant.device)
        velocity = self.velocity_field(y_reactant, y_product, t)
        midpoint = 0.5 * (y_reactant + y_product)
        return midpoint + 0.5 * velocity


__all__ = ["FlowMatchTSGenerator"]
