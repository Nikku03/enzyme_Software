from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class CatalyticActivationState:
    low_spin_score: torch.Tensor
    high_spin_score: torch.Tensor
    activation_gate: torch.Tensor
    activation_bias: torch.Tensor
    spin_state: str
    coordination_shift: torch.Tensor


class CatalyticStateActivation(nn.Module):
    def __init__(self, max_bias_norm: float = 0.15) -> None:
        super().__init__()
        self.max_bias_norm = float(max_bias_norm)
        self.low_spin_head = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 1))
        self.high_spin_head = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 1))
        self.bias_head = nn.Sequential(nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 3))

    def forward(
        self,
        manifold,
        reactivity_grad: torch.Tensor,
        energy_grad: torch.Tensor,
    ) -> CatalyticActivationState:
        pos = manifold.pos
        forces = manifold.forces.to(device=pos.device, dtype=pos.dtype)
        centroid = pos.mean(dim=0, keepdim=True)
        rel = pos - centroid
        rel_norm = rel.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        react_norm = reactivity_grad.norm(dim=-1, keepdim=True)
        energy_norm = energy_grad.norm(dim=-1, keepdim=True)
        force_norm = forces.norm(dim=-1, keepdim=True)
        alignment = (reactivity_grad * forces).sum(dim=-1, keepdim=True)
        anti_alignment = (reactivity_grad * energy_grad).sum(dim=-1, keepdim=True)

        activation_input = torch.cat(
            [
                rel_norm,
                react_norm,
                energy_norm,
                force_norm,
                alignment,
                anti_alignment,
            ],
            dim=-1,
        )
        low_spin_logits = self.low_spin_head(activation_input)
        high_spin_logits = self.high_spin_head(activation_input)
        activation_gate = torch.sigmoid(high_spin_logits - low_spin_logits)

        raw_bias = self.bias_head(activation_input)
        bias_norm = raw_bias.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        scale = torch.clamp(self.max_bias_norm / bias_norm, max=1.0)
        activation_bias = raw_bias * scale * activation_gate
        coordination_shift = activation_gate * torch.tanh(rel)

        mean_gate = float(activation_gate.mean().detach().cpu().item())
        spin_state = "5-coordinate high-spin" if mean_gate >= 0.5 else "6-coordinate low-spin"
        return CatalyticActivationState(
            low_spin_score=low_spin_logits.squeeze(-1),
            high_spin_score=high_spin_logits.squeeze(-1),
            activation_gate=activation_gate.squeeze(-1),
            activation_bias=activation_bias,
            spin_state=spin_state,
            coordination_shift=coordination_shift,
        )
