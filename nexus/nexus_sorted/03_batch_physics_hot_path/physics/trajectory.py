from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NEXUS_Trajectory:
    q_path: torch.Tensor
    p_path: torch.Tensor
    h_path: torch.Tensor
    hamiltonian_drift: torch.Tensor
    action_integral: torch.Tensor
    kinetic_path: torch.Tensor
    physical_path: torch.Tensor
    reactive_path: torch.Tensor
    lie_state_path: torch.Tensor | None = None


__all__ = ["NEXUS_Trajectory"]
