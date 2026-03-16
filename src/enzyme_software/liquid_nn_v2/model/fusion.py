from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class FusionGate(nn.Module):
        def __init__(self, physics_dim: int, liquid_dim: int, output_dim: int):
            super().__init__()
            self.gate = nn.Sequential(
                nn.Linear(physics_dim + liquid_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid(),
            )
            self.physics_transform = nn.Linear(physics_dim, output_dim)
            self.liquid_transform = nn.Linear(liquid_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)

        def forward(self, physics_out, liquid_out):
            combined = torch.cat([physics_out, liquid_out], dim=-1)
            gate_values = self.gate(combined)
            fused = gate_values * self.liquid_transform(liquid_out) + (1.0 - gate_values) * self.physics_transform(physics_out)
            return self.norm(fused), gate_values
else:  # pragma: no cover
    class FusionGate:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
