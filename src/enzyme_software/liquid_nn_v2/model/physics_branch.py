from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class PhysicsBranch(nn.Module):
        def __init__(self, output_dim: int = 32):
            super().__init__()
            self.num_physics_features = 1 + 12 + 1 + 1 + 16 + 1
            self.projection = nn.Linear(self.num_physics_features, output_dim)

        def forward(self, atom_features):
            bde_values = atom_features["bde_values"]
            bde_min = float(torch.min(bde_values)) if torch.numel(bde_values) else 0.0
            bde_max = float(torch.max(bde_values)) if torch.numel(bde_values) else 1.0
            scale = max(1e-6, bde_max - bde_min)
            bde_score = 1.0 - (bde_values - bde_min) / scale
            concat = torch.cat(
                [
                    bde_score.unsqueeze(-1),
                    atom_features["bond_classes"],
                    atom_features["electronegativity"].unsqueeze(-1),
                    atom_features["is_aromatic"].unsqueeze(-1),
                    atom_features["functional_groups"],
                    atom_features["radical_stability"].unsqueeze(-1),
                ],
                dim=-1,
            )
            return self.projection(concat)
else:  # pragma: no cover - exercised only when torch missing
    class PhysicsBranch:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
