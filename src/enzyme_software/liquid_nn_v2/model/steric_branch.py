from __future__ import annotations

from typing import Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
from enzyme_software.liquid_nn_v2.model.pooling import segment_sum


if TORCH_AVAILABLE:
    class StericFeatureProjector(nn.Module):
        def __init__(self, output_dim: int):
            super().__init__()
            self.output_dim = int(output_dim)
            self.projector = nn.Sequential(
                nn.LazyLinear(self.output_dim),
                nn.SiLU(),
                nn.LayerNorm(self.output_dim),
            )

        def forward(self, atom_3d_features):
            if atom_3d_features is None:
                return None
            if atom_3d_features.ndim == 1:
                atom_3d_features = atom_3d_features.unsqueeze(-1)
            return self.projector(atom_3d_features)


    class Steric3DBranch(nn.Module):
        """Optional low-cost steric descriptor branch."""

        def __init__(self, atom_dim: int, mol_dim: int):
            super().__init__()
            self.atom_projector = StericFeatureProjector(atom_dim)
            self.mol_projector = nn.Sequential(
                nn.Linear(atom_dim, mol_dim),
                nn.SiLU(),
                nn.LayerNorm(mol_dim),
            )

        def forward(self, atom_3d_features, batch, num_molecules: int):
            if atom_3d_features is None:
                return {
                    "atom_embedding": None,
                    "mol_embedding": None,
                    "diagnostics": {"steric_features_present": 0.0},
                }
            atom_embedding = self.atom_projector(atom_3d_features)
            if atom_embedding is None:
                return {
                    "atom_embedding": None,
                    "mol_embedding": None,
                    "diagnostics": {"steric_features_present": 0.0},
                }
            counts = torch.bincount(batch, minlength=num_molecules).clamp(min=1).unsqueeze(-1)
            pooled = segment_sum(atom_embedding, batch, num_molecules) / counts
            mol_embedding = self.mol_projector(pooled)
            diagnostics = {
                "steric_features_present": 1.0,
                "steric_atom_abs_mean": float(atom_embedding.detach().abs().mean().item()),
            }
            return {
                "atom_embedding": atom_embedding,
                "mol_embedding": mol_embedding,
                "diagnostics": diagnostics,
            }
else:  # pragma: no cover
    class StericFeatureProjector:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class Steric3DBranch:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
