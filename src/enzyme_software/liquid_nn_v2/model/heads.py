from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch
if TORCH_AVAILABLE:
    class SiteHead(nn.Module):
        def __init__(self, atom_dim: int, mol_dim: int, hidden_dim: int = 64):
            super().__init__()
            self.context_proj = nn.Linear(mol_dim, atom_dim)
            self.mlp = nn.Sequential(
                nn.Linear(atom_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(max(8, hidden_dim // 2), 1),
            )

        def forward(self, atom_features, mol_features, batch):
            atom_context = self.context_proj(mol_features)[batch]
            return self.mlp(torch.cat([atom_features, atom_context], dim=-1))


    class CYPHead(nn.Module):
        def __init__(self, mol_dim: int, num_classes: int, hidden_dim: int = 128, class_names=None):
            super().__init__()
            self.class_names = tuple(class_names or [f"CYP_{idx}" for idx in range(int(num_classes))])
            self.mlp = nn.Sequential(
                nn.Linear(mol_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(max(8, hidden_dim // 2), int(num_classes)),
            )

        def forward(self, mol_features):
            return self.mlp(mol_features)
else:  # pragma: no cover
    class SiteHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()

    class CYPHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
