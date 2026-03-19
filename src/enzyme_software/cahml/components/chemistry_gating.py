from __future__ import annotations

from typing import Dict, Tuple

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class ChemistryAwareGating(nn.Module):
        def __init__(self, *, mol_feature_dim: int = 64, atom_feature_dim: int = 32, n_models: int = 3, hidden_dim: int = 64, dropout: float = 0.1):
            super().__init__()
            self.n_models = int(n_models)
            self.mol_encoder = nn.Sequential(nn.Linear(mol_feature_dim, hidden_dim), nn.ReLU())
            self.atom_encoder = nn.Sequential(nn.Linear(atom_feature_dim, hidden_dim), nn.ReLU())
            self.gate_network = nn.Sequential(
                nn.Linear(hidden_dim * 2 + n_models, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, n_models),
            )
            self.temperature = nn.Parameter(torch.ones(1))

        def forward(self, mol_features: torch.Tensor, atom_features: torch.Tensor, base_predictions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            n_atoms = atom_features.shape[0]
            mol_enc = self.mol_encoder(mol_features).unsqueeze(0).expand(n_atoms, -1)
            atom_enc = self.atom_encoder(atom_features)
            combined = torch.cat([mol_enc, atom_enc, base_predictions], dim=-1)
            gate_logits = self.gate_network(combined)
            gate_weights = torch.softmax(gate_logits / self.temperature.clamp_min(1.0e-3), dim=-1)
            info = {
                "gate_weights_mean": gate_weights.mean(dim=0),
                "gate_entropy": -(gate_weights * torch.log(gate_weights.clamp_min(1.0e-8))).sum(dim=-1).mean(),
                "temperature": self.temperature.detach(),
            }
            return gate_weights, info
else:  # pragma: no cover
    class ChemistryAwareGating:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
