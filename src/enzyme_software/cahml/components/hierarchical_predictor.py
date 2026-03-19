from __future__ import annotations

from typing import Dict, Tuple

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, F, nn, require_torch, torch
from enzyme_software.cahml.config import REACTION_TYPES


if TORCH_AVAILABLE:
    class HierarchicalPredictor(nn.Module):
        def __init__(
            self,
            *,
            mol_feature_dim: int = 64,
            atom_feature_dim: int = 32,
            n_models: int = 3,
            n_cyp_classes: int = 5,
            n_reaction_types: int = 6,
            hidden_dim: int = 64,
            dropout: float = 0.1,
            use_base_cyp_prior: bool = True,
        ):
            super().__init__()
            self.use_base_cyp_prior = use_base_cyp_prior
            self.n_cyp_classes = int(n_cyp_classes)
            self.n_reaction_types = int(n_reaction_types)
            self.cyp_embeddings = nn.Embedding(n_cyp_classes, hidden_dim // 2)
            self.rxn_embeddings = nn.Embedding(n_reaction_types, hidden_dim // 2)
            self.cyp_head = nn.Sequential(
                nn.Linear(mol_feature_dim + n_models * n_cyp_classes, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_cyp_classes),
            )
            self.rxn_head = nn.Sequential(
                nn.Linear(mol_feature_dim + hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_reaction_types),
            )
            self.site_head = nn.Sequential(
                nn.Linear(atom_feature_dim + hidden_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.cyp_site_bias = nn.Parameter(torch.zeros(n_cyp_classes))

        def forward(
            self,
            mol_features: torch.Tensor,
            atom_features: torch.Tensor,
            base_site_scores: torch.Tensor,
            base_cyp_probs: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
            n_atoms = atom_features.shape[0]
            cyp_input = torch.cat([mol_features, base_cyp_probs.reshape(-1)], dim=-1)
            cyp_logits = self.cyp_head(cyp_input)
            if self.use_base_cyp_prior:
                base_cyp_mean = base_cyp_probs.float().mean(dim=0).clamp(1e-6)
                cyp_logits = cyp_logits + torch.log(base_cyp_mean)
            cyp_probs = F.softmax(cyp_logits, dim=-1)
            cyp_pred = int(torch.argmax(cyp_probs).item())
            cyp_emb = self.cyp_embeddings(torch.tensor(cyp_pred, device=mol_features.device))
            rxn_input = torch.cat([mol_features, cyp_emb], dim=-1)
            rxn_logits = self.rxn_head(rxn_input)
            rxn_probs = F.softmax(rxn_logits, dim=-1)
            rxn_pred = int(torch.argmax(rxn_probs).item())
            rxn_emb = self.rxn_embeddings(torch.tensor(rxn_pred, device=mol_features.device))
            context = torch.cat([cyp_emb, rxn_emb], dim=-1).unsqueeze(0).expand(n_atoms, -1)
            base_logits = torch.logit(base_site_scores.clamp(1.0e-4, 1.0 - 1.0e-4))
            site_adjustment = self.site_head(torch.cat([atom_features, context, base_logits.unsqueeze(-1)], dim=-1)).squeeze(-1)
            site_logits = base_logits + site_adjustment + self.cyp_site_bias[cyp_pred]
            info = {
                "cyp_prediction": cyp_pred,
                "cyp_confidence": float(cyp_probs.max().item()),
                "reaction_type": rxn_pred,
                "reaction_name": REACTION_TYPES[rxn_pred] if rxn_pred < len(REACTION_TYPES) else "unknown",
                "reaction_confidence": float(rxn_probs.max().item()),
            }
            return site_logits, cyp_logits, rxn_logits, info
else:  # pragma: no cover
    class HierarchicalPredictor:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
