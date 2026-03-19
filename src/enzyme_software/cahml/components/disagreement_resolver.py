from __future__ import annotations

from typing import Dict, Tuple

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class DisagreementResolver(nn.Module):
        def __init__(self, *, n_models: int = 3, atom_feature_dim: int = 32, hidden_dim: int = 64, disagreement_threshold: float = 0.25):
            super().__init__()
            self.n_models = int(n_models)
            self.disagreement_threshold = float(disagreement_threshold)
            n_pairs = n_models * (n_models - 1) // 2
            input_dim = n_models + atom_feature_dim + n_pairs
            self.analyzer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            self.resolver = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.trust_predictor = nn.Sequential(
                nn.Linear(hidden_dim // 2, n_models),
                nn.Softmax(dim=-1),
            )

        def compute_pairwise_disagreement(self, predictions: torch.Tensor) -> torch.Tensor:
            disagreements = []
            for i in range(self.n_models):
                for j in range(i + 1, self.n_models):
                    disagreements.append((predictions[:, i] - predictions[:, j]).abs())
            return torch.stack(disagreements, dim=-1)

        def forward(self, base_predictions: torch.Tensor, atom_features: torch.Tensor, weighted_predictions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
            pairwise = self.compute_pairwise_disagreement(base_predictions)
            max_disagreement = pairwise.max(dim=-1).values
            high_mask = max_disagreement > self.disagreement_threshold
            analysis = self.analyzer(torch.cat([base_predictions, atom_features, pairwise], dim=-1))
            trust_weights = self.trust_predictor(analysis)
            trust_weighted = (base_predictions * trust_weights).sum(dim=-1)
            resolution_adjustment = self.resolver(analysis).squeeze(-1)
            resolved = torch.where(high_mask, trust_weighted + 0.2 * resolution_adjustment, weighted_predictions)
            info = {
                "n_high_disagreement": int(high_mask.sum().item()),
                "frac_high_disagreement": float(high_mask.float().mean().item()),
                "mean_disagreement": float(max_disagreement.mean().item()),
                "trust_weights_mean": trust_weights.mean(dim=0).detach().cpu().tolist(),
            }
            return resolved, info
else:  # pragma: no cover
    class DisagreementResolver:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
