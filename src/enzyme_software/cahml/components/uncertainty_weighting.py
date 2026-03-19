from __future__ import annotations

from typing import Dict, Tuple

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class UncertaintyEstimator(nn.Module):
        def __init__(self, *, n_models: int = 3, atom_feature_dim: int = 32, hidden_dim: int = 32):
            super().__init__()
            self.uncertainty_net = nn.Sequential(
                nn.Linear(n_models + atom_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_models),
                nn.Softplus(),
            )
            self.uncertainty_scale = nn.Parameter(torch.ones(1))

        def forward(self, base_predictions: torch.Tensor, atom_features: torch.Tensor, gate_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
            pred_variance = base_predictions.var(dim=-1, keepdim=True, unbiased=False)
            pred_probs = base_predictions.clamp(1.0e-4, 1.0 - 1.0e-4)
            pred_entropy = -(pred_probs * torch.log(pred_probs) + (1.0 - pred_probs) * torch.log(1.0 - pred_probs)).mean(dim=-1, keepdim=True)
            learned_uncertainty = self.uncertainty_net(torch.cat([base_predictions, atom_features], dim=-1))
            total_uncertainty = learned_uncertainty * self.uncertainty_scale.clamp_min(1.0e-4)
            precision = 1.0 / (total_uncertainty + 1.0e-6)
            adjusted_weights = gate_weights * precision
            adjusted_weights = adjusted_weights / adjusted_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            confidence = torch.sigmoid(2.0 / (total_uncertainty.mean(dim=-1) + pred_variance.squeeze(-1) + pred_entropy.squeeze(-1) + 1.0e-6))
            info = {
                "prediction_variance": float(pred_variance.mean().item()),
                "prediction_entropy": float(pred_entropy.mean().item()),
                "learned_uncertainty_mean": float(learned_uncertainty.mean().item()),
                "confidence_mean": float(confidence.mean().item()),
            }
            return adjusted_weights, confidence, info
else:  # pragma: no cover
    class UncertaintyEstimator:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
