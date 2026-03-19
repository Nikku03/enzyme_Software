from __future__ import annotations

from typing import Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class MicroPatternReranker(nn.Module):
        def __init__(self, hidden_dim: int = 128, dropout: float = 0.1, scale_init: float = 0.2):
            super().__init__()
            self.feature_norm = nn.Identity()
            self.feature_proj = nn.Sequential(
                nn.LazyLinear(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
            self.xtb_proj = nn.Sequential(
                nn.LazyLinear(hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            self.delta_head = nn.Linear(hidden_dim + hidden_dim // 2, 1)
            self.gate_head = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            self.scale = nn.Parameter(torch.tensor(float(scale_init)))

        def forward(self, features, xtb_features: Optional[torch.Tensor], base_candidate_logits: torch.Tensor):
            shared = self.feature_proj(self.feature_norm(features))
            if xtb_features is None:
                xtb_features = torch.zeros((features.size(0), 1), device=features.device, dtype=features.dtype)
            xtb_hidden = self.xtb_proj(xtb_features)
            fused = torch.cat([shared, xtb_hidden], dim=-1)
            delta = self.delta_head(fused)
            gate = self.gate_head(fused)
            refined = base_candidate_logits + gate * self.scale * delta
            stats = {
                "gate_mean": float(gate.detach().mean().item()),
                "delta_mean": float(delta.detach().mean().item()),
                "delta_max": float(delta.detach().abs().max().item()),
            }
            return refined, {"delta": delta, "gate": gate, "stats": stats}
else:  # pragma: no cover
    class MicroPatternReranker:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
