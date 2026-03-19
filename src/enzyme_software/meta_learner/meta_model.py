from __future__ import annotations

from typing import Dict, Tuple

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class MetaLearner(nn.Module):
        def __init__(
            self,
            *,
            n_models: int = 3,
            n_cyp: int = 5,
            atom_feature_dim: int = 11,
            global_feature_dim: int = 19,
            hidden_dim: int = 32,
            use_attention: bool = True,
        ):
            super().__init__()
            self.n_models = int(n_models)
            self.n_cyp = int(n_cyp)
            self.use_attention = bool(use_attention)
            self.site_mlp = nn.Sequential(
                nn.Linear(atom_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            if self.use_attention:
                self.model_attention = nn.Sequential(
                    nn.Linear(atom_feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, self.n_models),
                    nn.Softmax(dim=-1),
                )
            self.cyp_mlp = nn.Sequential(
                nn.Linear(global_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_cyp),
            )

        def forward(
            self,
            atom_features: torch.Tensor,
            global_features: torch.Tensor,
            site_scores_raw: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            stats: Dict[str, torch.Tensor] = {}
            site_logits = self.site_mlp(atom_features).squeeze(-1)
            if self.use_attention and site_scores_raw is not None:
                attn = self.model_attention(atom_features)
                stats["attention_weights"] = attn.mean(dim=0)
                site_logits = (site_scores_raw * attn).sum(dim=1) + 0.5 * site_logits
            cyp_logits = self.cyp_mlp(global_features)
            return site_logits, cyp_logits, stats
else:  # pragma: no cover
    class MetaLearner:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
