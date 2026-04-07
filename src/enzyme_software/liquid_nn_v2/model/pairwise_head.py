from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch


if TORCH_AVAILABLE:
    class PairwiseHead(nn.Module):
        """Small diagnostic MLP over frozen atom-pair features."""

        def __init__(
            self,
            embedding_dim: int,
            *,
            hidden_scale: float = 2.0,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.embedding_dim = int(embedding_dim)
            self.input_dim = (4 * self.embedding_dim) + 2
            hidden_dim = max(self.embedding_dim, int(round(float(hidden_scale) * float(self.embedding_dim))))
            mid_dim = max(1, self.embedding_dim)
            self.net = nn.Sequential(
                nn.Linear(self.input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, mid_dim),
                nn.LayerNorm(mid_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(mid_dim, 1),
            )

        def forward(self, pair_features):
            if int(pair_features.size(-1)) != int(self.input_dim):
                raise ValueError(
                    f"Expected pairwise probe features with dim {self.input_dim}, "
                    f"got {int(pair_features.size(-1))}"
                )
            return self.net(pair_features)
else:  # pragma: no cover
    class PairwiseHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
