from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch


if TORCH_AVAILABLE:
    class DistilledProposerHead(nn.Module):
        """Compact scalar readout trained from pairwise-distilled soft targets."""

        def __init__(
            self,
            embedding_dim: int,
            *,
            hidden_dim: int | None = None,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.embedding_dim = int(embedding_dim)
            self.hidden_dim = max(1, int(hidden_dim or self.embedding_dim))
            self.net = nn.Sequential(
                nn.Linear(self.embedding_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, 1),
            )

        def forward(self, atom_features):
            if int(atom_features.size(-1)) != int(self.embedding_dim):
                raise ValueError(
                    f"Expected distilled proposer atom features with dim {self.embedding_dim}, "
                    f"got {int(atom_features.size(-1))}"
                )
            return self.net(atom_features)
else:  # pragma: no cover
    class DistilledProposerHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
