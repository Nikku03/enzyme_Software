from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class BarrierHead(nn.Module):
        """Local barrier surrogate for winner selection among plausible sites."""

        def __init__(self, *, hidden_dim: int = 32, dropout: float = 0.05):
            super().__init__()
            input_dim = 11 + 4 + 1
            inner = max(16, int(hidden_dim))
            self.output_dim = 4
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, inner),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(inner, inner),
                nn.SiLU(),
            )
            self.score_head = nn.Linear(inner, 1)
            self.ctx_head = nn.Linear(inner, 3)

        def forward(
            self,
            *,
            local_chem_features: torch.Tensor | None,
            accessibility_outputs: dict[str, torch.Tensor] | None,
            event_outputs: dict[str, torch.Tensor] | None = None,
        ) -> dict[str, torch.Tensor]:
            if local_chem_features is None:
                raise ValueError("BarrierHead requires local_chem_features")
            chem = local_chem_features
            device = chem.device
            dtype = chem.dtype
            rows = int(chem.size(0))
            access = accessibility_outputs["features"] if accessibility_outputs is not None else torch.zeros(rows, 4, device=device, dtype=dtype)
            strain = event_outputs["strain"] if event_outputs is not None and "strain" in event_outputs else torch.zeros(rows, 1, device=device, dtype=dtype)
            x = torch.cat([chem, access, strain], dim=-1)
            hidden = self.mlp(x)
            score = torch.nn.functional.softplus(self.score_head(hidden))
            ctx = torch.tanh(self.ctx_head(hidden))
            return {
                "score": score,
                "features": torch.cat([score, ctx], dim=-1),
            }
else:  # pragma: no cover
    class BarrierHead:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
