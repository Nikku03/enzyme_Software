from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class CandidateWinnerModel(nn.Module):
        def __init__(
            self,
            *,
            feature_dim: int,
            hidden_dim: int = 128,
            dropout: float = 0.10,
        ):
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.hidden_dim = max(32, int(hidden_dim))
            self.candidate_encoder = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )
            self.context_score = nn.Linear(self.hidden_dim, 1)
            self.compare = nn.Sequential(
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )
            self.winner_head = nn.Linear(self.hidden_dim, 1)

        def forward(self, candidate_features, candidate_mask):
            mask = candidate_mask.float()
            encoded = self.candidate_encoder(candidate_features)
            masked_scores = self.context_score(encoded).squeeze(-1)
            masked_scores = masked_scores.masked_fill(mask <= 0.5, float("-inf"))
            alpha = torch.softmax(masked_scores, dim=1).unsqueeze(-1)
            alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
            context = torch.sum(alpha * encoded, dim=1, keepdim=True).expand_as(encoded)
            compared = self.compare(torch.cat([encoded, context, encoded - context, encoded * context], dim=-1))
            logits = self.winner_head(compared).squeeze(-1)
            logits = logits.masked_fill(mask <= 0.5, -20.0)
            return {
                "winner_logits": logits,
                "winner_probs": torch.softmax(logits, dim=1),
                "attention": alpha.squeeze(-1),
            }
else:  # pragma: no cover
    class CandidateWinnerModel:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
