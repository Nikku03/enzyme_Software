from __future__ import annotations

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class PairwiseSiteTournamentModel(nn.Module):
        def __init__(
            self,
            *,
            feature_dim: int,
            hidden_dim: int = 128,
            dropout: float = 0.10,
            blend_weight: float = 0.75,
        ):
            super().__init__()
            self.feature_dim = int(feature_dim)
            self.hidden_dim = max(32, int(hidden_dim))
            self.blend_weight = float(blend_weight)
            self.candidate_encoder = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )
            self.pair_mlp = nn.Sequential(
                nn.Linear((self.hidden_dim * 4) + 2, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )
            self.pair_head = nn.Linear(self.hidden_dim, 1)

        @staticmethod
        def _comparison_mask(candidate_mask, proposal_scores, *, compare_top_n: int | None = None):
            valid = candidate_mask > 0.5
            if compare_top_n is None or int(compare_top_n) <= 0:
                compare_candidates = valid
            else:
                n = int(candidate_mask.shape[1])
                k = min(int(compare_top_n), n)
                top_idx = torch.argsort(proposal_scores.masked_fill(~valid, float("-inf")), dim=1, descending=True)
                compare_candidates = torch.zeros_like(valid)
                compare_candidates.scatter_(1, top_idx[:, :k], True)
                compare_candidates = compare_candidates & valid
            pair_mask = compare_candidates.unsqueeze(2) & compare_candidates.unsqueeze(1)
            diag = torch.eye(int(candidate_mask.shape[1]), device=candidate_mask.device, dtype=torch.bool).unsqueeze(0)
            return pair_mask & (~diag)

        def forward(self, candidate_features, candidate_mask, proposal_scores, *, compare_top_n: int | None = None):
            mask = candidate_mask.float()
            encoded = self.candidate_encoder(candidate_features)
            left = encoded.unsqueeze(2)
            right = encoded.unsqueeze(1)
            score_diff = proposal_scores.unsqueeze(2) - proposal_scores.unsqueeze(1)
            pair_input = torch.cat(
                [
                    left.expand(-1, -1, encoded.shape[1], -1),
                    right.expand(-1, encoded.shape[1], -1, -1),
                    (left - right),
                    (left * right),
                    score_diff.unsqueeze(-1),
                    score_diff.abs().unsqueeze(-1),
                ],
                dim=-1,
            )
            pair_hidden = self.pair_mlp(pair_input)
            pair_logits = self.pair_head(pair_hidden).squeeze(-1)
            comparison_mask = self._comparison_mask(mask, proposal_scores, compare_top_n=compare_top_n)
            pair_logits = pair_logits.masked_fill(~comparison_mask, 0.0)
            pair_probs = torch.sigmoid(pair_logits)
            opponent_count = comparison_mask.sum(dim=-1).clamp_min(1)
            tournament_margin = (((pair_probs - 0.5) * 2.0) * comparison_mask.float()).sum(dim=-1) / opponent_count.float()
            final_scores = proposal_scores + (float(self.blend_weight) * tournament_margin)
            final_scores = final_scores.masked_fill(mask <= 0.5, -20.0)
            tournament_margin = tournament_margin.masked_fill(mask <= 0.5, 0.0)
            return {
                "pair_logits": pair_logits,
                "pair_probs": pair_probs,
                "comparison_mask": comparison_mask,
                "tournament_margin": tournament_margin,
                "final_scores": final_scores,
                "encoded": encoded,
            }
else:  # pragma: no cover
    class PairwiseSiteTournamentModel:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
