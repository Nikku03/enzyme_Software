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
            self.extra_pair_dim = 10
            self.candidate_encoder = nn.Sequential(
                nn.Linear(self.feature_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )
            self.pair_mlp = nn.Sequential(
                nn.Linear((self.hidden_dim * 4) + self.extra_pair_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
            )
            self.pair_head = nn.Linear(self.hidden_dim, 1)

        @staticmethod
        def _comparison_mask(candidate_mask, proposal_scores, *, compare_top_n: int | None = None, local_rival_mask=None):
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
            if local_rival_mask is not None:
                pair_mask = pair_mask & (local_rival_mask > 0.5)
            diag = torch.eye(int(candidate_mask.shape[1]), device=candidate_mask.device, dtype=torch.bool).unsqueeze(0)
            return pair_mask & (~diag)

        def forward(
            self,
            candidate_features,
            candidate_mask,
            proposal_scores,
            *,
            compare_top_n: int | None = None,
            candidate_local_rival_mask=None,
            candidate_graph_distance=None,
            candidate_3d_distance=None,
            candidate_same_ring_system=None,
            candidate_same_topology_role=None,
            candidate_same_chem_family=None,
            candidate_branch_bulk=None,
            candidate_exposed_span=None,
            candidate_anti_score=None,
        ):
            mask = candidate_mask.float()
            encoded = self.candidate_encoder(candidate_features)
            left = encoded.unsqueeze(2)
            right = encoded.unsqueeze(1)
            score_diff = proposal_scores.unsqueeze(2) - proposal_scores.unsqueeze(1)
            def _pairwise_diff(value):
                if value is None:
                    return torch.zeros_like(score_diff)
                return value.unsqueeze(2) - value.unsqueeze(1)
            def _pairwise_matrix(value, default_fill: float = 0.0):
                if value is None:
                    return torch.full_like(score_diff, fill_value=float(default_fill))
                return value
            graph_distance = _pairwise_matrix(candidate_graph_distance, default_fill=99.0) / 6.0
            geom_distance = _pairwise_matrix(candidate_3d_distance, default_fill=99.0) / 8.0
            same_ring = _pairwise_matrix(candidate_same_ring_system, default_fill=0.0)
            same_role = _pairwise_matrix(candidate_same_topology_role, default_fill=0.0)
            same_family = _pairwise_matrix(candidate_same_chem_family, default_fill=0.0)
            branch_bulk_diff = _pairwise_diff(candidate_branch_bulk)
            exposed_span_diff = _pairwise_diff(candidate_exposed_span)
            anti_score_diff = _pairwise_diff(candidate_anti_score)
            pair_input = torch.cat(
                [
                    left.expand(-1, -1, encoded.shape[1], -1),
                    right.expand(-1, encoded.shape[1], -1, -1),
                    (left - right),
                    (left * right),
                    score_diff.unsqueeze(-1),
                    score_diff.abs().unsqueeze(-1),
                    graph_distance.unsqueeze(-1),
                    geom_distance.unsqueeze(-1),
                    same_ring.unsqueeze(-1),
                    same_role.unsqueeze(-1),
                    same_family.unsqueeze(-1),
                    branch_bulk_diff.unsqueeze(-1),
                    exposed_span_diff.unsqueeze(-1),
                    anti_score_diff.unsqueeze(-1),
                ],
                dim=-1,
            )
            pair_hidden = self.pair_mlp(pair_input)
            pair_logits = self.pair_head(pair_hidden).squeeze(-1)
            comparison_mask = self._comparison_mask(
                mask,
                proposal_scores,
                compare_top_n=compare_top_n,
                local_rival_mask=candidate_local_rival_mask,
            )
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
