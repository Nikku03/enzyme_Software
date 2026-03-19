from __future__ import annotations

from typing import Dict, Tuple

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, nn, require_torch, torch


if TORCH_AVAILABLE:
    class MultiHeadMetaLearner(nn.Module):
        """Multi-head meta-learner that routes toward the best-performing pillar.

        For site prediction: per-atom routing uses both atom chemistry AND the
        actual scores each model produced, so the pillar scoring an atom highest
        gets amplified for that atom.

        For CYP prediction: routing uses global chemistry AND model CYP
        confidence, so the most-confident model dominates.
        """

        def __init__(
            self,
            *,
            n_models: int = 3,
            n_cyp: int = 5,
            atom_feature_dim: int = 11,
            global_feature_dim: int = 19,
            hidden_dim: int = 32,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_models = int(n_models)
            self.n_cyp = int(n_cyp)

            # Site attention: conditioned on atom chemistry + raw scores from all models.
            # This lets the head see which model is scoring this atom high and route toward it.
            self.site_attention = nn.Sequential(
                nn.Linear(atom_feature_dim + n_models, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_models),
            )
            self.site_refine = nn.Sequential(
                nn.Linear(atom_feature_dim + self.n_models, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

            # CYP attention: conditioned on global features + all model CYP probs.
            # This lets the head see which model is most confident and route toward it.
            self.cyp_attention = nn.Sequential(
                nn.Linear(global_feature_dim + n_models * n_cyp, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.n_models),
            )
            self.cyp_refine = nn.Sequential(
                nn.Linear(global_feature_dim + self.n_models * self.n_cyp, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.n_cyp),
            )

            # Learned attention temperatures (softness of per-model weighting)
            self.site_temperature = nn.Parameter(torch.ones(1))
            self.cyp_temperature = nn.Parameter(torch.ones(1))

            # Score-routing temperatures: controls how sharply we select the highest-scoring pillar.
            # Initialized to 0.1 so routing starts sharp (lower = more winner-take-all).
            self.site_score_temperature = nn.Parameter(torch.full((1,), 0.1))
            self.cyp_conf_temperature = nn.Parameter(torch.full((1,), 0.1))

            self.confidence_head = nn.Sequential(
                nn.Linear(atom_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            atom_features: torch.Tensor,
            global_features: torch.Tensor,
            site_scores_raw: torch.Tensor,
            cyp_probs_raw: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
            # ------------------------------------------------------------------ #
            # Site prediction
            # ------------------------------------------------------------------ #

            # Path 1 – learned attention conditioned on atom chemistry + raw scores.
            # The network can see which model is scoring this atom high, so it can
            # learn to route toward that pillar for this atom.
            site_attn_logits = self.site_attention(
                torch.cat([atom_features, site_scores_raw], dim=-1)
            )
            site_attn = torch.softmax(
                site_attn_logits / self.site_temperature.clamp_min(1.0e-3), dim=-1
            )
            site_attended = (site_scores_raw * site_attn).sum(dim=-1)

            # Path 2 – score-based routing: per atom, soft-select the pillar with
            # the highest score.  At low temperature this approaches argmax.
            score_gate = torch.softmax(
                site_scores_raw / self.site_score_temperature.clamp_min(0.01), dim=-1
            )
            site_best = (site_scores_raw * score_gate).sum(dim=-1)

            # Combine: score-based routing is primary (0.7), learned attention corrects (0.3).
            # Before retraining, score-based routing is always sensible; learned attention is random.
            # After retraining, the learned attention learns to refine routing per atom type.
            site_combined = 0.3 * site_attended + 0.7 * site_best
            site_refine = self.site_refine(
                torch.cat([atom_features, site_scores_raw], dim=-1)
            ).squeeze(-1)
            site_scores = site_combined + 0.3 * site_refine

            # ------------------------------------------------------------------ #
            # CYP prediction
            # ------------------------------------------------------------------ #

            # Path 1 – learned attention conditioned on global features + all CYP probs.
            # The network can see each model's full CYP distribution and learn to
            # route toward the most informative pillar.
            cyp_attn_input = torch.cat([global_features, cyp_probs_raw.reshape(-1)], dim=0)
            cyp_attn_logits = self.cyp_attention(cyp_attn_input)
            cyp_attn = torch.softmax(
                cyp_attn_logits / self.cyp_temperature.clamp_min(1.0e-3), dim=-1
            )

            # Path 2 – confidence-based routing: route toward the model that is
            # most confident (highest max-probability across CYP classes).
            cyp_confidence = cyp_probs_raw.max(dim=-1).values  # [n_models]
            conf_gate = torch.softmax(
                cyp_confidence / self.cyp_conf_temperature.clamp_min(0.01), dim=-1
            )

            # Blend: confidence routing is primary (0.7), learned attention corrects (0.3).
            final_cyp_attn = 0.3 * cyp_attn + 0.7 * conf_gate
            cyp_combined = (cyp_probs_raw * final_cyp_attn.unsqueeze(-1)).sum(dim=0)
            cyp_refine = self.cyp_refine(
                torch.cat([global_features, cyp_probs_raw.reshape(-1)], dim=0)
            )
            cyp_logits = torch.log(cyp_combined.clamp_min(1.0e-8)) + 0.3 * cyp_refine

            stats = {
                "site_attention": site_attn.mean(dim=0),
                "site_attention_std": site_attn.std(dim=0, unbiased=False),
                "cyp_attention": final_cyp_attn,
                "confidence": self.confidence_head(atom_features).mean(),
                "site_temperature": self.site_temperature.detach(),
                "cyp_temperature": self.cyp_temperature.detach(),
                "site_score_temperature": self.site_score_temperature.detach(),
                "cyp_conf_temperature": self.cyp_conf_temperature.detach(),
            }
            return site_scores, cyp_logits, stats

else:  # pragma: no cover
    class MultiHeadMetaLearner:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
