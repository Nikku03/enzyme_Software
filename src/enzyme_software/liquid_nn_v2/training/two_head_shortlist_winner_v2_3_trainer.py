from __future__ import annotations

from collections import OrderedDict

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch
from enzyme_software.liquid_nn_v2.training.two_head_shortlist_winner_v2_2_trainer import (
    TwoHeadShortlistWinnerV2_2Trainer,
)


if TORCH_AVAILABLE:
    class TwoHeadShortlistWinnerV2_3Trainer(TwoHeadShortlistWinnerV2_2Trainer):
        def __init__(
            self,
            *,
            model,
            winner_head,
            learning_rate: float = 1.0e-4,
            weight_decay: float = 1.0e-4,
            max_grad_norm: float = 5.0,
            frozen_shortlist_topk: int = 6,
            winner_v2_3_use_existing_candidate_features: bool = True,
            winner_v2_3_use_score_gap_features: bool = True,
            winner_v2_3_use_rank_features: bool = True,
            winner_v2_3_use_normalized_score_features: bool = True,
            winner_v2_3_use_pairwise_features: bool = False,
            winner_v2_3_use_graph_local_features: bool = False,
            winner_v2_3_use_3d_local_features: bool = False,
            winner_v2_3_use_extra_candidate_features: bool = False,
            winner_v2_3_use_soft_multi_positive_targets: bool = False,
            winner_v2_3_use_source_weighting: bool = False,
            winner_v2_3_use_source_oversampling: bool = False,
            winner_v2_3_train_only_on_hits: bool = True,
            winner_v2_3_loss_weight: float = 1.0,
            winner_v2_3_hard_source_weight: float = 2.0,
            winner_v2_3_normal_source_weight: float = 1.0,
            winner_v2_3_hard_sources: str = "attnsom,cyp_dbs_external",
            winner_v2_3_log_feature_summary: bool = True,
            shortlist_checkpoint_path: str = "",
            device=None,
        ):
            super().__init__(
                model=model,
                winner_head=winner_head,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                frozen_shortlist_topk=frozen_shortlist_topk,
                winner_v2_2_use_existing_candidate_features=winner_v2_3_use_existing_candidate_features,
                winner_v2_2_use_score_gap_features=winner_v2_3_use_score_gap_features,
                winner_v2_2_use_rank_features=winner_v2_3_use_rank_features,
                winner_v2_2_use_normalized_score_features=winner_v2_3_use_normalized_score_features,
                winner_v2_2_use_pairwise_features=winner_v2_3_use_pairwise_features,
                winner_v2_2_use_graph_local_features=winner_v2_3_use_graph_local_features,
                winner_v2_2_use_3d_local_features=winner_v2_3_use_3d_local_features,
                winner_v2_2_use_extra_candidate_features=winner_v2_3_use_extra_candidate_features,
                winner_v2_2_use_soft_multi_positive_targets=winner_v2_3_use_soft_multi_positive_targets,
                winner_v2_2_train_only_on_hits=winner_v2_3_train_only_on_hits,
                winner_v2_2_loss_weight=winner_v2_3_loss_weight,
                winner_v2_2_use_source_weighting=winner_v2_3_use_source_weighting,
                winner_v2_2_hard_source_weight=winner_v2_3_hard_source_weight,
                winner_v2_2_normal_source_weight=winner_v2_3_normal_source_weight,
                winner_v2_2_hard_sources=winner_v2_3_hard_sources,
                winner_v2_2_log_source_weight_stats=winner_v2_3_log_feature_summary,
                shortlist_checkpoint_path=shortlist_checkpoint_path,
                device=device,
            )
            self.winner_v2_3_feature_groups_enabled = OrderedDict(
                [
                    ("existing_candidate_features", bool(winner_v2_3_use_existing_candidate_features)),
                    ("score_gap_features", bool(winner_v2_3_use_score_gap_features)),
                    ("rank_features", bool(winner_v2_3_use_rank_features)),
                    ("normalized_score_features", bool(winner_v2_3_use_normalized_score_features)),
                    ("pairwise_features", bool(winner_v2_3_use_pairwise_features)),
                    ("graph_local_features", bool(winner_v2_3_use_graph_local_features)),
                    ("3d_local_features", bool(winner_v2_3_use_3d_local_features)),
                    ("extra_candidate_features", bool(winner_v2_3_use_extra_candidate_features)),
                ]
            )
            self.winner_v2_3_use_soft_multi_positive_targets = bool(winner_v2_3_use_soft_multi_positive_targets)
            self.winner_v2_3_use_source_weighting = bool(winner_v2_3_use_source_weighting)
            self.winner_v2_3_use_source_oversampling = bool(winner_v2_3_use_source_oversampling)
            self.winner_v2_3_log_feature_summary = bool(winner_v2_3_log_feature_summary)
            self.trainable_module_summary = [
                {
                    "name": "winner_head_v2_3",
                    "lr": float(self.learning_rate),
                    "param_count": int(sum(param.numel() for param in self.winner_head.parameters() if param.requires_grad)),
                }
            ]

        def _finalize_epoch_metrics(self, **kwargs):
            metrics = super()._finalize_epoch_metrics(**kwargs)
            metrics["winner_v2_3_feature_groups_enabled"] = dict(self.winner_v2_3_feature_groups_enabled)
            metrics["winner_v2_3_source_weighting_enabled"] = bool(self.winner_v2_3_use_source_weighting)
            metrics["winner_v2_3_soft_multi_positive_enabled"] = bool(self.winner_v2_3_use_soft_multi_positive_targets)
            metrics["winner_v2_3_source_oversampling_enabled"] = bool(self.winner_v2_3_use_source_oversampling)
            return metrics
else:  # pragma: no cover
    class TwoHeadShortlistWinnerV2_3Trainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
