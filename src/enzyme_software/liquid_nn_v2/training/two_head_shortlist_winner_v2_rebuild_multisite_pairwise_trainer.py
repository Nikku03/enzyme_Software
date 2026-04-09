from __future__ import annotations

from collections import OrderedDict, defaultdict

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.training.hard_negative_mining import compute_hard_negative_margin_loss
from enzyme_software.liquid_nn_v2.training.loss import AdaptiveLossV2
from enzyme_software.liquid_nn_v2.training.two_head_shortlist_winner_v2_rebuild_trainer import (
    TwoHeadShortlistWinnerV2RebuildTrainer,
)


if TORCH_AVAILABLE:
    class TwoHeadShortlistWinnerV2RebuildMultisitePairwiseTrainer(TwoHeadShortlistWinnerV2RebuildTrainer):
        SOURCE_VOCAB = ("unknown", "az120", "cyp_dbs_external", "drugbank", "metxbiodb")

        def __init__(
            self,
            *,
            model,
            winner_head,
            learning_rate: float = 1.0e-4,
            weight_decay: float = 1.0e-4,
            max_grad_norm: float = 5.0,
            frozen_shortlist_topk: int = 6,
            winner_v2_rebuild_loss_weight: float = 1.0,
            shortlist_checkpoint_path: str = "",
            winner_use_multi_positive_targets: bool = True,
            winner_multi_positive_mode: str = "softmax_uniform",
            winner_multi_positive_only_for_multisite: bool = True,
            winner_multisite_loss_weight: float = 1.0,
            winner_enable_pairwise_ranking: bool = True,
            winner_pairwise_margin: float = 0.2,
            winner_pairwise_loss_weight: float = 0.5,
            winner_pairwise_sample_mode: str = "hard_false_only",
            winner_use_source_embedding: bool = True,
            winner_source_embedding_dim: int = 8,
            winner_use_source_bias: bool = True,
            shortlist_enable_hard_negative_emphasis: bool = False,
            shortlist_hard_negative_rank_min: int = 2,
            shortlist_hard_negative_rank_max: int = 12,
            shortlist_hard_negative_loss_weight: float = 0.0,
            shortlist_hard_negative_mode: str = "top_false",
            shortlist_pairwise_margin: float = 0.20,
            shortlist_pairwise_loss_weight: float = 0.0,
            shortlist_hard_negative_max_per_true: int = 3,
            shortlist_hard_negative_sample_mode: str = "top_false_only",
            device=None,
        ):
            super().__init__(
                model=model,
                winner_head=winner_head,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                frozen_shortlist_topk=frozen_shortlist_topk,
                winner_v2_rebuild_loss_weight=winner_v2_rebuild_loss_weight,
                shortlist_checkpoint_path=shortlist_checkpoint_path,
                device=device,
            )
            self.winner_use_multi_positive_targets = bool(winner_use_multi_positive_targets)
            self.winner_multi_positive_mode = str(winner_multi_positive_mode or "softmax_uniform").strip().lower()
            self.winner_multi_positive_only_for_multisite = bool(winner_multi_positive_only_for_multisite)
            self.winner_multisite_loss_weight = max(0.0, float(winner_multisite_loss_weight))
            self.winner_enable_pairwise_ranking = bool(winner_enable_pairwise_ranking)
            self.winner_pairwise_margin = max(0.0, float(winner_pairwise_margin))
            self.winner_pairwise_loss_weight = max(0.0, float(winner_pairwise_loss_weight))
            self.winner_pairwise_sample_mode = str(winner_pairwise_sample_mode or "hard_false_only").strip().lower()
            self.winner_use_source_embedding = bool(winner_use_source_embedding)
            self.winner_source_embedding_dim = max(1, int(winner_source_embedding_dim))
            self.winner_use_source_bias = bool(winner_use_source_bias)
            self.shortlist_hard_negative_rank_min = max(1, int(shortlist_hard_negative_rank_min))
            self.shortlist_hard_negative_rank_max = max(self.shortlist_hard_negative_rank_min, int(shortlist_hard_negative_rank_max))
            self.shortlist_hard_negative_loss_weight = max(0.0, float(shortlist_hard_negative_loss_weight))
            self.shortlist_hard_negative_mode = str(shortlist_hard_negative_mode or "top_false").strip().lower()
            self.shortlist_pairwise_margin = max(0.0, float(shortlist_pairwise_margin))
            self.shortlist_pairwise_loss_weight = max(
                0.0,
                float(shortlist_pairwise_loss_weight if float(shortlist_pairwise_loss_weight) > 0.0 else shortlist_hard_negative_loss_weight),
            )
            self.shortlist_hard_negative_max_per_true = max(1, int(shortlist_hard_negative_max_per_true))
            self.shortlist_hard_negative_sample_mode = str(
                shortlist_hard_negative_sample_mode or "top_false_only"
            ).strip().lower()
            self.shortlist_hard_negative_requested_flag = bool(shortlist_enable_hard_negative_emphasis)
            self.shortlist_enable_hard_negative_emphasis = bool(
                self.shortlist_hard_negative_requested_flag
                or self.shortlist_hard_negative_loss_weight > 0.0
                or self.shortlist_pairwise_loss_weight > 0.0
            )
            self.source_vocab = tuple(self.SOURCE_VOCAB)
            self.source_to_index = {name: idx for idx, name in enumerate(self.source_vocab)}
            self._active_split_name = "unknown"
            self._shortlist_hard_negative_warning_emitted = False
            base_impl = getattr(getattr(self.model, "base_lnn", None), "impl", None)
            self.proposer_head = getattr(base_impl, "site_head", None)
            self.source_site_heads = getattr(base_impl, "source_site_heads", None)
            self.shortlist_loss_wrapper = None
            self.shortlist_site_shortlist_weight = 0.0
            self.shortlist_site_hard_negative_weight = 0.0
            self.shortlist_use_top_score_hard_neg = False
            self.shortlist_use_graph_local_hard_neg = False
            self.shortlist_use_3d_local_hard_neg = False
            self.shortlist_use_rank_weighted_shortlist = False
            self.shortlist_use_rank_weighted_hard_neg = False
            if self.shortlist_enable_hard_negative_emphasis:
                self._configure_shortlist_hard_negative_training()
            self.trainable_module_summary = self._build_trainable_module_summary()
            self.restore_summary = OrderedDict(self.restore_summary)
            self.restore_summary["multisite_pairwise"] = OrderedDict(
                [
                    ("winner_use_multi_positive_targets", bool(self.winner_use_multi_positive_targets)),
                    ("winner_multi_positive_mode", str(self.winner_multi_positive_mode)),
                    ("winner_multi_positive_only_for_multisite", bool(self.winner_multi_positive_only_for_multisite)),
                    ("winner_multisite_loss_weight", float(self.winner_multisite_loss_weight)),
                    ("winner_enable_pairwise_ranking", bool(self.winner_enable_pairwise_ranking)),
                    ("winner_pairwise_margin", float(self.winner_pairwise_margin)),
                    ("winner_pairwise_loss_weight", float(self.winner_pairwise_loss_weight)),
                    ("winner_pairwise_sample_mode", str(self.winner_pairwise_sample_mode)),
                    ("winner_use_source_embedding", bool(self.winner_use_source_embedding)),
                    ("winner_source_embedding_dim", int(self.winner_source_embedding_dim)),
                    ("winner_use_source_bias", bool(self.winner_use_source_bias)),
                    ("shortlist_hard_negative_requested_flag", bool(self.shortlist_hard_negative_requested_flag)),
                    ("shortlist_enable_hard_negative_emphasis", bool(self.shortlist_enable_hard_negative_emphasis)),
                    ("shortlist_hard_negative_rank_window", [int(self.shortlist_hard_negative_rank_min), int(self.shortlist_hard_negative_rank_max)]),
                    ("shortlist_hard_negative_loss_weight", float(self.shortlist_hard_negative_loss_weight)),
                    ("shortlist_hard_negative_mode", str(self.shortlist_hard_negative_mode)),
                    ("shortlist_pairwise_margin", float(self.shortlist_pairwise_margin)),
                    ("shortlist_pairwise_loss_weight", float(self.shortlist_pairwise_loss_weight)),
                    ("shortlist_hard_negative_max_per_true", int(self.shortlist_hard_negative_max_per_true)),
                    ("shortlist_hard_negative_sample_mode", str(self.shortlist_hard_negative_sample_mode)),
                    ("source_vocab", list(self.source_vocab)),
                ]
            )

        def _iter_trainable_source_head_modules(self):
            if self.source_site_heads is None:
                return []
            if hasattr(self.source_site_heads, "items"):
                return list(self.source_site_heads.items())
            return []

        def _rebuild_optimizer(self):
            param_groups = [
                {
                    "params": [param for param in self.winner_head.parameters() if param.requires_grad],
                    "lr": float(self.learning_rate),
                    "weight_decay": float(self.weight_decay),
                }
            ]
            proposer_params = []
            if self.proposer_head is not None:
                proposer_params.extend([param for param in self.proposer_head.parameters() if param.requires_grad])
            for _source_name, source_head in self._iter_trainable_source_head_modules():
                proposer_params.extend([param for param in source_head.parameters() if param.requires_grad])
            if proposer_params:
                param_groups.append(
                    {
                        "params": proposer_params,
                        "lr": float(self.learning_rate),
                        "weight_decay": float(self.weight_decay),
                    }
                )
            self.optimizer = torch.optim.AdamW(param_groups)

        def _build_trainable_module_summary(self):
            summary = [
                {
                    "name": "winner_head_v2_rebuild_multisite_pairwise",
                    "lr": float(self.learning_rate),
                    "param_count": int(sum(param.numel() for param in self.winner_head.parameters() if param.requires_grad)),
                }
            ]
            if self.proposer_head is not None and any(param.requires_grad for param in self.proposer_head.parameters()):
                summary.append(
                    {
                        "name": "base_lnn.impl.site_head",
                        "lr": float(self.learning_rate),
                        "param_count": int(sum(param.numel() for param in self.proposer_head.parameters() if param.requires_grad)),
                    }
                )
            source_param_count = 0
            for _source_name, source_head in self._iter_trainable_source_head_modules():
                source_param_count += int(sum(param.numel() for param in source_head.parameters() if param.requires_grad))
            if source_param_count > 0:
                summary.append(
                    {
                        "name": "base_lnn.impl.source_site_heads",
                        "lr": float(self.learning_rate),
                        "param_count": int(source_param_count),
                    }
                )
            return summary

        def _configure_shortlist_hard_negative_training(self):
            if self.proposer_head is not None:
                for param in self.proposer_head.parameters():
                    param.requires_grad = True
            for _source_name, source_head in self._iter_trainable_source_head_modules():
                for param in source_head.parameters():
                    param.requires_grad = True
            shortlist_topk = max(1, int(self.shortlist_hard_negative_rank_max - self.shortlist_hard_negative_rank_min + 1))
            mode = str(self.shortlist_hard_negative_mode or "top_false").strip().lower()
            sample_mode = str(self.shortlist_hard_negative_sample_mode or "top_false_only").strip().lower()
            use_rank_window = mode in {"rank_window", "top_false_plus_rank_window"}
            use_pairwise = mode in {"top_false", "top_false_plus_rank_window", "near_true_neighbors"}
            use_top_score = False
            use_graph_local = False
            use_3d_local = False
            if use_pairwise:
                if sample_mode == "all_hard_false":
                    use_top_score = True
                    use_graph_local = True
                    use_3d_local = True
                elif mode == "near_true_neighbors":
                    use_graph_local = True
                    use_3d_local = True
                else:
                    use_top_score = True
            self.shortlist_site_shortlist_weight = (
                float(self.shortlist_hard_negative_loss_weight) if use_rank_window else 0.0
            )
            self.shortlist_site_hard_negative_weight = (
                float(self.shortlist_pairwise_loss_weight) if use_pairwise else 0.0
            )
            self.shortlist_use_top_score_hard_neg = bool(use_top_score)
            self.shortlist_use_graph_local_hard_neg = bool(use_graph_local)
            self.shortlist_use_3d_local_hard_neg = bool(use_3d_local)
            self.shortlist_use_rank_weighted_shortlist = bool(use_rank_window)
            self.shortlist_use_rank_weighted_hard_neg = bool(
                self.shortlist_hard_negative_mode in {"top_false_plus_rank_window", "rank_window"}
            )
            model_config = getattr(self.model, "config", None)
            self.shortlist_loss_wrapper = AdaptiveLossV2(
                tau_reg_weight=0.0,
                energy_loss_weight=0.0,
                deliberation_loss_weight=0.0,
                site_label_smoothing=float(getattr(model_config, "site_label_smoothing", 0.0)),
                site_top1_margin_weight=float(getattr(model_config, "site_top1_margin_weight", 0.0)),
                site_top1_margin_value=float(getattr(model_config, "site_top1_margin_value", 0.5)),
                site_ranking_weight=float(getattr(model_config, "site_ranking_weight", 0.5)),
                site_hard_negative_fraction=float(getattr(model_config, "site_hard_negative_fraction", 0.5)),
                site_top1_margin_topk=int(getattr(model_config, "site_top1_margin_topk", 1)),
                site_top1_margin_decay=float(getattr(model_config, "site_top1_margin_decay", 1.0)),
                site_cover_weight=float(getattr(model_config, "site_cover_weight", 0.0)),
                site_cover_margin=float(getattr(model_config, "site_cover_margin", 0.20)),
                site_cover_topk=int(getattr(model_config, "site_cover_topk", 5)),
                site_shortlist_weight=float(self.shortlist_site_shortlist_weight),
                site_shortlist_temperature=float(getattr(model_config, "site_shortlist_temperature", 0.70)),
                site_shortlist_topk=int(shortlist_topk),
                site_use_rank_weighted_shortlist=bool(self.shortlist_use_rank_weighted_shortlist),
                site_hard_negative_weight=float(self.shortlist_site_hard_negative_weight),
                site_hard_negative_margin=float(self.shortlist_pairwise_margin),
                site_hard_negative_max_per_true=int(self.shortlist_hard_negative_max_per_true),
                site_use_top_score_hard_neg=bool(self.shortlist_use_top_score_hard_neg),
                site_use_graph_local_hard_neg=bool(self.shortlist_use_graph_local_hard_neg),
                site_use_3d_local_hard_neg=bool(self.shortlist_use_3d_local_hard_neg),
                site_use_rank_weighted_hard_neg=bool(self.shortlist_use_rank_weighted_hard_neg),
            ).to(self.device)
            for param in self.shortlist_loss_wrapper.parameters():
                param.requires_grad = False
            self.shortlist_loss_wrapper.eval()
            self._rebuild_optimizer()
            self.frozen_module_summary = [
                name
                for name in list(self.frozen_module_summary)
                if name not in {"base_lnn.impl.site_head", "base_lnn.impl.source_site_heads"}
            ]

        def _forward_shortlist_provider(self, batch):
            self.model.eval()
            if self.shortlist_enable_hard_negative_emphasis:
                if self.proposer_head is not None:
                    self.proposer_head.train()
                for _source_name, source_head in self._iter_trainable_source_head_modules():
                    source_head.train()
                return self.model(batch)
            return super()._forward_shortlist_provider(batch)

        def _shortlist_hard_negative_epoch_stats(
            self,
            *,
            merged_site_scores,
            merged_site_labels,
            merged_site_batch,
            merged_site_supervision_mask,
            merged_candidate_mask,
            merged_edge_index,
            merged_atom_coordinates,
        ):
            if not self.shortlist_enable_hard_negative_emphasis:
                return {
                    "shortlist_pairwise_example_count": 0.0,
                    "shortlist_hard_negative_count": 0.0,
                    "shortlist_top_false_beats_true_fraction": 0.0,
                }
            hard_negative_loss, hard_negative_stats = compute_hard_negative_margin_loss(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                margin=float(self.shortlist_pairwise_margin),
                supervision_mask=merged_site_supervision_mask,
                candidate_mask=merged_candidate_mask,
                edge_index=merged_edge_index,
                atom_coordinates=merged_atom_coordinates,
                use_top_score=bool(self.shortlist_use_top_score_hard_neg),
                use_graph_local=bool(self.shortlist_use_graph_local_hard_neg),
                use_3d_local=bool(self.shortlist_use_3d_local_hard_neg),
                max_hard_negs_per_true=int(self.shortlist_hard_negative_max_per_true),
                use_rank_weighting=bool(self.shortlist_use_rank_weighted_hard_neg),
            )
            _ = hard_negative_loss
            return {
                "shortlist_pairwise_example_count": float(hard_negative_stats.get("hard_negative_pair_count", 0.0)),
                "shortlist_hard_negative_count": float(hard_negative_stats.get("hard_negative_pair_count", 0.0)),
                "shortlist_top_false_beats_true_fraction": 1.0
                - float(hard_negative_stats.get("top_score_true_beats_fraction", 0.0)),
            }

        def _shortlist_rank_window_stats(self, shortlist_scores, batch):
            batch_index = batch["batch"].view(-1)
            site_labels = batch["site_labels"].view(-1) > 0.5
            supervision = (
                self._supervision_mask(batch).view(-1) > 0.5
                if self._supervision_mask(batch) is not None
                else torch.ones_like(site_labels, dtype=torch.bool)
            )
            ranking = (
                self._candidate_mask(batch).view(-1) > 0.5
                if self._candidate_mask(batch) is not None
                else torch.ones_like(site_labels, dtype=torch.bool)
            )
            valid = supervision & ranking
            num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
            rank_window_false_count = 0
            rank_window_true_recovery_count = 0
            rank_min = int(self.shortlist_hard_negative_rank_min)
            rank_max = int(self.shortlist_hard_negative_rank_max)
            for mol_idx in range(num_molecules):
                mol_valid = (batch_index == mol_idx) & valid
                if not bool(mol_valid.any()):
                    continue
                mol_scores = shortlist_scores[mol_valid]
                mol_labels = site_labels[mol_valid]
                order = torch.argsort(mol_scores, descending=True)
                ordered_labels = mol_labels[order]
                start = min(max(0, rank_min - 1), int(ordered_labels.numel()))
                end = min(rank_max, int(ordered_labels.numel()))
                if end <= start:
                    continue
                window_labels = ordered_labels[start:end]
                rank_window_false_count += int((~window_labels).sum().item())
                if bool(window_labels.any().item()) and not bool(ordered_labels[:start].any().item()):
                    rank_window_true_recovery_count += 1
            return {
                "shortlist_rank_window_false_count": float(rank_window_false_count),
                "shortlist_rank_window_true_recovery_count": float(rank_window_true_recovery_count),
            }

        def _shortlist_auxiliary_loss(self, masked_shortlist_logits, batch):
            zero = masked_shortlist_logits.sum() * 0.0
            if not self.shortlist_enable_hard_negative_emphasis or self.shortlist_loss_wrapper is None:
                return zero, {
                    "shortlist_loss_base_component": 0.0,
                    "shortlist_loss_hard_negative_component": 0.0,
                    "shortlist_pairwise_example_count": 0.0,
                    "shortlist_hard_negative_count": 0.0,
                }
            shortlist_loss, shortlist_stats = self.shortlist_loss_wrapper.site_loss(
                masked_shortlist_logits,
                batch["site_labels"],
                batch["batch"],
                supervision_mask=self._supervision_mask(batch),
                candidate_mask=self._candidate_mask(batch),
                edge_index=batch.get("edge_index"),
                atom_coordinates=batch.get("atom_coordinates"),
            )
            weighted_shortlist_component = float(self.shortlist_site_shortlist_weight) * float(
                shortlist_stats.get("shortlist_loss", 0.0)
            )
            weighted_pairwise_component = float(self.shortlist_site_hard_negative_weight) * float(
                shortlist_stats.get("hard_negative_loss_raw", 0.0)
            )
            shortlist_hard_negative_component = weighted_shortlist_component + weighted_pairwise_component
            shortlist_loss_base_component = max(
                0.0,
                float(shortlist_stats.get("site_loss", 0.0)) - float(shortlist_hard_negative_component),
            )
            metrics = {
                "shortlist_loss_base_component": float(shortlist_loss_base_component),
                "shortlist_loss_hard_negative_component": float(shortlist_hard_negative_component),
                "shortlist_pairwise_example_count": float(shortlist_stats.get("hard_negative_pair_count", 0.0)),
                "shortlist_hard_negative_count": float(shortlist_stats.get("hard_negative_pair_count", 0.0)),
            }
            return shortlist_loss, metrics

        def _source_index(self, source: str) -> int:
            normalized = str(source or "unknown").strip().lower() or "unknown"
            return int(self.source_to_index.get(normalized, self.source_to_index["unknown"]))

        def _build_winner_examples(self, atom_features, shortlist_scores, batch):
            batch_index = batch["batch"].view(-1)
            site_labels = batch["site_labels"].view(-1) > 0.5
            supervision = (
                self._supervision_mask(batch).view(-1) > 0.5
                if self._supervision_mask(batch) is not None
                else torch.ones_like(site_labels, dtype=torch.bool)
            )
            ranking = (
                self._candidate_mask(batch).view(-1) > 0.5
                if self._candidate_mask(batch) is not None
                else torch.ones_like(site_labels, dtype=torch.bool)
            )
            valid = supervision & ranking
            metadata = list(batch.get("graph_metadata") or [])
            edge_index = batch.get("edge_index")
            atom_coordinates = batch.get("atom_coordinates")
            num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
            examples = []
            candidate_counts = []
            skipped_empty = 0
            multi_positive_shortlist_count = 0
            multisite_example_count = 0
            single_site_example_count = 0

            for mol_idx in range(num_molecules):
                mol_valid = (batch_index == mol_idx) & valid
                if not bool(mol_valid.any()):
                    skipped_empty += 1
                    continue
                mol_indices = torch.nonzero(mol_valid, as_tuple=False).view(-1)
                mol_scores = shortlist_scores[mol_indices]
                mol_labels = site_labels[mol_indices]
                candidate_count = int(mol_indices.numel())
                if candidate_count <= 0:
                    skipped_empty += 1
                    continue

                full_order = torch.argsort(mol_scores, descending=True)
                full_rank_labels = mol_labels[full_order]
                true_rank = 0.0
                if bool(full_rank_labels.any()):
                    true_rank = float(int(torch.nonzero(full_rank_labels, as_tuple=False).view(-1)[0].item()) + 1)

                hit_at_6 = bool(full_rank_labels[: min(6, candidate_count)].any().item())
                hit_at_12 = bool(full_rank_labels[: min(12, candidate_count)].any().item())
                k = min(int(self.frozen_shortlist_topk), candidate_count)
                selected_order = full_order[:k]
                selected_indices = mol_indices[selected_order]
                selected_scores = mol_scores[selected_order]
                selected_labels = mol_labels[selected_order]
                hit_at_train_k = bool(selected_labels.any().item())
                candidate_counts.append(float(k))

                winner_features = self._build_winner_features(
                    atom_features=atom_features,
                    selected_indices=selected_indices,
                    selected_scores=selected_scores,
                    mol_atom_indices=torch.nonzero(batch_index == mol_idx, as_tuple=False).view(-1),
                    edge_index=edge_index,
                    atom_coordinates=atom_coordinates,
                )

                positive_local = torch.nonzero(selected_labels, as_tuple=False).view(-1)
                shortlisted_positive_count = int(positive_local.numel())
                all_positive_count = int(mol_labels.sum().item())
                is_multisite = bool(all_positive_count > 1)
                if is_multisite:
                    multisite_example_count += 1
                else:
                    single_site_example_count += 1
                if shortlisted_positive_count > 1:
                    multi_positive_shortlist_count += 1

                target_distribution = None
                target_index = -1
                use_multi_positive = bool(self.winner_use_multi_positive_targets) and shortlisted_positive_count > 0
                if use_multi_positive and bool(self.winner_multi_positive_only_for_multisite):
                    use_multi_positive = bool(is_multisite)
                if shortlisted_positive_count > 0:
                    if use_multi_positive and shortlisted_positive_count > 1 and self.winner_multi_positive_mode == "softmax_uniform":
                        target_distribution = torch.zeros(k, device=selected_scores.device, dtype=selected_scores.dtype)
                        target_distribution[positive_local] = 1.0 / float(shortlisted_positive_count)
                        target_index = int(positive_local[0].item())
                    else:
                        best_true_local = positive_local[torch.argmax(selected_scores[positive_local])]
                        target_index = int(best_true_local.item())

                source = "unknown"
                if mol_idx < len(metadata) and isinstance(metadata[mol_idx], dict):
                    source = str(metadata[mol_idx].get("source") or metadata[mol_idx].get("data_source") or "").strip().lower() or "unknown"
                source_index = self._source_index(source)
                source_index_tensor = torch.full(
                    (int(selected_indices.numel()),),
                    int(source_index),
                    device=winner_features.device,
                    dtype=torch.long,
                )
                examples.append(
                    {
                        "winner_features": winner_features,
                        "selected_indices": selected_indices,
                        "selected_labels": selected_labels.to(dtype=torch.float32),
                        "selected_scores": selected_scores,
                        "hit": hit_at_train_k,
                        "hit_at_6": hit_at_6,
                        "hit_at_12": hit_at_12,
                        "rescued_by_12": bool(hit_at_12 and not hit_at_6),
                        "true_rank": true_rank,
                        "trainable": bool(hit_at_train_k or (not self.winner_v2_train_only_on_hits)),
                        "target_index": target_index,
                        "target_distribution": target_distribution,
                        "is_multisite": bool(is_multisite),
                        "all_positive_count": int(all_positive_count),
                        "shortlisted_positive_count": int(shortlisted_positive_count),
                        "multi_positive": bool(shortlisted_positive_count > 1),
                        "source": source,
                        "source_index_tensor": source_index_tensor,
                    }
                )

            metrics = {
                "shortlist_candidate_count_mean": float(sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
                "winner_eval_molecule_count": float(sum(1 for ex in examples if ex["hit"])),
                "winner_trainable_molecule_count": float(sum(1 for ex in examples if ex["trainable"] and ex["target_index"] >= 0)),
                "winner_multi_positive_shortlist_count": float(multi_positive_shortlist_count),
                "winner_multi_positive_shortlist_fraction": (
                    float(multi_positive_shortlist_count) / float(max(1, sum(1 for ex in examples if ex["hit"])))
                ),
                "multisite_example_count": float(multisite_example_count),
                "single_site_example_count": float(single_site_example_count),
                "skipped_empty_shortlist_molecules": float(skipped_empty),
            }
            return examples, metrics

        def _winner_logits(self, example):
            return self.winner_head(
                example["winner_features"],
                source_indices=example["source_index_tensor"],
            ).view(-1)

        def _select_pairwise_negative_indices(self, example, positive_local):
            negative_local = torch.nonzero(example["selected_labels"] <= 0.5, as_tuple=False).view(-1)
            if int(negative_local.numel()) <= 0:
                return negative_local
            if self.winner_pairwise_sample_mode == "all_false":
                return negative_local
            negative_scores = example["selected_scores"][negative_local]
            if self.winner_pairwise_sample_mode == "top_false_only":
                best_negative = negative_local[torch.argmax(negative_scores)]
                return best_negative.view(1)
            if int(positive_local.numel()) > 0:
                best_positive_score = torch.max(example["selected_scores"][positive_local])
                hard_mask = negative_scores >= best_positive_score
                hard_negative_local = negative_local[hard_mask]
                if int(hard_negative_local.numel()) > 0:
                    return hard_negative_local
            best_negative = negative_local[torch.argmax(negative_scores)]
            return best_negative.view(1)

        def _winner_loss(self, examples):
            trainable = [ex for ex in examples if ex["trainable"] and ex["target_index"] >= 0]
            if not trainable:
                zero = next(self.winner_head.parameters()).sum() * 0.0
                return zero, {
                    "winner_loss_ce_component": 0.0,
                    "winner_loss_pairwise_component": 0.0,
                    "winner_pairwise_example_count": 0.0,
                    "winner_pairwise_hard_false_count": 0.0,
                }
            total_losses = []
            ce_losses = []
            pairwise_losses = []
            pairwise_example_count = 0
            pairwise_hard_false_count = 0
            for example in trainable:
                logits = self._winner_logits(example)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Non-finite winner multisite-pairwise logits detected")
                target_distribution = example.get("target_distribution")
                if target_distribution is not None:
                    target_distribution = target_distribution.to(device=logits.device, dtype=logits.dtype)
                    target_distribution = target_distribution / target_distribution.sum().clamp_min(1.0e-6)
                    ce_loss = -(target_distribution * F.log_softmax(logits, dim=0)).sum()
                else:
                    target = torch.tensor([int(example["target_index"])], device=logits.device, dtype=torch.long)
                    ce_loss = F.cross_entropy(logits.unsqueeze(0), target)
                if bool(example.get("is_multisite", False)):
                    ce_loss = ce_loss * float(self.winner_multisite_loss_weight)
                ce_losses.append(ce_loss)

                pairwise_loss = logits.sum() * 0.0
                if bool(self.winner_enable_pairwise_ranking) and float(self.winner_pairwise_loss_weight) > 0.0:
                    positive_local = torch.nonzero(example["selected_labels"] > 0.5, as_tuple=False).view(-1)
                    selected_negative_local = self._select_pairwise_negative_indices(example, positive_local)
                    if int(positive_local.numel()) > 0 and int(selected_negative_local.numel()) > 0:
                        pos_logits = logits[positive_local].view(-1, 1)
                        neg_logits = logits[selected_negative_local].view(1, -1)
                        pairwise_margin = float(self.winner_pairwise_margin)
                        pairwise_terms = torch.relu(pairwise_margin - (pos_logits - neg_logits))
                        pairwise_loss = pairwise_terms.mean()
                        pairwise_example_count += 1
                        pairwise_hard_false_count += int(selected_negative_local.numel())
                pairwise_losses.append(pairwise_loss)
                total_losses.append(ce_loss + (float(self.winner_pairwise_loss_weight) * pairwise_loss))
            loss = torch.stack(total_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite winner multisite-pairwise loss detected")
            stats = {
                "winner_loss_ce_component": float(torch.stack(ce_losses).mean().detach().item()) if ce_losses else 0.0,
                "winner_loss_pairwise_component": float(torch.stack(pairwise_losses).mean().detach().item()) if pairwise_losses else 0.0,
                "winner_pairwise_example_count": float(pairwise_example_count),
                "winner_pairwise_hard_false_count": float(pairwise_hard_false_count),
            }
            return loss, stats

        def _run_batch(self, batch):
            outputs = self._forward_shortlist_provider(batch)
            atom_features = outputs.get("atom_features")
            shortlist_logits = outputs.get("site_logits")
            if atom_features is None:
                raise RuntimeError("two_head_shortlist_winner_v2_rebuild_multisite_pairwise requires model outputs['atom_features']")
            if shortlist_logits is None:
                raise RuntimeError("two_head_shortlist_winner_v2_rebuild_multisite_pairwise requires model outputs['site_logits']")
            shortlist_logits = shortlist_logits.view(-1)
            config = getattr(self.model, "config", None)
            from enzyme_software.liquid_nn_v2.training.pairwise_probe import apply_candidate_mask_to_site_logits

            masked_shortlist_logits = apply_candidate_mask_to_site_logits(
                shortlist_logits,
                self._candidate_mask(batch),
                mask_mode=str(getattr(config, "candidate_mask_mode", "hard") or "hard"),
                logit_bias=float(getattr(config, "candidate_mask_logit_bias", 2.0)),
            )
            shortlist_scores = torch.sigmoid(masked_shortlist_logits)
            if not bool(torch.isfinite(shortlist_scores).all()):
                raise FloatingPointError("Non-finite frozen shortlist scores detected")
            examples, shortlist_metrics = self._build_winner_examples(atom_features, shortlist_scores, batch)
            winner_loss, winner_loss_stats = self._winner_loss(examples)
            shortlist_aux_loss, shortlist_aux_stats = self._shortlist_auxiliary_loss(masked_shortlist_logits, batch)
            rank_window_stats = self._shortlist_rank_window_stats(shortlist_scores, batch)
            total_loss = (float(self.winner_v2_loss_weight) * winner_loss) + shortlist_aux_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Non-finite two-head multisite-pairwise total loss detected")
            metrics = {
                "winner_loss": float(winner_loss.detach().item()),
                "shortlist_aux_loss": float(shortlist_aux_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                **shortlist_metrics,
                **shortlist_aux_stats,
                **rank_window_stats,
                **winner_loss_stats,
            }
            return total_loss, shortlist_scores, examples, metrics

        def _maybe_warn_shortlist_aux_inactive(self, metrics):
            if self._active_split_name != "train":
                metrics["shortlist_hard_negative_warning_emitted"] = 0.0
                return
            if not bool(self.shortlist_enable_hard_negative_emphasis):
                metrics["shortlist_hard_negative_warning_emitted"] = 0.0
                return
            if self._shortlist_hard_negative_warning_emitted:
                metrics["shortlist_hard_negative_warning_emitted"] = 1.0
                return
            shortlist_aux_loss = float(metrics.get("shortlist_aux_loss", 0.0))
            shortlist_hard_negative_count = float(metrics.get("shortlist_hard_negative_count", 0.0))
            shortlist_pairwise_example_count = float(metrics.get("shortlist_pairwise_example_count", 0.0))
            shortlist_rank_window_false_count = float(metrics.get("shortlist_rank_window_false_count", 0.0))
            if (
                shortlist_aux_loss == 0.0
                and shortlist_hard_negative_count == 0.0
                and shortlist_pairwise_example_count == 0.0
                and shortlist_rank_window_false_count == 0.0
            ):
                print(
                    "WARNING: shortlist hard-negative emphasis is effectively enabled but produced zero "
                    "auxiliary activity for this epoch. Check candidate masks, supervision masks, or preset wiring.",
                    flush=True,
                )
                self._shortlist_hard_negative_warning_emitted = True
                metrics["shortlist_hard_negative_warning_emitted"] = 1.0
                return
            metrics["shortlist_hard_negative_warning_emitted"] = 0.0

        def _winner_eval_metrics(self, examples):
            winner_eval_count = 0
            winner_top1 = 0
            winner_top2 = 0
            winner_top3 = 0
            end_to_end_top1 = 0
            end_to_end_top3 = 0
            rescued_by_12_count = 0
            winner_miss_among_rescued_by_12_count = 0
            source_rows = defaultdict(
                lambda: {
                    "n": 0,
                    "shortlist_hit": 0.0,
                    "shortlist_hit_at_12": 0.0,
                    "winner_hit": 0.0,
                    "end_to_end_top1": 0.0,
                }
            )
            subset = {
                "multisite": {"total": 0, "hit": 0, "winner_top1": 0, "end_to_end_top1": 0},
                "single_site": {"total": 0, "hit": 0, "winner_top1": 0, "end_to_end_top1": 0},
            }
            total_examples = len(examples)
            for example in examples:
                logits = self._winner_logits(example)
                order = torch.argsort(logits, descending=True)
                labels = example["selected_labels"] > 0.5
                top1_hit = bool(labels[order[0]].item()) if int(order.numel()) > 0 else False
                top3_hit = bool(labels[order[: min(3, int(order.numel()))]].any().item()) if int(order.numel()) > 0 else False
                if top1_hit:
                    end_to_end_top1 += 1
                if top3_hit:
                    end_to_end_top3 += 1
                rescued_by_12 = bool(example.get("rescued_by_12", False))
                if rescued_by_12:
                    rescued_by_12_count += 1
                source_row = source_rows[str(example["source"])]
                source_row["n"] += 1
                source_row["shortlist_hit"] += float(bool(example.get("hit_at_6", example["hit"])))
                source_row["shortlist_hit_at_12"] += float(bool(example.get("hit_at_12", example["hit"])))
                source_row["end_to_end_top1"] += float(top1_hit)

                subset_name = "multisite" if bool(example.get("is_multisite", False)) else "single_site"
                subset[subset_name]["total"] += 1
                subset[subset_name]["end_to_end_top1"] += int(top1_hit)
                if example["hit"]:
                    winner_eval_count += 1
                    winner_top1 += int(top1_hit)
                    winner_top2 += int(bool(labels[order[: min(2, int(order.numel()))]].any().item()))
                    winner_top3 += int(top3_hit)
                    source_row["winner_hit"] += float(top1_hit)
                    subset[subset_name]["hit"] += 1
                    subset[subset_name]["winner_top1"] += int(top1_hit)
                    if rescued_by_12 and not top1_hit:
                        winner_miss_among_rescued_by_12_count += 1

            source_breakdown = {}
            for name, row in sorted(source_rows.items()):
                n = int(row["n"])
                hit_n = sum(1 for ex in examples if str(ex["source"]) == name and ex["hit"])
                source_breakdown[name] = {
                    "n": n,
                    "shortlist_recall_at_6": float(row["shortlist_hit"]) / float(n) if n > 0 else 0.0,
                    "shortlist_recall_at_12": float(row["shortlist_hit_at_12"]) / float(n) if n > 0 else 0.0,
                    "winner_acc_given_hit": float(row["winner_hit"]) / float(hit_n) if hit_n > 0 else 0.0,
                    "end_to_end_top1": float(row["end_to_end_top1"]) / float(n) if n > 0 else 0.0,
                }

            return {
                "winner_acc_given_hit": float(winner_top1) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_acc_given_hit_at_k": float(winner_top1) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_top2_given_hit": float(winner_top2) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_top3_given_hit": float(winner_top3) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_eval_molecule_count": float(winner_eval_count),
                "end_to_end_top1": float(end_to_end_top1) / float(total_examples) if total_examples > 0 else 0.0,
                "end_to_end_top3": float(end_to_end_top3) / float(total_examples) if total_examples > 0 else 0.0,
                "end_to_end_hit_then_win_fraction": float(end_to_end_top1) / float(total_examples) if total_examples > 0 else 0.0,
                "shortlist_rescued_by_12_count": float(rescued_by_12_count),
                "shortlist_rescued_by_12_fraction": float(rescued_by_12_count) / float(total_examples) if total_examples > 0 else 0.0,
                "winner_miss_among_rescued_by_12_count": float(winner_miss_among_rescued_by_12_count),
                "multisite_winner_acc_given_hit": float(subset["multisite"]["winner_top1"]) / float(subset["multisite"]["hit"])
                if subset["multisite"]["hit"] > 0
                else 0.0,
                "single_site_winner_acc_given_hit": float(subset["single_site"]["winner_top1"]) / float(subset["single_site"]["hit"])
                if subset["single_site"]["hit"] > 0
                else 0.0,
                "multisite_end_to_end_top1": float(subset["multisite"]["end_to_end_top1"]) / float(subset["multisite"]["total"])
                if subset["multisite"]["total"] > 0
                else 0.0,
                "single_site_end_to_end_top1": float(subset["single_site"]["end_to_end_top1"]) / float(subset["single_site"]["total"])
                if subset["single_site"]["total"] > 0
                else 0.0,
                "source_breakdown": source_breakdown,
            }

        def _count_pairwise_stats(self, examples):
            pairwise_example_count = 0
            pairwise_hard_false_count = 0
            for example in examples:
                if not (example["trainable"] and example["target_index"] >= 0):
                    continue
                positive_local = torch.nonzero(example["selected_labels"] > 0.5, as_tuple=False).view(-1)
                negative_local = self._select_pairwise_negative_indices(example, positive_local)
                if int(positive_local.numel()) > 0 and int(negative_local.numel()) > 0:
                    pairwise_example_count += 1
                    pairwise_hard_false_count += int(negative_local.numel())
            return pairwise_example_count, pairwise_hard_false_count

        def _finalize_epoch_metrics(
            self,
            *,
            shortlist_scores,
            site_labels,
            site_batches,
            site_supervision_masks,
            candidate_masks,
            merged_edge_parts,
            merged_coord_parts,
            graph_sources,
            batch_metrics_rows,
            winner_examples,
        ):
            metrics = super()._finalize_epoch_metrics(
                shortlist_scores=shortlist_scores,
                site_labels=site_labels,
                site_batches=site_batches,
                site_supervision_masks=site_supervision_masks,
                candidate_masks=candidate_masks,
                merged_edge_parts=merged_edge_parts,
                merged_coord_parts=merged_coord_parts,
                graph_sources=graph_sources,
                batch_metrics_rows=batch_metrics_rows,
                winner_examples=winner_examples,
            )
            merged_site_scores = torch.cat(shortlist_scores, dim=0)
            merged_site_labels = torch.cat(site_labels, dim=0)
            merged_site_batch = torch.cat(site_batches, dim=0)
            merged_site_supervision_mask = torch.cat(site_supervision_masks, dim=0)
            merged_candidate_mask = torch.cat(candidate_masks, dim=0)
            merged_edge_index = (
                torch.cat(merged_edge_parts, dim=1) if merged_edge_parts else torch.zeros((2, 0), dtype=torch.long)
            )
            merged_atom_coordinates = torch.cat(merged_coord_parts, dim=0) if merged_coord_parts else None
            shortlist_epoch_stats = self._shortlist_hard_negative_epoch_stats(
                merged_site_scores=merged_site_scores,
                merged_site_labels=merged_site_labels,
                merged_site_batch=merged_site_batch,
                merged_site_supervision_mask=merged_site_supervision_mask,
                merged_candidate_mask=merged_candidate_mask,
                merged_edge_index=merged_edge_index,
                merged_atom_coordinates=merged_atom_coordinates,
            )
            pairwise_example_count, pairwise_hard_false_count = self._count_pairwise_stats(winner_examples)
            if self._active_split_name == "train":
                metrics["multisite_train_example_count"] = float(
                    sum(1 for ex in winner_examples if ex["trainable"] and bool(ex.get("is_multisite", False)))
                )
                metrics["single_site_train_example_count"] = float(
                    sum(1 for ex in winner_examples if ex["trainable"] and not bool(ex.get("is_multisite", False)))
                )
            metrics["winner_use_multi_positive_targets"] = bool(self.winner_use_multi_positive_targets)
            metrics["winner_multi_positive_only_for_multisite"] = bool(self.winner_multi_positive_only_for_multisite)
            metrics["winner_enable_pairwise_ranking"] = bool(self.winner_enable_pairwise_ranking)
            metrics["winner_pairwise_loss_weight"] = float(self.winner_pairwise_loss_weight)
            metrics["winner_use_source_embedding"] = bool(self.winner_use_source_embedding)
            metrics["winner_source_embedding_dim"] = int(self.winner_source_embedding_dim)
            metrics["winner_use_source_bias"] = bool(self.winner_use_source_bias)
            metrics["shortlist_hard_negative_requested_flag"] = bool(self.shortlist_hard_negative_requested_flag)
            metrics["shortlist_enable_hard_negative_emphasis"] = bool(self.shortlist_enable_hard_negative_emphasis)
            metrics["shortlist_hard_negative_loss_weight"] = float(self.shortlist_hard_negative_loss_weight)
            metrics["shortlist_hard_negative_mode"] = str(self.shortlist_hard_negative_mode)
            metrics["shortlist_hard_negative_rank_window"] = [
                int(self.shortlist_hard_negative_rank_min),
                int(self.shortlist_hard_negative_rank_max),
            ]
            metrics["shortlist_pairwise_margin"] = float(self.shortlist_pairwise_margin)
            metrics["shortlist_pairwise_loss_weight"] = float(self.shortlist_pairwise_loss_weight)
            metrics["shortlist_hard_negative_max_per_true"] = int(self.shortlist_hard_negative_max_per_true)
            metrics["shortlist_hard_negative_sample_mode"] = str(self.shortlist_hard_negative_sample_mode)
            metrics["winner_pairwise_example_count"] = float(pairwise_example_count)
            metrics["winner_pairwise_hard_false_count"] = float(pairwise_hard_false_count)
            metrics["shortlist_pairwise_example_count"] = float(shortlist_epoch_stats.get("shortlist_pairwise_example_count", 0.0))
            metrics["shortlist_hard_negative_count"] = float(shortlist_epoch_stats.get("shortlist_hard_negative_count", 0.0))
            metrics["shortlist_top_false_beats_true_fraction"] = float(
                shortlist_epoch_stats.get("shortlist_top_false_beats_true_fraction", 0.0)
            )
            metrics["multisite_pairwise_restore_summary"] = dict(self.restore_summary)
            metrics["multisite_pairwise_source_conditioning_enabled"] = bool(
                self.winner_use_source_embedding or self.winner_use_source_bias
            )
            self._maybe_warn_shortlist_aux_inactive(metrics)
            return metrics

        def train_loader_epoch(self, loader):
            self._active_split_name = "train"
            return super().train_loader_epoch(loader)

        def evaluate_loader(self, loader):
            self._active_split_name = "eval"
            return super().evaluate_loader(loader)
else:  # pragma: no cover
    class TwoHeadShortlistWinnerV2RebuildMultisitePairwiseTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
