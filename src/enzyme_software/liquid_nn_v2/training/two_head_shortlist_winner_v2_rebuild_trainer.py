from __future__ import annotations

from collections import OrderedDict, defaultdict

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.training.hard_negative_mining import mine_hard_negative_pairs
from enzyme_software.liquid_nn_v2.training.metrics import compute_site_metrics_v2
from enzyme_software.liquid_nn_v2.training.two_head_shortlist_winner_v2_trainer import (
    TwoHeadShortlistWinnerV2Trainer,
)


if TORCH_AVAILABLE:
    class TwoHeadShortlistWinnerV2RebuildTrainer(TwoHeadShortlistWinnerV2Trainer):
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
            device=None,
        ):
            super().__init__(
                model=model,
                winner_head=winner_head,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                frozen_shortlist_topk=frozen_shortlist_topk,
                winner_v2_use_existing_candidate_features=True,
                winner_v2_use_score_gap_features=True,
                winner_v2_use_rank_features=True,
                winner_v2_use_pairwise_features=True,
                winner_v2_use_graph_local_features=True,
                winner_v2_use_3d_local_features=True,
                winner_v2_train_only_on_hits=True,
                winner_v2_loss_weight=winner_v2_rebuild_loss_weight,
                shortlist_checkpoint_path=shortlist_checkpoint_path,
                device=device,
            )
            self.trainable_module_summary = [
                {
                    "name": "winner_head_v2_rebuild",
                    "lr": float(self.learning_rate),
                    "param_count": int(sum(param.numel() for param in self.winner_head.parameters() if param.requires_grad)),
                }
            ]
            self.winner_feature_dim = int(getattr(self.winner_head, "feature_dim", 0))
            self.restore_summary = OrderedDict(
                [
                    (
                        "feature_groups",
                        OrderedDict(
                            [
                                ("existing_candidate_features", True),
                                ("score_gap_features", True),
                                ("rank_features", True),
                                ("pairwise_features", True),
                                ("graph_local_features", True),
                                ("3d_local_features", True),
                            ]
                        ),
                    ),
                    ("multi_positive_behavior", "deterministic_best_true_by_shortlist_score"),
                    ("source_weighting_behavior", "disabled"),
                    ("loss_type", "hard_cross_entropy"),
                    ("train_only_on_hits", True),
                ]
            )

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

                target_index = -1
                if hit_at_train_k:
                    positive_local = torch.nonzero(selected_labels, as_tuple=False).view(-1)
                    best_true_local = positive_local[torch.argmax(selected_scores[positive_local])]
                    target_index = int(best_true_local.item())

                source = "unknown"
                if mol_idx < len(metadata) and isinstance(metadata[mol_idx], dict):
                    source = str(metadata[mol_idx].get("source") or metadata[mol_idx].get("data_source") or "").strip().lower() or "unknown"

                examples.append(
                    {
                        "winner_features": winner_features,
                        "selected_indices": selected_indices,
                        "selected_labels": selected_labels.to(dtype=torch.float32),
                        "selected_scores": selected_scores,
                        "hit": hit_at_train_k,
                        "hit_at_6": hit_at_6,
                        "hit_at_12": hit_at_12,
                        "true_rank": true_rank,
                        "trainable": bool(hit_at_train_k or (not self.winner_v2_train_only_on_hits)),
                        "target_index": target_index,
                        "source": source,
                    }
                )

            metrics = {
                "shortlist_candidate_count_mean": float(sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
                "winner_eval_molecule_count": float(sum(1 for ex in examples if ex["hit"])),
                "winner_trainable_molecule_count": float(sum(1 for ex in examples if ex["trainable"] and ex["target_index"] >= 0)),
                "skipped_empty_shortlist_molecules": float(skipped_empty),
            }
            return examples, metrics

        def _run_batch(self, batch):
            total_loss, shortlist_scores, examples, metrics = super()._run_batch(batch)
            metrics["winner_feature_dim"] = float(self.winner_feature_dim)
            return total_loss, shortlist_scores, examples, metrics

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
            if not shortlist_scores:
                raise RuntimeError("two_head_shortlist_winner_v2_rebuild received zero valid batches")

            merged_site_scores = torch.cat(shortlist_scores, dim=0)
            merged_site_labels = torch.cat(site_labels, dim=0)
            merged_site_batch = torch.cat(site_batches, dim=0)
            merged_site_supervision_mask = torch.cat(site_supervision_masks, dim=0)
            merged_candidate_mask = torch.cat(candidate_masks, dim=0)
            merged_edge_index = (
                torch.cat(merged_edge_parts, dim=1) if merged_edge_parts else torch.zeros((2, 0), dtype=torch.long)
            )
            merged_atom_coordinates = torch.cat(merged_coord_parts, dim=0) if merged_coord_parts else None
            shortlist_metrics = compute_site_metrics_v2(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                supervision_mask=merged_site_supervision_mask,
                ranking_mask=merged_candidate_mask,
            )
            hard_neg_stats = mine_hard_negative_pairs(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                supervision_mask=merged_site_supervision_mask,
                candidate_mask=merged_candidate_mask,
                edge_index=merged_edge_index,
                atom_coordinates=merged_atom_coordinates,
                use_top_score=bool(getattr(self.model.config, "site_use_top_score_hard_neg", True)),
                use_graph_local=bool(getattr(self.model.config, "site_use_graph_local_hard_neg", True)),
                use_3d_local=bool(getattr(self.model.config, "site_use_3d_local_hard_neg", True)),
                max_hard_negs_per_true=int(getattr(self.model.config, "site_hard_negative_max_per_true", 3)),
            )["stats"]

            valid_molecule_count = int(len(winner_examples))
            hit_at_6_count = int(sum(1 for ex in winner_examples if bool(ex.get("hit_at_6", False))))
            hit_at_12_count = int(sum(1 for ex in winner_examples if bool(ex.get("hit_at_12", False))))
            hit_at_train_k_count = int(sum(1 for ex in winner_examples if bool(ex.get("hit", False))))
            true_rank_values = [float(ex.get("true_rank", 0.0)) for ex in winner_examples if float(ex.get("true_rank", 0.0)) > 0.0]
            shortlist_candidate_count_mean = (
                float(sum(int(ex["selected_indices"].numel()) for ex in winner_examples)) / float(valid_molecule_count)
                if valid_molecule_count > 0
                else 0.0
            )

            source_counts = defaultdict(int)
            source_hit_at_6 = defaultdict(int)
            for example in winner_examples:
                source = str(example.get("source") or "unknown")
                source_counts[source] += 1
                source_hit_at_6[source] += int(bool(example.get("hit_at_6", False)))
            source_recall = {
                source: float(source_hit_at_6[source]) / float(source_counts[source])
                for source in sorted(source_counts)
                if source_counts[source] > 0
            }

            shortlist_recall_at_6 = float(hit_at_6_count) / float(valid_molecule_count) if valid_molecule_count > 0 else 0.0
            shortlist_recall_at_12 = float(hit_at_12_count) / float(valid_molecule_count) if valid_molecule_count > 0 else 0.0
            shortlist_hit_fraction = float(hit_at_train_k_count) / float(valid_molecule_count) if valid_molecule_count > 0 else 0.0
            shortlist_true_site_rank_mean = float(sum(true_rank_values) / len(true_rank_values)) if true_rank_values else 0.0

            metrics = {
                "shortlist_top1_acc": float(shortlist_metrics.get("site_top1_acc_all_molecules", 0.0)),
                "shortlist_top2_acc": float(shortlist_metrics.get("site_top2_acc_all_molecules", 0.0)),
                "shortlist_top3_acc": float(shortlist_metrics.get("site_top3_acc_all_molecules", 0.0)),
                "shortlist_recall_at_6": shortlist_recall_at_6,
                "shortlist_recall_at_12": shortlist_recall_at_12,
                "shortlist_true_site_rank_mean": shortlist_true_site_rank_mean,
                "shortlist_candidate_count_mean": shortlist_candidate_count_mean,
                "shortlist_top_score_hard_neg_rank_mean": float(hard_neg_stats.get("top_score_hard_neg_rank_mean", 0.0)),
                "shortlist_top_score_margin_mean": float(hard_neg_stats.get("top_score_margin_mean", 0.0)),
                "shortlist_top_score_true_beats_fraction": float(hard_neg_stats.get("top_score_true_beats_fraction", 0.0)),
                "shortlist_source_recall_at_6": source_recall,
                "frozen_shortlist_checkpoint_path": str(self.shortlist_checkpoint_path or ""),
            }
            if batch_metrics_rows:
                keys = sorted({key for row in batch_metrics_rows for key in row.keys()})
                aggregate_excluded = {
                    "shortlist_candidate_count_mean",
                    "winner_eval_molecule_count",
                    "winner_trainable_molecule_count",
                    "shortlist_hit_fraction",
                    "shortlist_recall_at_6",
                    "shortlist_recall_at_12",
                    "shortlist_true_site_rank_mean",
                }
                for key in keys:
                    if key in aggregate_excluded:
                        continue
                    metrics[key] = float(sum(float(row.get(key, 0.0)) for row in batch_metrics_rows) / len(batch_metrics_rows))
            metrics.update(self._winner_eval_metrics(winner_examples))
            metrics["shortlist_hit_fraction"] = shortlist_hit_fraction
            if int(self.frozen_shortlist_topk) == 6:
                absdiff = abs(float(metrics.get("shortlist_hit_fraction", 0.0)) - float(metrics.get("shortlist_recall_at_6", 0.0)))
                metrics["shortlist_hit_fraction_minus_recall6_absdiff"] = float(absdiff)
                metrics["shortlist_metric_consistency_ok"] = bool(absdiff <= 1.0e-12)
                if float(metrics.get("shortlist_hit_fraction", 0.0)) > 0.0 and float(metrics.get("shortlist_recall_at_6", 0.0)) <= 0.0:
                    raise RuntimeError("Inconsistent shortlist metrics: hit_fraction > 0 but recall_at_6 <= 0 for topk=6")
            else:
                metrics["shortlist_hit_fraction_minus_recall6_absdiff"] = float("nan")
                metrics["shortlist_metric_consistency_ok"] = True
            metrics["shortlist_metric_debug"] = {
                "num_valid_molecules": int(valid_molecule_count),
                "num_hit_at_6": int(hit_at_6_count),
                "num_hit_at_12": int(hit_at_12_count),
                "num_hit_at_train_k": int(hit_at_train_k_count),
                "mean_hit_at_6": float(shortlist_recall_at_6),
                "mean_hit_at_train_k": float(shortlist_hit_fraction),
                "metric_definition": "molecule-level any-true-site-in-topk",
            }
            metrics["v2_rebuild_matches_original_feature_dim"] = bool(
                int(self.winner_feature_dim) == int(getattr(self.winner_head, "feature_dim", 0))
            )
            metrics["v2_rebuild_matches_original_target_mode"] = True
            metrics["v2_rebuild_matches_original_loss_mode"] = True
            metrics["v2_rebuild_source_weighting_enabled"] = False
            metrics["v2_rebuild_soft_multi_positive_enabled"] = False
            metrics["v2_rebuild_restore_summary"] = dict(self.restore_summary)
            return metrics
else:  # pragma: no cover
    class TwoHeadShortlistWinnerV2RebuildTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
