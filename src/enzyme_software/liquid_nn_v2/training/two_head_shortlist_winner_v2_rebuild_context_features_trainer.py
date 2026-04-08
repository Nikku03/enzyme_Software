from __future__ import annotations

from collections import OrderedDict

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.training.hard_negative_mining import (
    _bfs_shortest_paths,
    _build_local_adjacency,
    _valid_3d_coordinates,
)
from enzyme_software.liquid_nn_v2.training.two_head_shortlist_winner_v2_rebuild_trainer import (
    TwoHeadShortlistWinnerV2RebuildTrainer,
)


if TORCH_AVAILABLE:
    class TwoHeadShortlistWinnerV2RebuildContextFeaturesTrainer(TwoHeadShortlistWinnerV2RebuildTrainer):
        SOURCE_VOCAB = ("unknown", "attnsom", "az120", "cyp_dbs_external", "drugbank", "metxbiodb")

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
            hard_source_names: str = "attnsom,cyp_dbs_external",
            winner_context_use_source_features: bool = True,
            winner_context_use_hard_source_indicator: bool = True,
            winner_context_use_local_competition_features: bool = True,
            winner_context_use_relative_top_candidate_features: bool = True,
            winner_context_use_geometry_proxy_features: bool = True,
            winner_context_use_only_existing_repo_features: bool = True,
            winner_context_init_checkpoint_path: str = "",
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
            self.hard_source_name_set = {
                token.strip().lower()
                for token in str(hard_source_names or "").split(",")
                if token.strip()
            }
            self.source_vocab = tuple(self.SOURCE_VOCAB)
            self.source_to_index = {name: idx for idx, name in enumerate(self.source_vocab)}
            self.winner_context_use_source_features = bool(winner_context_use_source_features)
            self.winner_context_use_hard_source_indicator = bool(winner_context_use_hard_source_indicator)
            self.winner_context_use_local_competition_features = bool(winner_context_use_local_competition_features)
            self.winner_context_use_relative_top_candidate_features = bool(winner_context_use_relative_top_candidate_features)
            self.winner_context_use_geometry_proxy_features = bool(winner_context_use_geometry_proxy_features)
            self.winner_context_use_only_existing_repo_features = bool(winner_context_use_only_existing_repo_features)
            self.winner_context_init_checkpoint_path = str(winner_context_init_checkpoint_path or "").strip()
            self.trainable_module_summary = [
                {
                    "name": "winner_head_v2_rebuild_context_features",
                    "lr": float(self.learning_rate),
                    "param_count": int(sum(param.numel() for param in self.winner_head.parameters() if param.requires_grad)),
                }
            ]
            self.restore_summary = OrderedDict(self.restore_summary)
            self.restore_summary["context_features"] = OrderedDict(
                [
                    ("source_features_enabled", bool(self.winner_context_use_source_features)),
                    ("hard_source_indicator_enabled", bool(self.winner_context_use_hard_source_indicator)),
                    ("local_competition_features_enabled", bool(self.winner_context_use_local_competition_features)),
                    ("relative_top_candidate_features_enabled", bool(self.winner_context_use_relative_top_candidate_features)),
                    ("geometry_proxy_features_enabled", bool(self.winner_context_use_geometry_proxy_features)),
                    ("source_vocab", list(self.source_vocab)),
                    ("winner_context_init_checkpoint_path", self.winner_context_init_checkpoint_path),
                ]
            )

        def _source_index(self, source: str) -> int:
            normalized = str(source or "unknown").strip().lower() or "unknown"
            return int(self.source_to_index.get(normalized, self.source_to_index["unknown"]))

        def _build_winner_features(
            self,
            *,
            atom_features,
            selected_indices,
            selected_scores,
            mol_atom_indices,
            edge_index,
            atom_coordinates,
        ):
            base_features = super()._build_winner_features(
                atom_features=atom_features,
                selected_indices=selected_indices,
                selected_scores=selected_scores,
                mol_atom_indices=mol_atom_indices,
                edge_index=edge_index,
                atom_coordinates=atom_coordinates,
            )
            extra_parts = []
            candidate_count = int(selected_indices.numel())
            if bool(self.winner_context_use_relative_top_candidate_features):
                if candidate_count > 1:
                    top2_score = selected_scores[min(1, candidate_count - 1)]
                    top2_gap = (top2_score - selected_scores).unsqueeze(-1)
                else:
                    top2_gap = torch.zeros((candidate_count, 1), device=selected_scores.device, dtype=selected_scores.dtype)
                extra_parts.append(top2_gap)
            if bool(self.winner_context_use_local_competition_features):
                mean_score = selected_scores.mean() if candidate_count > 0 else selected_scores.new_tensor(0.0)
                score_centered = (selected_scores - mean_score).unsqueeze(-1)
                score_std = selected_scores.std(unbiased=False) if candidate_count > 1 else selected_scores.new_tensor(0.0)
                score_z = (selected_scores - mean_score) / (score_std + selected_scores.new_tensor(1.0e-6))
                score_z = score_z.unsqueeze(-1)
                candidate_count_norm = torch.full(
                    (candidate_count, 1),
                    float(candidate_count) / float(max(1, int(self.frozen_shortlist_topk))),
                    device=selected_scores.device,
                    dtype=selected_scores.dtype,
                )
                adjacency = _build_local_adjacency(mol_atom_indices, edge_index)
                local_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(mol_atom_indices.tolist())}
                mean_graph_closeness = []
                for candidate_idx in selected_indices.tolist():
                    start_local = int(local_lookup[int(candidate_idx)])
                    graph_distances = _bfs_shortest_paths(
                        adjacency,
                        start_local,
                        device=selected_scores.device,
                        dtype=selected_scores.dtype,
                    )
                    selected_local = torch.tensor(
                        [local_lookup[int(idx.item())] for idx in selected_indices],
                        device=selected_scores.device,
                        dtype=torch.long,
                    )
                    competitor_distances = graph_distances[selected_local]
                    competitor_closeness = torch.where(
                        torch.isfinite(competitor_distances),
                        1.0 / (1.0 + competitor_distances),
                        torch.zeros_like(competitor_distances),
                    )
                    mean_graph_closeness.append(float(competitor_closeness.mean().item()))
                mean_graph_closeness = torch.tensor(
                    mean_graph_closeness,
                    device=selected_scores.device,
                    dtype=selected_scores.dtype,
                ).unsqueeze(-1)
                extra_parts.extend([score_centered, score_z, candidate_count_norm, mean_graph_closeness])
            if bool(self.winner_context_use_geometry_proxy_features):
                mol_coords = atom_coordinates[mol_atom_indices] if atom_coordinates is not None else None
                if _valid_3d_coordinates(mol_coords):
                    local_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(mol_atom_indices.tolist())}
                    selected_local = torch.tensor(
                        [local_lookup[int(idx.item())] for idx in selected_indices],
                        device=selected_scores.device,
                        dtype=torch.long,
                    )
                    selected_coords = mol_coords[selected_local]
                    geometry_values = []
                    for row_idx in range(candidate_count):
                        deltas = selected_coords - selected_coords[row_idx : row_idx + 1]
                        euclidean = torch.norm(deltas, dim=-1)
                        closeness = 1.0 / (1.0 + euclidean)
                        geometry_values.append(float(closeness.mean().item()))
                    geometry_proxy = torch.tensor(
                        geometry_values,
                        device=selected_scores.device,
                        dtype=selected_scores.dtype,
                    ).unsqueeze(-1)
                else:
                    geometry_proxy = torch.zeros((candidate_count, 1), device=selected_scores.device, dtype=selected_scores.dtype)
                extra_parts.append(geometry_proxy)
            winner_features = base_features if not extra_parts else torch.cat([base_features, *extra_parts], dim=-1)
            if not bool(torch.isfinite(winner_features).all()):
                raise FloatingPointError("Non-finite winner context features detected")
            return winner_features

        def _build_winner_examples(self, atom_features, shortlist_scores, batch):
            examples, metrics = super()._build_winner_examples(atom_features, shortlist_scores, batch)
            for example in examples:
                source = str(example.get("source") or "unknown").strip().lower() or "unknown"
                source_index = self._source_index(source)
                candidate_count = int(example["selected_indices"].numel())
                example["source_index_tensor"] = torch.full(
                    (candidate_count,),
                    int(source_index),
                    device=example["winner_features"].device,
                    dtype=torch.long,
                )
                example["hard_source_indicator_tensor"] = torch.full(
                    (candidate_count,),
                    1.0 if source in self.hard_source_name_set else 0.0,
                    device=example["winner_features"].device,
                    dtype=example["winner_features"].dtype,
                )
            metrics["winner_context_feature_dim"] = float(getattr(self.winner_head, "feature_dim", 0))
            return examples, metrics

        def _winner_logits(self, example):
            return self.winner_head(
                example["winner_features"],
                source_indices=example["source_index_tensor"],
                hard_source_indicator=example["hard_source_indicator_tensor"],
            ).view(-1)

        def _winner_loss(self, examples):
            trainable = [example for example in examples if bool(example.get("trainable", False)) and int(example.get("target_index", -1)) >= 0]
            if not trainable:
                zero = next(self.winner_head.parameters()).sum() * 0.0
                return zero
            losses = []
            for example in trainable:
                logits = self._winner_logits(example)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Non-finite winner context logits detected")
                target = torch.tensor([int(example["target_index"])], device=logits.device, dtype=torch.long)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite winner context loss detected")
            return loss

        def _winner_eval_metrics(self, examples):
            hit_examples = [example for example in examples if bool(example.get("hit", False))]
            winner_eval_count = len(hit_examples)
            winner_top1 = 0
            winner_top2 = 0
            winner_top3 = 0
            end_to_end_top1 = 0
            end_to_end_top3 = 0
            source_rows = {}
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
                source = str(example.get("source") or "unknown")
                row = source_rows.setdefault(source, {"n": 0, "shortlist_hit": 0.0, "winner_hit": 0.0, "end_to_end_top1": 0.0})
                row["n"] += 1
                row["shortlist_hit"] += float(bool(example.get("hit", False)))
                row["end_to_end_top1"] += float(top1_hit)
                if bool(example.get("hit", False)):
                    winner_top1 += int(top1_hit)
                    winner_top2 += int(bool(labels[order[: min(2, int(order.numel()))]].any().item()))
                    winner_top3 += int(top3_hit)
                    row["winner_hit"] += float(top1_hit)
            source_breakdown = {}
            for name, row in sorted(source_rows.items()):
                n = int(row["n"])
                hit_n = sum(1 for example in examples if str(example.get("source") or "unknown") == name and bool(example.get("hit", False)))
                source_breakdown[name] = {
                    "n": n,
                    "shortlist_recall_at_6": float(row["shortlist_hit"]) / float(n) if n > 0 else 0.0,
                    "winner_acc_given_hit": float(row["winner_hit"]) / float(hit_n) if hit_n > 0 else 0.0,
                    "end_to_end_top1": float(row["end_to_end_top1"]) / float(n) if n > 0 else 0.0,
                }
            hard_source_examples = [
                example for example in examples if str(example.get("source") or "unknown").strip().lower() in self.hard_source_name_set
            ]
            non_hard_source_examples = [
                example for example in examples if str(example.get("source") or "unknown").strip().lower() not in self.hard_source_name_set
            ]

            def _subset_metrics(subset_examples):
                total = len(subset_examples)
                if total <= 0:
                    return {"winner_acc_given_hit": 0.0, "end_to_end_top1": 0.0}
                subset_eval = sum(1 for example in subset_examples if bool(example.get("hit", False)))
                subset_end_to_end = 0
                subset_winner = 0
                for example in subset_examples:
                    logits = self._winner_logits(example)
                    order = torch.argsort(logits, descending=True)
                    labels = example["selected_labels"] > 0.5
                    top1_hit = bool(labels[order[0]].item()) if int(order.numel()) > 0 else False
                    subset_end_to_end += int(top1_hit)
                    if bool(example.get("hit", False)):
                        subset_winner += int(top1_hit)
                return {
                    "winner_acc_given_hit": float(subset_winner) / float(subset_eval) if subset_eval > 0 else 0.0,
                    "end_to_end_top1": float(subset_end_to_end) / float(total),
                }

            hard_subset_metrics = _subset_metrics(hard_source_examples)
            non_hard_subset_metrics = _subset_metrics(non_hard_source_examples)

            return {
                "winner_acc_given_hit": float(winner_top1) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_top2_given_hit": float(winner_top2) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_top3_given_hit": float(winner_top3) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_eval_molecule_count": float(winner_eval_count),
                "end_to_end_top1": float(end_to_end_top1) / float(len(examples)) if examples else 0.0,
                "end_to_end_top3": float(end_to_end_top3) / float(len(examples)) if examples else 0.0,
                "end_to_end_hit_then_win_fraction": float(end_to_end_top1) / float(len(examples)) if examples else 0.0,
                "source_breakdown": source_breakdown,
                "hard_source_end_to_end_top1": float(hard_subset_metrics["end_to_end_top1"]),
                "hard_source_winner_acc_given_hit": float(hard_subset_metrics["winner_acc_given_hit"]),
                "non_hard_source_end_to_end_top1": float(non_hard_subset_metrics["end_to_end_top1"]),
                "non_hard_source_winner_acc_given_hit": float(non_hard_subset_metrics["winner_acc_given_hit"]),
            }

        def _run_batch(self, batch):
            total_loss, shortlist_scores, examples, metrics = super()._run_batch(batch)
            metrics["winner_context_feature_dim"] = float(getattr(self.winner_head, "feature_dim", 0))
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
            metrics["winner_context_feature_dim"] = float(getattr(self.winner_head, "feature_dim", 0))
            metrics["winner_context_source_features_enabled"] = bool(self.winner_context_use_source_features)
            metrics["winner_context_local_competition_features_enabled"] = bool(
                self.winner_context_use_local_competition_features or self.winner_context_use_relative_top_candidate_features
            )
            metrics["winner_context_geometry_proxy_features_enabled"] = bool(self.winner_context_use_geometry_proxy_features)
            metrics["winner_context_source_vocab"] = list(self.source_vocab)
            metrics["winner_context_init_checkpoint_path"] = str(self.winner_context_init_checkpoint_path or "")
            metrics["winner_context_source_weighting_enabled"] = False
            metrics["winner_context_soft_multi_positive_enabled"] = False
            return metrics
else:  # pragma: no cover
    class TwoHeadShortlistWinnerV2RebuildContextFeaturesTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
