from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, torch
from enzyme_software.liquid_nn_v2.training.hard_negative_mining import compute_hard_negative_margin_loss
from enzyme_software.liquid_nn_v2.training.loss import AdaptiveLossV2
from enzyme_software.liquid_nn_v2.training.metrics import compute_site_metrics_v2
from enzyme_software.liquid_nn_v2.training.pairwise_probe import apply_candidate_mask_to_site_logits
from enzyme_software.liquid_nn_v2.training.two_head_shortlist_winner_v2_trainer import (
    TwoHeadShortlistWinnerV2Trainer,
    _graph_sources_from_metadata,
)
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs, move_to_device


if TORCH_AVAILABLE:
    @dataclass
    class MainlineShortlistFirstTrainer:
        model: object
        winner_head: object
        shortlist_loss_weight: float = 1.0
        winner_loss_weight: float = 0.35
        learning_rate: float = 1.0e-4
        weight_decay: float = 1.0e-4
        max_grad_norm: float = 5.0
        shortlist_candidate_topk: int = 12
        local_winner_topk: int = 6
        shortlist_ranking_weight: float = 0.25
        shortlist_rank_window_weight: float = 0.25
        shortlist_hard_negative_weight: float = 0.25
        shortlist_pairwise_margin: float = 0.20
        shortlist_hard_negative_max_per_true: int = 3
        shortlist_use_rank_weighting: bool = True
        device: torch.device | None = None

        def __post_init__(self):
            self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.winner_head.to(self.device)
            self.base_impl = getattr(getattr(self.model, "base_lnn", None), "impl", None)
            self.som_branch = getattr(self.base_impl, "som_branch", None)
            self.site_head = getattr(self.base_impl, "site_head", None)
            self.source_site_heads = getattr(self.base_impl, "source_site_heads", None)
            for param in self.model.parameters():
                param.requires_grad = False
            if self.som_branch is not None:
                for param in self.som_branch.parameters():
                    param.requires_grad = True
            if self.site_head is not None:
                for param in self.site_head.parameters():
                    param.requires_grad = True
            for param in self.winner_head.parameters():
                param.requires_grad = True
            params = []
            if self.som_branch is not None:
                params.extend([param for param in self.som_branch.parameters() if param.requires_grad])
            if self.site_head is not None:
                params.extend([param for param in self.site_head.parameters() if param.requires_grad])
            params.extend([param for param in self.winner_head.parameters() if param.requires_grad])
            self.optimizer = torch.optim.AdamW(
                [{"params": params, "lr": float(self.learning_rate), "weight_decay": float(self.weight_decay)}]
            )
            model_config = getattr(self.model, "config", None)
            self.shortlist_loss_wrapper = AdaptiveLossV2(
                tau_reg_weight=0.0,
                energy_loss_weight=0.0,
                deliberation_loss_weight=0.0,
                site_label_smoothing=float(getattr(model_config, "site_label_smoothing", 0.0)),
                site_ranking_weight=float(self.shortlist_ranking_weight),
                site_hard_negative_fraction=float(getattr(model_config, "site_hard_negative_fraction", 0.5)),
                site_top1_margin_weight=float(getattr(model_config, "site_top1_margin_weight", 0.0)),
                site_top1_margin_value=float(getattr(model_config, "site_top1_margin_value", 0.5)),
                site_cover_weight=float(getattr(model_config, "site_cover_weight", 0.0)),
                site_cover_margin=float(getattr(model_config, "site_cover_margin", 0.20)),
                site_cover_topk=int(getattr(model_config, "site_cover_topk", 5)),
                site_shortlist_weight=float(self.shortlist_rank_window_weight),
                site_shortlist_temperature=float(getattr(model_config, "site_shortlist_temperature", 0.70)),
                site_shortlist_topk=max(1, int(self.shortlist_candidate_topk)),
                site_use_rank_weighted_shortlist=bool(self.shortlist_use_rank_weighting),
                site_hard_negative_weight=float(self.shortlist_hard_negative_weight),
                site_hard_negative_margin=float(self.shortlist_pairwise_margin),
                site_hard_negative_max_per_true=int(self.shortlist_hard_negative_max_per_true),
                site_use_top_score_hard_neg=True,
                site_use_graph_local_hard_neg=True,
                site_use_3d_local_hard_neg=True,
                site_use_rank_weighted_hard_neg=bool(self.shortlist_use_rank_weighting),
            ).to(self.device)
            for param in self.shortlist_loss_wrapper.parameters():
                param.requires_grad = False
            self.shortlist_loss_wrapper.eval()
            self.trainable_module_summary = [
                {
                    "name": "base_lnn.impl.som_branch",
                    "param_count": int(sum(param.numel() for param in (self.som_branch.parameters() if self.som_branch is not None else []) if param.requires_grad)),
                },
                {
                    "name": "base_lnn.impl.site_head",
                    "param_count": int(sum(param.numel() for param in (self.site_head.parameters() if self.site_head is not None else []) if param.requires_grad)),
                },
                {
                    "name": "mainline_small_local_winner",
                    "param_count": int(sum(param.numel() for param in self.winner_head.parameters() if param.requires_grad)),
                },
            ]
            self.frozen_module_summary = [
                "backbone.shared_encoder",
                "physics_branch",
                "manual_priors",
                "cyp_branch",
                "source_site_heads",
                "legacy_rerankers",
                "legacy_routing_branches",
            ]

        def _set_mode(self, *, train: bool) -> None:
            self.model.eval()
            self.winner_head.train(mode=train)
            if self.som_branch is not None:
                self.som_branch.train(mode=train)
            if self.site_head is not None:
                self.site_head.train(mode=train)

        def _prepare_batch(self, batch_or_graphs):
            if isinstance(batch_or_graphs, dict):
                return move_to_device(batch_or_graphs, self.device)
            return move_to_device(collate_molecule_graphs(batch_or_graphs), self.device)

        def _candidate_mask(self, batch: dict[str, Any]):
            return batch.get("candidate_train_mask", batch.get("candidate_mask"))

        def _supervision_mask(self, batch: dict[str, Any]):
            return batch.get("effective_site_supervision_mask", batch.get("site_supervision_mask"))

        def _forward_model(self, batch: dict[str, Any]):
            return self.model(batch)

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
            return TwoHeadShortlistWinnerV2Trainer._build_winner_features(
                self,
                atom_features=atom_features,
                selected_indices=selected_indices,
                selected_scores=selected_scores,
                mol_atom_indices=mol_atom_indices,
                edge_index=edge_index,
                atom_coordinates=atom_coordinates,
            )

        def _rank_window_stats(self, shortlist_scores, batch):
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
            for mol_idx in range(num_molecules):
                mol_valid = (batch_index == mol_idx) & valid
                if not bool(mol_valid.any()):
                    continue
                mol_scores = shortlist_scores[mol_valid]
                mol_labels = site_labels[mol_valid]
                order = torch.argsort(mol_scores, descending=True)
                ordered_labels = mol_labels[order]
                start = min(1, int(ordered_labels.numel()))
                end = min(int(self.shortlist_candidate_topk), int(ordered_labels.numel()))
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
                k = min(int(self.local_winner_topk), candidate_count)
                selected_order = full_order[:k]
                selected_indices = mol_indices[selected_order]
                selected_scores = mol_scores[selected_order]
                selected_labels = mol_labels[selected_indices]
                positive_local = torch.nonzero(selected_labels, as_tuple=False).view(-1)
                target_distribution = None
                target_index = -1
                if int(positive_local.numel()) > 0:
                    if int(positive_local.numel()) > 1:
                        target_distribution = torch.zeros(k, device=selected_scores.device, dtype=selected_scores.dtype)
                        target_distribution[positive_local] = 1.0 / float(int(positive_local.numel()))
                        target_index = int(positive_local[0].item())
                    else:
                        target_index = int(positive_local[0].item())
                winner_features = self._build_winner_features(
                    atom_features=atom_features,
                    selected_indices=selected_indices,
                    selected_scores=selected_scores,
                    mol_atom_indices=torch.nonzero(batch_index == mol_idx, as_tuple=False).view(-1),
                    edge_index=edge_index,
                    atom_coordinates=atom_coordinates,
                )
                meta = metadata[mol_idx] if mol_idx < len(metadata) and isinstance(metadata[mol_idx], dict) else {}
                true_rank = 0.0
                if bool(full_rank_labels.any()):
                    true_rank = float(int(torch.nonzero(full_rank_labels, as_tuple=False).view(-1)[0].item()) + 1)
                examples.append(
                    {
                        "winner_features": winner_features,
                        "selected_indices": selected_indices,
                        "selected_labels": selected_labels.to(dtype=torch.float32),
                        "selected_scores": selected_scores,
                        "target_distribution": target_distribution,
                        "target_index": target_index,
                        "trainable": bool(target_index >= 0),
                        "hit_at_6": bool(full_rank_labels[: min(6, candidate_count)].any().item()),
                        "hit_at_12": bool(full_rank_labels[: min(12, candidate_count)].any().item()),
                        "rescued_by_12": bool(
                            full_rank_labels[: min(12, candidate_count)].any().item()
                            and not full_rank_labels[: min(6, candidate_count)].any().item()
                        ),
                        "true_rank": true_rank,
                        "metadata": meta,
                        "source": str(meta.get("source") or "unknown"),
                        "training_regime": str(meta.get("training_regime") or ""),
                    }
                )
            return examples, {"skipped_empty_shortlist_molecules": float(skipped_empty)}

        def _winner_loss(self, examples):
            trainable = [example for example in examples if example["trainable"]]
            if not trainable:
                zero = next(self.winner_head.parameters()).sum() * 0.0
                return zero, {"winner_loss_ce_component": 0.0}
            losses = []
            for example in trainable:
                logits = self.winner_head(example["winner_features"]).view(-1)
                target_distribution = example.get("target_distribution")
                if target_distribution is not None:
                    target_distribution = target_distribution.to(device=logits.device, dtype=logits.dtype)
                    loss = -(target_distribution * F.log_softmax(logits, dim=0)).sum()
                else:
                    target = torch.tensor([int(example["target_index"])], device=logits.device, dtype=torch.long)
                    loss = F.cross_entropy(logits.unsqueeze(0), target)
                losses.append(loss)
            return torch.stack(losses).mean(), {
                "winner_loss_ce_component": float(torch.stack([loss.detach() for loss in losses]).mean().item())
            }

        def _winner_eval_metrics(self, examples):
            total_examples = len(examples)
            winner_eval_count = 0
            winner_top1 = 0
            winner_top3 = 0
            end_to_end_top1 = 0
            end_to_end_top3 = 0
            source_rows = defaultdict(lambda: {"n": 0, "shortlist_hit_6": 0, "shortlist_hit_12": 0, "winner_hit": 0, "end_to_end_top1": 0})
            for example in examples:
                logits = self.winner_head(example["winner_features"]).view(-1)
                order = torch.argsort(logits, descending=True)
                labels = example["selected_labels"] > 0.5
                top1_hit = bool(int(order.numel()) > 0 and labels[order[0]].item())
                top3_hit = bool(int(order.numel()) > 0 and labels[order[: min(3, int(order.numel()))]].any().item())
                end_to_end_top1 += int(top1_hit)
                end_to_end_top3 += int(top3_hit)
                source = str(example["source"] or "unknown")
                source_rows[source]["n"] += 1
                source_rows[source]["shortlist_hit_6"] += int(bool(example.get("hit_at_6", False)))
                source_rows[source]["shortlist_hit_12"] += int(bool(example.get("hit_at_12", False)))
                source_rows[source]["end_to_end_top1"] += int(top1_hit)
                if bool(example["trainable"]):
                    winner_eval_count += 1
                    winner_top1 += int(top1_hit)
                    winner_top3 += int(top3_hit)
                    source_rows[source]["winner_hit"] += int(top1_hit)
            source_breakdown = {}
            for source_name, row in sorted(source_rows.items()):
                n = int(row["n"])
                trainable_n = sum(1 for example in examples if str(example["source"] or "unknown") == source_name and bool(example["trainable"]))
                source_breakdown[source_name] = {
                    "n": int(n),
                    "shortlist_recall_at_6": float(row["shortlist_hit_6"]) / float(n) if n > 0 else 0.0,
                    "shortlist_recall_at_12": float(row["shortlist_hit_12"]) / float(n) if n > 0 else 0.0,
                    "winner_acc_given_hit": float(row["winner_hit"]) / float(trainable_n) if trainable_n > 0 else 0.0,
                    "end_to_end_top1": float(row["end_to_end_top1"]) / float(n) if n > 0 else 0.0,
                }
            return {
                "winner_acc_given_hit": float(winner_top1) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_top3_given_hit": float(winner_top3) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_eval_molecule_count": float(winner_eval_count),
                "end_to_end_top1": float(end_to_end_top1) / float(total_examples) if total_examples > 0 else 0.0,
                "end_to_end_top3": float(end_to_end_top3) / float(total_examples) if total_examples > 0 else 0.0,
                "source_breakdown": source_breakdown,
            }

        def _run_batch(self, batch):
            outputs = self._forward_model(batch)
            atom_features = outputs.get("atom_features")
            shortlist_logits = outputs.get("site_logits")
            if atom_features is None or shortlist_logits is None:
                raise RuntimeError("Mainline shortlist-first trainer requires model outputs['atom_features'] and ['site_logits']")
            shortlist_logits = shortlist_logits.view(-1)
            config = getattr(self.model, "config", None)
            masked_shortlist_logits = apply_candidate_mask_to_site_logits(
                shortlist_logits,
                self._candidate_mask(batch),
                mask_mode=str(getattr(config, "candidate_mask_mode", "hard") or "hard"),
                logit_bias=float(getattr(config, "candidate_mask_logit_bias", 2.0)),
            )
            shortlist_scores = torch.sigmoid(masked_shortlist_logits)
            site_labels = batch["site_labels"].view(-1)
            supervision_mask = self._supervision_mask(batch)
            if supervision_mask is not None:
                supervision_mask = supervision_mask.view(-1)
            candidate_mask = self._candidate_mask(batch)
            if candidate_mask is not None:
                candidate_mask = candidate_mask.view(-1)
            shortlist_loss, shortlist_stats = self.shortlist_loss_wrapper.site_loss(
                masked_shortlist_logits,
                site_labels,
                batch["batch"],
                supervision_mask=supervision_mask,
                candidate_mask=candidate_mask,
                edge_index=batch.get("edge_index"),
                atom_coordinates=batch.get("atom_coordinates"),
            )
            examples, shortlist_example_stats = self._build_winner_examples(atom_features, shortlist_scores, batch)
            winner_loss, winner_loss_stats = self._winner_loss(examples)
            total_loss = (float(self.shortlist_loss_weight) * shortlist_loss) + (float(self.winner_loss_weight) * winner_loss)
            shortlist_hard_negative_component = (
                float(self.shortlist_rank_window_weight) * float(shortlist_stats.get("shortlist_loss", 0.0))
                + float(self.shortlist_hard_negative_weight) * float(shortlist_stats.get("hard_negative_loss_raw", 0.0))
            )
            shortlist_base_component = max(0.0, float(shortlist_stats.get("site_loss", 0.0)) - shortlist_hard_negative_component)
            metrics = {
                "shortlist_loss": float(shortlist_loss.detach().item()),
                "winner_loss": float(winner_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "shortlist_loss_base_component": float(shortlist_base_component),
                "shortlist_loss_hard_negative_component": float(shortlist_hard_negative_component),
                "shortlist_pairwise_example_count": float(shortlist_stats.get("hard_negative_pair_count", 0.0)),
                "shortlist_hard_negative_count": float(shortlist_stats.get("hard_negative_pair_count", 0.0)),
                "shortlist_top_false_beats_true_fraction": 1.0 - float(shortlist_stats.get("top_score_true_beats_fraction", 0.0)),
                **shortlist_example_stats,
                **winner_loss_stats,
            }
            return total_loss, shortlist_scores, examples, metrics

        def _finalize_epoch_metrics(
            self,
            *,
            shortlist_scores_parts,
            site_labels_parts,
            site_batches,
            site_supervision_masks,
            candidate_masks,
            edge_parts,
            coord_parts,
            graph_sources,
            batch_metrics_rows,
            winner_examples,
        ):
            merged_site_scores = torch.cat(shortlist_scores_parts, dim=0)
            merged_site_labels = torch.cat(site_labels_parts, dim=0)
            merged_site_batch = torch.cat(site_batches, dim=0)
            merged_site_supervision_mask = torch.cat(site_supervision_masks, dim=0)
            merged_candidate_mask = torch.cat(candidate_masks, dim=0)
            merged_edge_index = torch.cat(edge_parts, dim=1) if edge_parts else torch.zeros((2, 0), dtype=torch.long)
            merged_atom_coordinates = torch.cat(coord_parts, dim=0) if coord_parts else None
            shortlist_metrics = compute_site_metrics_v2(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                supervision_mask=merged_site_supervision_mask,
                ranking_mask=merged_candidate_mask,
            )
            hard_neg_diag_loss, hard_neg_diag_stats = compute_hard_negative_margin_loss(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                margin=float(self.shortlist_pairwise_margin),
                supervision_mask=merged_site_supervision_mask,
                candidate_mask=merged_candidate_mask,
                edge_index=merged_edge_index,
                atom_coordinates=merged_atom_coordinates,
                use_top_score=True,
                use_graph_local=True,
                use_3d_local=True,
                max_hard_negs_per_true=int(self.shortlist_hard_negative_max_per_true),
                use_rank_weighting=bool(self.shortlist_use_rank_weighting),
            )
            _ = hard_neg_diag_loss
            metrics = {
                "shortlist_recall_at_6": float(hard_neg_diag_stats.get("recall_at_6", 0.0)),
                "shortlist_recall_at_12": float(hard_neg_diag_stats.get("recall_at_12", 0.0)),
                "shortlist_true_site_rank_mean": float(hard_neg_diag_stats.get("true_site_rank_mean", 0.0)),
                "shortlist_top1_acc": float(shortlist_metrics.get("site_top1_acc_all_molecules", 0.0)),
                "shortlist_top3_acc": float(shortlist_metrics.get("site_top3_acc_all_molecules", 0.0)),
                "shortlist_candidate_positive_coverage_molecules": float(shortlist_metrics.get("site_candidate_positive_coverage_molecules", 0.0)),
                "shortlist_candidate_positive_coverage_atoms": float(shortlist_metrics.get("site_candidate_positive_coverage_atoms", 0.0)),
                "shortlist_top_false_beats_true_fraction": 1.0 - float(hard_neg_diag_stats.get("top_score_true_beats_fraction", 0.0)),
            }
            if batch_metrics_rows:
                keys = sorted({key for row in batch_metrics_rows for key in row.keys()})
                for key in keys:
                    metrics[key] = float(sum(float(row.get(key, 0.0)) for row in batch_metrics_rows) / len(batch_metrics_rows))
            metrics.update(self._rank_window_stats(merged_site_scores, {
                "batch": merged_site_batch,
                "site_labels": merged_site_labels,
                "site_supervision_mask": merged_site_supervision_mask,
                "candidate_mask": merged_candidate_mask,
            }))
            metrics.update(self._winner_eval_metrics(winner_examples))
            return metrics

        def _iterate_loader(self, loader, *, train: bool):
            self._set_mode(train=train)
            shortlist_scores_parts = []
            site_labels_parts = []
            site_supervision_masks = []
            candidate_masks = []
            site_batches = []
            edge_parts = []
            coord_parts = []
            graph_sources = []
            batch_metrics_rows = []
            winner_examples = []
            graph_offset = 0
            atom_offset = 0
            for raw_batch in loader:
                if raw_batch is None:
                    continue
                batch = self._prepare_batch(raw_batch)
                total_loss, batch_shortlist_scores, batch_winner_examples, batch_metrics = self._run_batch(batch)
                if train:
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [param for group in self.optimizer.param_groups for param in group["params"]],
                        float(self.max_grad_norm),
                    )
                    self.optimizer.step()
                batch_metrics_rows.append(batch_metrics)
                winner_examples.extend(batch_winner_examples)
                shortlist_scores_parts.append(batch_shortlist_scores.detach().cpu())
                site_labels_parts.append(batch["site_labels"].detach().cpu())
                site_supervision_masks.append(
                    self._supervision_mask(batch).detach().cpu()
                    if self._supervision_mask(batch) is not None
                    else torch.ones_like(batch["site_labels"]).detach().cpu()
                )
                candidate_masks.append(
                    self._candidate_mask(batch).detach().cpu()
                    if self._candidate_mask(batch) is not None
                    else torch.ones_like(batch["site_labels"]).detach().cpu()
                )
                site_batches.append(batch["batch"].detach().cpu() + graph_offset)
                edge_index = batch.get("edge_index")
                if edge_index is not None:
                    edge_parts.append(edge_index.detach().cpu() + atom_offset)
                atom_coordinates = batch.get("atom_coordinates")
                if atom_coordinates is not None:
                    coord_parts.append(atom_coordinates.detach().cpu())
                metadata = list(batch.get("graph_metadata") or [])
                graph_sources.extend(_graph_sources_from_metadata(metadata))
                graph_offset += len(metadata) if metadata else (int(batch["batch"].max().item()) + 1 if batch["batch"].numel() else 0)
                atom_offset += int(batch["site_labels"].shape[0])
            if not shortlist_scores_parts:
                raise RuntimeError("Mainline shortlist-first trainer received zero valid batches")
            return self._finalize_epoch_metrics(
                shortlist_scores_parts=shortlist_scores_parts,
                site_labels_parts=site_labels_parts,
                site_batches=site_batches,
                site_supervision_masks=site_supervision_masks,
                candidate_masks=candidate_masks,
                edge_parts=edge_parts,
                coord_parts=coord_parts,
                graph_sources=graph_sources,
                batch_metrics_rows=batch_metrics_rows,
                winner_examples=winner_examples,
            )

        def train_loader_epoch(self, loader):
            return self._iterate_loader(loader, train=True)

        @torch.no_grad()
        def evaluate_loader(self, loader):
            return self._iterate_loader(loader, train=False)

        @torch.no_grad()
        def collect_predictions(self, loader) -> list[dict[str, Any]]:
            self._set_mode(train=False)
            rows: list[dict[str, Any]] = []
            for raw_batch in loader:
                if raw_batch is None:
                    continue
                batch = self._prepare_batch(raw_batch)
                outputs = self._forward_model(batch)
                atom_features = outputs.get("atom_features")
                shortlist_logits = outputs.get("site_logits")
                if atom_features is None or shortlist_logits is None:
                    raise RuntimeError("Mainline shortlist-first trainer requires model outputs['atom_features'] and ['site_logits']")
                shortlist_logits = shortlist_logits.view(-1)
                config = getattr(self.model, "config", None)
                masked_shortlist_logits = apply_candidate_mask_to_site_logits(
                    shortlist_logits,
                    self._candidate_mask(batch),
                    mask_mode=str(getattr(config, "candidate_mask_mode", "hard") or "hard"),
                    logit_bias=float(getattr(config, "candidate_mask_logit_bias", 2.0)),
                )
                shortlist_scores = torch.sigmoid(masked_shortlist_logits)
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
                num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
                for mol_idx in range(num_molecules):
                    mol_valid = (batch_index == mol_idx) & valid
                    if not bool(mol_valid.any()):
                        continue
                    mol_indices = torch.nonzero(mol_valid, as_tuple=False).view(-1)
                    mol_scores = shortlist_scores[mol_indices]
                    candidate_count = int(mol_indices.numel())
                    full_order = torch.argsort(mol_scores, descending=True)
                    shortlist_top6 = mol_indices[full_order[: min(6, candidate_count)]]
                    shortlist_top12 = mol_indices[full_order[: min(12, candidate_count)]]
                    winner_order = full_order[: min(int(self.local_winner_topk), candidate_count)]
                    selected_indices = mol_indices[winner_order]
                    selected_scores = mol_scores[winner_order]
                    winner_features = self._build_winner_features(
                        atom_features=atom_features,
                        selected_indices=selected_indices,
                        selected_scores=selected_scores,
                        mol_atom_indices=torch.nonzero(batch_index == mol_idx, as_tuple=False).view(-1),
                        edge_index=batch.get("edge_index"),
                        atom_coordinates=batch.get("atom_coordinates"),
                    )
                    winner_logits = self.winner_head(winner_features).view(-1)
                    winner_probs = torch.softmax(winner_logits, dim=0)
                    order = torch.argsort(winner_logits, descending=True)
                    meta = metadata[mol_idx] if mol_idx < len(metadata) and isinstance(metadata[mol_idx], dict) else {}
                    predicted_local = int(order[0].item()) if int(order.numel()) > 0 else -1
                    predicted_atom_idx = int(selected_indices[predicted_local].item()) if predicted_local >= 0 else None
                    confidence_score = 0.0
                    if predicted_local >= 0:
                        confidence_score = float(
                            (winner_probs[predicted_local] * selected_scores[predicted_local]).detach().item()
                        )
                    rows.append(
                        {
                            "id": str(meta.get("id") or meta.get("name") or mol_idx),
                            "name": str(meta.get("name") or ""),
                            "source": str(meta.get("source") or ""),
                            "training_regime": str(meta.get("training_regime") or ""),
                            "label_regime": str(meta.get("label_regime") or ""),
                            "site_atoms": [int(v) for v in list(meta.get("site_atoms") or [])],
                            "primary_site_atoms": [int(v) for v in list(meta.get("primary_site_atoms") or [])],
                            "secondary_site_atoms": [int(v) for v in list(meta.get("secondary_site_atoms") or [])],
                            "tertiary_site_atoms": [int(v) for v in list(meta.get("tertiary_site_atoms") or [])],
                            "all_labeled_site_atoms": [int(v) for v in list(meta.get("all_labeled_site_atoms") or [])],
                            "predicted_atom_idx": predicted_atom_idx,
                            "winner_top3_atom_indices": [int(selected_indices[idx].item()) for idx in order[: min(3, int(order.numel()))]],
                            "winner_candidate_atom_indices": [int(v.item()) for v in selected_indices],
                            "shortlist_top6_atom_indices": [int(v.item()) for v in shortlist_top6],
                            "shortlist_top12_atom_indices": [int(v.item()) for v in shortlist_top12],
                            "confidence_score": float(confidence_score),
                        }
                    )
            return rows
