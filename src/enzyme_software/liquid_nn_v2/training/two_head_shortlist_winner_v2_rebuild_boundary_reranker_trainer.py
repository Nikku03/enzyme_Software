from __future__ import annotations

from collections import OrderedDict

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.training.two_head_shortlist_winner_v2_rebuild_trainer import (
    TwoHeadShortlistWinnerV2RebuildTrainer,
)


if TORCH_AVAILABLE:
    class TwoHeadShortlistWinnerV2RebuildBoundaryRerankerTrainer(TwoHeadShortlistWinnerV2RebuildTrainer):
        def __init__(
            self,
            *,
            model,
            winner_head,
            boundary_reranker_head,
            learning_rate: float = 1.0e-4,
            weight_decay: float = 1.0e-4,
            max_grad_norm: float = 5.0,
            frozen_shortlist_topk: int = 12,
            winner_v2_rebuild_loss_weight: float = 1.0,
            shortlist_checkpoint_path: str = "",
            boundary_reranker_output_k: int = 6,
            boundary_reranker_train_on_rescued_only: bool = True,
            boundary_reranker_train_on_hits_only: bool = True,
            boundary_reranker_use_pairwise_mode: bool = False,
            boundary_reranker_use_listwise_mode: bool = True,
            boundary_reranker_loss_weight: float = 1.0,
            boundary_reranker_focus_true_rank_min: int = 7,
            boundary_reranker_focus_true_rank_max: int = 12,
            boundary_reranker_winner_init_checkpoint_path: str = "",
            hard_source_names: str = "attnsom,cyp_dbs_external",
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
            self.boundary_reranker_head = boundary_reranker_head.to(self.device)
            self.boundary_reranker_output_k = max(1, int(boundary_reranker_output_k))
            self.boundary_reranker_train_on_rescued_only = bool(boundary_reranker_train_on_rescued_only)
            self.boundary_reranker_train_on_hits_only = bool(boundary_reranker_train_on_hits_only)
            self.boundary_reranker_use_pairwise_mode = bool(boundary_reranker_use_pairwise_mode)
            self.boundary_reranker_use_listwise_mode = bool(boundary_reranker_use_listwise_mode)
            self.boundary_reranker_loss_weight = max(0.0, float(boundary_reranker_loss_weight))
            self.boundary_reranker_focus_true_rank_min = max(1, int(boundary_reranker_focus_true_rank_min))
            self.boundary_reranker_focus_true_rank_max = max(
                self.boundary_reranker_focus_true_rank_min,
                int(boundary_reranker_focus_true_rank_max),
            )
            self.boundary_reranker_winner_init_checkpoint_path = str(
                boundary_reranker_winner_init_checkpoint_path or ""
            ).strip()
            self.hard_source_name_set = {
                token.strip().lower()
                for token in str(hard_source_names or "").split(",")
                if token.strip()
            }
            self.hard_source_names = tuple(sorted(self.hard_source_name_set))
            if self.boundary_reranker_use_pairwise_mode:
                raise ValueError(
                    "boundary reranker pairwise mode is not implemented in this branch; "
                    "use listwise mode."
                )
            if not self.boundary_reranker_use_listwise_mode:
                raise ValueError("boundary reranker requires listwise mode to be enabled.")

            for param in self.winner_head.parameters():
                param.requires_grad = False
            for param in self.boundary_reranker_head.parameters():
                param.requires_grad = True
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [param for param in self.boundary_reranker_head.parameters() if param.requires_grad],
                        "lr": float(self.learning_rate),
                        "weight_decay": float(self.weight_decay),
                    }
                ]
            )
            self.trainable_module_summary = [
                {
                    "name": "boundary_reranker_head_v2_rebuild",
                    "lr": float(self.learning_rate),
                    "param_count": int(
                        sum(param.numel() for param in self.boundary_reranker_head.parameters() if param.requires_grad)
                    ),
                }
            ]
            self.frozen_module_summary = list(self.frozen_module_summary) + ["winner_head_v2_rebuild"]
            self.restore_summary = OrderedDict(self.restore_summary)
            self.restore_summary["boundary_reranker"] = OrderedDict(
                [
                    ("enabled", True),
                    ("shortlist_k", int(self.frozen_shortlist_topk)),
                    ("output_k", int(self.boundary_reranker_output_k)),
                    ("train_on_rescued_only", bool(self.boundary_reranker_train_on_rescued_only)),
                    ("train_on_hits_only", bool(self.boundary_reranker_train_on_hits_only)),
                    ("use_pairwise_mode", bool(self.boundary_reranker_use_pairwise_mode)),
                    ("use_listwise_mode", bool(self.boundary_reranker_use_listwise_mode)),
                    ("loss_type", "listwise_hard_cross_entropy"),
                    ("focus_true_rank_min", int(self.boundary_reranker_focus_true_rank_min)),
                    ("focus_true_rank_max", int(self.boundary_reranker_focus_true_rank_max)),
                    ("winner_init_checkpoint_path", self.boundary_reranker_winner_init_checkpoint_path),
                    ("hard_source_names", list(self.hard_source_names)),
                ]
            )

        def _boundary_reranker_trainable_examples(self, examples):
            trainable = []
            for example in examples:
                if self.boundary_reranker_train_on_hits_only and not bool(example.get("hit_at_12", False)):
                    continue
                if self.boundary_reranker_train_on_rescued_only and not bool(example.get("rescued_by_12", False)):
                    continue
                if int(example.get("reranker_target_index", -1)) < 0:
                    continue
                trainable.append(example)
            return trainable

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
            final_candidate_counts = []
            shortlist_candidate_counts = []
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
                shortlist_k = min(int(self.frozen_shortlist_topk), candidate_count)
                shortlist_order = full_order[:shortlist_k]
                shortlist_indices = mol_indices[shortlist_order]
                shortlist_selected_scores = mol_scores[shortlist_order]
                shortlist_selected_labels = mol_labels[shortlist_order]
                shortlist_candidate_counts.append(float(shortlist_k))

                reranker_features = self._build_winner_features(
                    atom_features=atom_features,
                    selected_indices=shortlist_indices,
                    selected_scores=shortlist_selected_scores,
                    mol_atom_indices=torch.nonzero(batch_index == mol_idx, as_tuple=False).view(-1),
                    edge_index=edge_index,
                    atom_coordinates=atom_coordinates,
                )

                reranker_logits = self.boundary_reranker_head(reranker_features).view(-1)
                if not bool(torch.isfinite(reranker_logits).all()):
                    raise FloatingPointError("Non-finite boundary reranker logits detected")

                output_k = min(int(self.boundary_reranker_output_k), int(shortlist_indices.numel()))
                reranker_order = torch.argsort(reranker_logits, descending=True)[:output_k]
                # Keep shortlist-relative ordering for the existing winner head after subset selection.
                reranker_subset_positions = torch.sort(reranker_order).values
                selected_indices = shortlist_indices[reranker_subset_positions]
                selected_scores = shortlist_selected_scores[reranker_subset_positions]
                selected_labels = shortlist_selected_labels[reranker_subset_positions]
                hit = bool(selected_labels.any().item())
                final_candidate_counts.append(float(output_k))

                winner_features = self._build_winner_features(
                    atom_features=atom_features,
                    selected_indices=selected_indices,
                    selected_scores=selected_scores,
                    mol_atom_indices=torch.nonzero(batch_index == mol_idx, as_tuple=False).view(-1),
                    edge_index=edge_index,
                    atom_coordinates=atom_coordinates,
                )

                reranker_target_index = -1
                if hit_at_12:
                    positive_local_12 = torch.nonzero(shortlist_selected_labels, as_tuple=False).view(-1)
                    best_true_local_12 = positive_local_12[torch.argmax(shortlist_selected_scores[positive_local_12])]
                    reranker_target_index = int(best_true_local_12.item())

                target_index = -1
                if hit:
                    positive_local = torch.nonzero(selected_labels, as_tuple=False).view(-1)
                    best_true_local = positive_local[torch.argmax(selected_scores[positive_local])]
                    target_index = int(best_true_local.item())

                source = "unknown"
                if mol_idx < len(metadata) and isinstance(metadata[mol_idx], dict):
                    source = (
                        str(metadata[mol_idx].get("source") or metadata[mol_idx].get("data_source") or "").strip().lower()
                        or "unknown"
                    )

                examples.append(
                    {
                        "winner_features": winner_features,
                        "reranker_features": reranker_features,
                        "selected_indices": selected_indices,
                        "selected_labels": selected_labels.to(dtype=torch.float32),
                        "selected_scores": selected_scores,
                        "shortlist_indices": shortlist_indices,
                        "shortlist_selected_labels": shortlist_selected_labels.to(dtype=torch.float32),
                        "shortlist_selected_scores": shortlist_selected_scores,
                        "hit": hit,
                        "hit_at_6": hit_at_6,
                        "hit_at_12": hit_at_12,
                        "rescued_by_12": bool(hit_at_12 and not hit_at_6),
                        "reranker_rescued_to_top6": bool(hit and not hit_at_6),
                        "true_rank": true_rank,
                        "trainable": bool(hit or (not self.winner_v2_train_only_on_hits)),
                        "target_index": target_index,
                        "reranker_target_index": reranker_target_index,
                        "source": source,
                    }
                )

            metrics = {
                "shortlist_candidate_count_mean": (
                    float(sum(final_candidate_counts) / len(final_candidate_counts)) if final_candidate_counts else 0.0
                ),
                "boundary_reranker_shortlist_candidate_count_mean": (
                    float(sum(shortlist_candidate_counts) / len(shortlist_candidate_counts))
                    if shortlist_candidate_counts
                    else 0.0
                ),
                "winner_eval_molecule_count": float(sum(1 for ex in examples if ex["hit"])),
                "winner_trainable_molecule_count": float(sum(1 for ex in examples if ex["trainable"] and ex["target_index"] >= 0)),
                "skipped_empty_shortlist_molecules": float(skipped_empty),
            }
            return examples, metrics

        def _winner_eval_metrics(self, examples):
            metrics = dict(super()._winner_eval_metrics(examples))
            reranker_rescued_examples = [ex for ex in examples if bool(ex.get("reranker_rescued_to_top6", False))]
            reranker_rescued_count = len(reranker_rescued_examples)
            winner_miss_among_reranker_rescued_count = 0
            for example in reranker_rescued_examples:
                logits = self.winner_head(example["winner_features"]).view(-1)
                order = torch.argsort(logits, descending=True)
                labels = example["selected_labels"] > 0.5
                top1_hit = bool(labels[order[0]].item()) if int(order.numel()) > 0 else False
                if not top1_hit:
                    winner_miss_among_reranker_rescued_count += 1
            rescued_by_12_count = int(metrics.get("shortlist_rescued_by_12_count", 0.0))
            metrics["reranker_rescued_to_top6_count"] = float(reranker_rescued_count)
            metrics["reranker_rescued_to_top6_fraction"] = (
                float(reranker_rescued_count) / float(rescued_by_12_count) if rescued_by_12_count > 0 else 0.0
            )
            metrics["winner_miss_among_reranker_rescued_count"] = float(winner_miss_among_reranker_rescued_count)
            return metrics

        def _run_batch(self, batch):
            outputs = self._forward_shortlist_provider(batch)
            atom_features = outputs.get("atom_features")
            shortlist_logits = outputs.get("site_logits")
            if atom_features is None:
                raise RuntimeError("boundary reranker requires model outputs['atom_features']")
            if shortlist_logits is None:
                raise RuntimeError("boundary reranker requires model outputs['site_logits']")
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
            trainable = self._boundary_reranker_trainable_examples(examples)
            if not trainable:
                reranker_loss = next(self.boundary_reranker_head.parameters()).sum() * 0.0
            else:
                losses = []
                for example in trainable:
                    logits = self.boundary_reranker_head(example["reranker_features"]).view(-1)
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("Non-finite boundary reranker logits detected")
                    target = torch.tensor([int(example["reranker_target_index"])], device=logits.device, dtype=torch.long)
                    losses.append(F.cross_entropy(logits.unsqueeze(0), target))
                reranker_loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(reranker_loss)):
                raise FloatingPointError("Non-finite boundary reranker loss detected")
            total_loss = float(self.boundary_reranker_loss_weight) * reranker_loss
            metrics = {
                "winner_loss": 0.0,
                "boundary_reranker_loss": float(reranker_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "boundary_reranker_train_example_count": float(len(trainable)),
                "boundary_reranker_train_rescued_count": float(
                    sum(1 for ex in trainable if bool(ex.get("rescued_by_12", False)))
                ),
                "boundary_reranker_train_true_rank_7_to_12_count": float(
                    sum(
                        1
                        for ex in trainable
                        if self.boundary_reranker_focus_true_rank_min
                        <= int(float(ex.get("true_rank", 0.0)) or 0)
                        <= self.boundary_reranker_focus_true_rank_max
                    )
                ),
                **shortlist_metrics,
            }
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

            hard_source_examples = [
                example
                for example in winner_examples
                if str(example.get("source") or "unknown").strip().lower() in self.hard_source_name_set
            ]

            def _subset_metrics(examples):
                total = len(examples)
                if total <= 0:
                    return {
                        "shortlist_recall_at_6": 0.0,
                        "shortlist_recall_at_12": 0.0,
                        "winner_acc_given_hit": 0.0,
                        "end_to_end_top1": 0.0,
                    }
                hit_at_6_count = int(sum(1 for example in examples if bool(example.get("hit_at_6", False))))
                hit_at_12_count = int(sum(1 for example in examples if bool(example.get("hit_at_12", False))))
                winner_hit_count = 0
                end_to_end_top1 = 0
                winner_eval_count = int(sum(1 for example in examples if bool(example.get("hit", False))))
                for example in examples:
                    with torch.no_grad():
                        logits = self.winner_head(example["winner_features"]).view(-1)
                    order = torch.argsort(logits, descending=True)
                    labels = example["selected_labels"] > 0.5
                    top1_hit = bool(labels[order[0]].item()) if int(order.numel()) > 0 else False
                    end_to_end_top1 += int(top1_hit)
                    if bool(example.get("hit", False)):
                        winner_hit_count += int(top1_hit)
                return {
                    "shortlist_recall_at_6": float(hit_at_6_count) / float(total),
                    "shortlist_recall_at_12": float(hit_at_12_count) / float(total),
                    "winner_acc_given_hit": float(winner_hit_count) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                    "end_to_end_top1": float(end_to_end_top1) / float(total),
                }

            hard_subset_metrics = _subset_metrics(hard_source_examples)
            metrics["boundary_reranker_enabled"] = True
            metrics["boundary_reranker_shortlist_k"] = float(int(self.frozen_shortlist_topk))
            metrics["boundary_reranker_output_k"] = float(int(self.boundary_reranker_output_k))
            metrics["boundary_reranker_train_example_count"] = float(
                sum(float(row.get("boundary_reranker_train_example_count", 0.0)) for row in batch_metrics_rows)
            )
            metrics["boundary_reranker_train_rescued_count"] = float(
                sum(float(row.get("boundary_reranker_train_rescued_count", 0.0)) for row in batch_metrics_rows)
            )
            metrics["boundary_reranker_train_true_rank_7_to_12_count"] = float(
                sum(float(row.get("boundary_reranker_train_true_rank_7_to_12_count", 0.0)) for row in batch_metrics_rows)
            )
            metrics["boundary_reranker_focus_true_rank_min"] = float(int(self.boundary_reranker_focus_true_rank_min))
            metrics["boundary_reranker_focus_true_rank_max"] = float(int(self.boundary_reranker_focus_true_rank_max))
            metrics["boundary_reranker_use_listwise_mode"] = bool(self.boundary_reranker_use_listwise_mode)
            metrics["boundary_reranker_use_pairwise_mode"] = bool(self.boundary_reranker_use_pairwise_mode)
            metrics["boundary_reranker_train_on_rescued_only"] = bool(self.boundary_reranker_train_on_rescued_only)
            metrics["boundary_reranker_train_on_hits_only"] = bool(self.boundary_reranker_train_on_hits_only)
            metrics["boundary_reranker_winner_init_checkpoint_path"] = str(
                self.boundary_reranker_winner_init_checkpoint_path or ""
            )
            metrics["hard_source_shortlist_recall_at_6"] = float(hard_subset_metrics["shortlist_recall_at_6"])
            metrics["hard_source_shortlist_recall_at_12"] = float(hard_subset_metrics["shortlist_recall_at_12"])
            metrics["hard_source_winner_acc_given_hit"] = float(hard_subset_metrics["winner_acc_given_hit"])
            metrics["hard_source_end_to_end_top1"] = float(hard_subset_metrics["end_to_end_top1"])
            metrics["boundary_reranker_source_weighting_enabled"] = False
            metrics["boundary_reranker_soft_multi_positive_enabled"] = False
            return metrics

        def train_loader_epoch(self, loader):
            self.model.eval()
            self.winner_head.eval()
            self.boundary_reranker_head.train()
            shortlist_scores = []
            site_labels = []
            site_supervision_masks = []
            candidate_masks = []
            site_batches = []
            merged_edge_parts = []
            merged_coord_parts = []
            graph_sources = []
            graph_offset = 0
            atom_offset = 0
            batch_metrics_rows = []
            winner_examples = []
            zero_hit_batches = 0
            for raw_batch in loader:
                if raw_batch is None:
                    continue
                batch = self._prepare_batch(raw_batch)
                loss, batch_shortlist_scores, batch_winner_examples, batch_metrics = self._run_batch(batch)
                trainable_count = float(batch_metrics.get("boundary_reranker_train_example_count", 0.0))
                if trainable_count > 0.0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [param for group in self.optimizer.param_groups for param in group["params"]],
                        float(self.max_grad_norm),
                    )
                    self.optimizer.step()
                else:
                    zero_hit_batches += 1
                batch_metrics["winner_zero_hit_batches"] = 1.0 if trainable_count <= 0.0 else 0.0
                batch_metrics_rows.append(batch_metrics)
                winner_examples.extend(batch_winner_examples)
                shortlist_scores.append(batch_shortlist_scores.detach().cpu())
                site_labels.append(batch["site_labels"].detach().cpu())
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
                    merged_edge_parts.append(edge_index.detach().cpu() + atom_offset)
                atom_coordinates = batch.get("atom_coordinates")
                if atom_coordinates is not None:
                    merged_coord_parts.append(atom_coordinates.detach().cpu())
                metadata = list(batch.get("graph_metadata") or [])
                graph_sources.extend(
                    str(entry.get("source") or entry.get("data_source") or "").strip().lower() or "unknown"
                    if isinstance(entry, dict)
                    else "unknown"
                    for entry in metadata
                )
                graph_offset += len(metadata) if metadata else (int(batch["batch"].max().item()) + 1 if batch["batch"].numel() else 0)
                atom_offset += int(batch["site_labels"].shape[0])
            metrics = self._finalize_epoch_metrics(
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
            metrics["winner_zero_hit_batches"] = float(zero_hit_batches)
            return metrics

        def evaluate_loader(self, loader):
            self.model.eval()
            self.winner_head.eval()
            self.boundary_reranker_head.eval()
            return super().evaluate_loader(loader)
else:  # pragma: no cover
    class TwoHeadShortlistWinnerV2RebuildBoundaryRerankerTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
