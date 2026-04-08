from __future__ import annotations

from collections import OrderedDict

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.training.two_head_shortlist_winner_v2_rebuild_trainer import (
    TwoHeadShortlistWinnerV2RebuildTrainer,
)
from enzyme_software.liquid_nn_v2.training.two_head_shortlist_winner_v2_trainer import _graph_sources_from_metadata


if TORCH_AVAILABLE:
    class TwoHeadShortlistWinnerV2RebuildHardSourceFinetuneTrainer(TwoHeadShortlistWinnerV2RebuildTrainer):
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
            hard_source_finetune_require_hit: bool = True,
            hard_source_finetune_skip_non_hard_sources: bool = True,
            winner_finetune_init_checkpoint_path: str = "",
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
            self.hard_source_names = tuple(sorted(self.hard_source_name_set))
            self.hard_source_finetune_require_hit = bool(hard_source_finetune_require_hit)
            self.hard_source_finetune_skip_non_hard_sources = bool(hard_source_finetune_skip_non_hard_sources)
            self.winner_finetune_init_checkpoint_path = str(winner_finetune_init_checkpoint_path or "").strip()
            self.trainable_module_summary = [
                {
                    "name": "winner_head_v2_rebuild_hard_source_finetune",
                    "lr": float(self.learning_rate),
                    "param_count": int(sum(param.numel() for param in self.winner_head.parameters() if param.requires_grad)),
                }
            ]
            self.restore_summary = OrderedDict(self.restore_summary)
            self.restore_summary["hard_source_finetune"] = OrderedDict(
                [
                    ("enabled", True),
                    ("hard_source_names", list(self.hard_source_names)),
                    ("require_hit", bool(self.hard_source_finetune_require_hit)),
                    ("skip_non_hard_sources", bool(self.hard_source_finetune_skip_non_hard_sources)),
                    ("winner_finetune_init_checkpoint_path", self.winner_finetune_init_checkpoint_path),
                ]
            )

        def _hard_source_trainable_examples(self, examples):
            filtered = []
            for example in examples:
                source = str(example.get("source") or "unknown").strip().lower() or "unknown"
                if self.hard_source_finetune_skip_non_hard_sources and source not in self.hard_source_name_set:
                    continue
                if self.hard_source_finetune_require_hit and not bool(example.get("hit", False)):
                    continue
                if not bool(example.get("trainable", False)):
                    continue
                if int(example.get("target_index", -1)) < 0:
                    continue
                filtered.append(example)
            return filtered

        def _winner_loss(self, examples):
            trainable = self._hard_source_trainable_examples(examples)
            if not trainable:
                zero = next(self.winner_head.parameters()).sum() * 0.0
                return zero
            losses = []
            for example in trainable:
                logits = self.winner_head(example["winner_features"]).view(-1)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Non-finite hard-source fine-tune winner logits detected")
                target = torch.tensor([int(example["target_index"])], device=logits.device, dtype=torch.long)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite hard-source fine-tune winner loss detected")
            return loss

        def _run_batch(self, batch):
            total_loss, shortlist_scores, examples, metrics = super()._run_batch(batch)
            hard_source_trainable = self._hard_source_trainable_examples(examples)
            metrics["hard_source_finetune_train_example_count"] = float(len(hard_source_trainable))
            metrics["hard_source_finetune_train_molecule_count"] = float(len(hard_source_trainable))
            metrics["hard_source_finetune_skipped_non_hard_source_count"] = float(
                sum(
                    1
                    for example in examples
                    if bool(example.get("trainable", False))
                    and int(example.get("target_index", -1)) >= 0
                    and str(example.get("source") or "unknown").strip().lower() not in self.hard_source_name_set
                )
            )
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
            non_hard_source_examples = [
                example
                for example in winner_examples
                if str(example.get("source") or "unknown").strip().lower() not in self.hard_source_name_set
            ]

            def _subset_metrics(examples):
                total = len(examples)
                if total <= 0:
                    return {
                        "shortlist_recall_at_6": 0.0,
                        "winner_acc_given_hit": 0.0,
                        "end_to_end_top1": 0.0,
                    }
                hit_at_6_count = int(sum(1 for example in examples if bool(example.get("hit_at_6", False))))
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
                    "winner_acc_given_hit": float(winner_hit_count) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                    "end_to_end_top1": float(end_to_end_top1) / float(total),
                }

            hard_subset_metrics = _subset_metrics(hard_source_examples)
            non_hard_subset_metrics = _subset_metrics(non_hard_source_examples)

            metrics["hard_source_finetune_train_example_count"] = float(
                sum(float(row.get("hard_source_finetune_train_example_count", 0.0)) for row in batch_metrics_rows)
            )
            metrics["hard_source_finetune_train_molecule_count"] = float(
                sum(float(row.get("hard_source_finetune_train_molecule_count", 0.0)) for row in batch_metrics_rows)
            )
            metrics["hard_source_names_used"] = list(self.hard_source_names)
            metrics["winner_finetune_init_checkpoint_path"] = str(self.winner_finetune_init_checkpoint_path or "")
            metrics["hard_source_end_to_end_top1"] = float(hard_subset_metrics["end_to_end_top1"])
            metrics["hard_source_winner_acc_given_hit"] = float(hard_subset_metrics["winner_acc_given_hit"])
            metrics["hard_source_shortlist_recall_at_6"] = float(hard_subset_metrics["shortlist_recall_at_6"])
            metrics["non_hard_source_end_to_end_top1"] = float(non_hard_subset_metrics["end_to_end_top1"])
            metrics["non_hard_source_winner_acc_given_hit"] = float(non_hard_subset_metrics["winner_acc_given_hit"])
            metrics["hard_source_finetune_source_weighting_enabled"] = False
            metrics["hard_source_finetune_soft_multi_positive_enabled"] = False
            return metrics

        def train_loader_epoch(self, loader):
            self.model.eval()
            self.winner_head.train()
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
                hard_source_trainable_count = float(batch_metrics.get("hard_source_finetune_train_molecule_count", 0.0))
                if hard_source_trainable_count > 0.0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [param for group in self.optimizer.param_groups for param in group["params"]],
                        float(self.max_grad_norm),
                    )
                    self.optimizer.step()
                else:
                    zero_hit_batches += 1
                batch_metrics["winner_zero_hit_batches"] = 1.0 if hard_source_trainable_count <= 0.0 else 0.0
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
                graph_sources.extend(_graph_sources_from_metadata(metadata))
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
else:  # pragma: no cover
    class TwoHeadShortlistWinnerV2RebuildHardSourceFinetuneTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
