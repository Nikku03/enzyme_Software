from __future__ import annotations

from collections import OrderedDict

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.training.two_head_shortlist_winner_v2_rebuild_trainer import (
    TwoHeadShortlistWinnerV2RebuildTrainer,
)


if TORCH_AVAILABLE:
    class TwoHeadShortlistWinnerV2RebuildDualWinnerRoutingTrainer(TwoHeadShortlistWinnerV2RebuildTrainer):
        def __init__(
            self,
            *,
            model,
            global_winner_head,
            specialist_winner_head,
            frozen_shortlist_topk: int = 6,
            shortlist_checkpoint_path: str = "",
            hard_source_names: str = "attnsom,cyp_dbs_external",
            dual_winner_route_by_source: bool = True,
            dual_winner_use_global_for_non_hard: bool = True,
            dual_winner_use_specialist_for_hard: bool = True,
            global_winner_checkpoint_path: str = "",
            hard_source_winner_checkpoint_path: str = "",
            device=None,
        ):
            super().__init__(
                model=model,
                winner_head=global_winner_head,
                learning_rate=0.0,
                weight_decay=0.0,
                max_grad_norm=0.0,
                frozen_shortlist_topk=frozen_shortlist_topk,
                winner_v2_rebuild_loss_weight=1.0,
                shortlist_checkpoint_path=shortlist_checkpoint_path,
                device=device,
            )
            self.global_winner_head = self.winner_head
            self.specialist_winner_head = specialist_winner_head.to(self.device)
            self.specialist_winner_head.eval()
            for param in self.global_winner_head.parameters():
                param.requires_grad = False
            for param in self.specialist_winner_head.parameters():
                param.requires_grad = False
            self.optimizer = None
            self.trainable_module_summary = []
            self.frozen_module_summary = [
                "frozen_shortlist_provider",
                "backbone",
                "base_lnn.impl.site_head",
                "global_winner_head_v2_rebuild",
                "hard_source_specialist_winner_head_v2_rebuild",
                "pairwise_teacher",
                "tournament",
                "winner_branches",
            ]
            self.hard_source_name_set = {
                token.strip().lower()
                for token in str(hard_source_names or "").split(",")
                if token.strip()
            }
            self.hard_source_names = tuple(sorted(self.hard_source_name_set))
            self.dual_winner_route_by_source = bool(dual_winner_route_by_source)
            self.dual_winner_use_global_for_non_hard = bool(dual_winner_use_global_for_non_hard)
            self.dual_winner_use_specialist_for_hard = bool(dual_winner_use_specialist_for_hard)
            self.global_winner_checkpoint_path = str(global_winner_checkpoint_path or "").strip()
            self.hard_source_winner_checkpoint_path = str(hard_source_winner_checkpoint_path or "").strip()
            self.restore_summary = OrderedDict(self.restore_summary)
            self.restore_summary["dual_winner_routing"] = OrderedDict(
                [
                    ("enabled", True),
                    ("hard_source_names", list(self.hard_source_names)),
                    ("route_by_source", bool(self.dual_winner_route_by_source)),
                    ("use_global_for_non_hard", bool(self.dual_winner_use_global_for_non_hard)),
                    ("use_specialist_for_hard", bool(self.dual_winner_use_specialist_for_hard)),
                    ("global_winner_checkpoint_path", self.global_winner_checkpoint_path),
                    ("hard_source_winner_checkpoint_path", self.hard_source_winner_checkpoint_path),
                ]
            )

        def _winner_head_for_example(self, example):
            source = str(example.get("source") or "unknown").strip().lower() or "unknown"
            is_hard_source = source in self.hard_source_name_set
            if bool(self.dual_winner_route_by_source) and is_hard_source and bool(self.dual_winner_use_specialist_for_hard):
                return self.specialist_winner_head, "specialist", True
            if bool(self.dual_winner_use_global_for_non_hard) or not is_hard_source:
                return self.global_winner_head, "global", False
            return self.global_winner_head, "global", is_hard_source

        def _build_winner_examples(self, atom_features, shortlist_scores, batch):
            examples, metrics = super()._build_winner_examples(atom_features, shortlist_scores, batch)
            for example in examples:
                winner_head, route_name, routed_hard = self._winner_head_for_example(example)
                example["routed_winner_name"] = route_name
                example["routed_is_hard_source"] = bool(routed_hard)
                example["routed_winner_feature_dim"] = int(getattr(winner_head, "feature_dim", 0))
            return examples, metrics

        def _winner_loss(self, examples):
            trainable = [example for example in examples if bool(example.get("trainable", False)) and int(example.get("target_index", -1)) >= 0]
            if not trainable:
                zero = next(self.global_winner_head.parameters()).sum() * 0.0
                return zero
            losses = []
            for example in trainable:
                winner_head, _, _ = self._winner_head_for_example(example)
                logits = winner_head(example["winner_features"]).view(-1)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Non-finite dual-winner routing logits detected")
                target = torch.tensor([int(example["target_index"])], device=logits.device, dtype=torch.long)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite dual-winner routing loss detected")
            return loss

        def _winner_eval_metrics(self, examples):
            hit_examples = [example for example in examples if bool(example.get("hit", False))]
            winner_eval_count = len(hit_examples)
            winner_top1 = 0
            winner_top2 = 0
            winner_top3 = 0
            end_to_end_top1 = 0
            end_to_end_top3 = 0
            global_only_end_to_end_top1 = 0
            specialist_only_end_to_end_top1 = 0
            hard_source_examples = []
            non_hard_source_examples = []
            specialist_used_count = 0
            global_used_count = 0
            source_rows = {}

            for example in examples:
                winner_head, route_name, is_hard_source = self._winner_head_for_example(example)
                with torch.no_grad():
                    logits = winner_head(example["winner_features"]).view(-1)
                    global_logits = self.global_winner_head(example["winner_features"]).view(-1)
                    specialist_logits = self.specialist_winner_head(example["winner_features"]).view(-1)
                order = torch.argsort(logits, descending=True)
                global_order = torch.argsort(global_logits, descending=True)
                specialist_order = torch.argsort(specialist_logits, descending=True)
                labels = example["selected_labels"] > 0.5
                top1_hit = bool(labels[order[0]].item()) if int(order.numel()) > 0 else False
                top3_hit = bool(labels[order[: min(3, int(order.numel()))]].any().item()) if int(order.numel()) > 0 else False
                global_top1_hit = (
                    bool(labels[global_order[0]].item()) if int(global_order.numel()) > 0 else False
                )
                specialist_top1_hit = (
                    bool(labels[specialist_order[0]].item()) if int(specialist_order.numel()) > 0 else False
                )
                if top1_hit:
                    end_to_end_top1 += 1
                if top3_hit:
                    end_to_end_top3 += 1
                if global_top1_hit:
                    global_only_end_to_end_top1 += 1
                if specialist_top1_hit:
                    specialist_only_end_to_end_top1 += 1
                if route_name == "specialist":
                    specialist_used_count += 1
                else:
                    global_used_count += 1
                if is_hard_source:
                    hard_source_examples.append((example, top1_hit))
                else:
                    non_hard_source_examples.append((example, top1_hit))

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

            def _subset_metrics(examples_with_hits):
                total = len(examples_with_hits)
                if total <= 0:
                    return {"winner_acc_given_hit": 0.0, "end_to_end_top1": 0.0}
                winner_eval = sum(1 for example, _ in examples_with_hits if bool(example.get("hit", False)))
                end_to_end_hits = sum(int(top1_hit) for _example, top1_hit in examples_with_hits)
                winner_hits = sum(int(top1_hit) for example, top1_hit in examples_with_hits if bool(example.get("hit", False)))
                return {
                    "winner_acc_given_hit": float(winner_hits) / float(winner_eval) if winner_eval > 0 else 0.0,
                    "end_to_end_top1": float(end_to_end_hits) / float(total),
                }

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
                "dual_winner_hard_source_eval_count": float(len(hard_source_examples)),
                "dual_winner_non_hard_source_eval_count": float(len(non_hard_source_examples)),
                "dual_winner_specialist_used_count": float(specialist_used_count),
                "dual_winner_global_used_count": float(global_used_count),
                "dual_winner_vs_global_only_end_to_end_top1_delta": (
                    (float(end_to_end_top1) - float(global_only_end_to_end_top1)) / float(len(examples))
                    if examples
                    else 0.0
                ),
                "dual_winner_vs_specialist_only_end_to_end_top1_delta": (
                    (float(end_to_end_top1) - float(specialist_only_end_to_end_top1)) / float(len(examples))
                    if examples
                    else 0.0
                ),
            }

        def _run_batch(self, batch):
            total_loss, shortlist_scores, examples, metrics = super()._run_batch(batch)
            metrics["winner_feature_dim"] = float(self.winner_feature_dim)
            metrics["dual_winner_route_by_source_enabled"] = bool(self.dual_winner_route_by_source)
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
            hard_hit6 = int(sum(1 for example in hard_source_examples if bool(example.get("hit_at_6", False))))
            metrics["hard_source_shortlist_recall_at_6"] = (
                float(hard_hit6) / float(len(hard_source_examples)) if hard_source_examples else 0.0
            )
            metrics["hard_source_names_used"] = list(self.hard_source_names)
            metrics["dual_winner_route_by_source_enabled"] = bool(self.dual_winner_route_by_source)
            metrics["dual_winner_use_global_for_non_hard"] = bool(self.dual_winner_use_global_for_non_hard)
            metrics["dual_winner_use_specialist_for_hard"] = bool(self.dual_winner_use_specialist_for_hard)
            metrics["dual_winner_source_weighting_enabled"] = False
            metrics["dual_winner_soft_multi_positive_enabled"] = False
            return metrics

        def train_loader_epoch(self, loader):
            raise RuntimeError("two_head_shortlist_winner_v2_rebuild_dual_winner_routing is eval-only by default")
else:  # pragma: no cover
    class TwoHeadShortlistWinnerV2RebuildDualWinnerRoutingTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
