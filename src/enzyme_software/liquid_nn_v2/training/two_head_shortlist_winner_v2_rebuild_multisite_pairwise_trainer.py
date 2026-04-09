from __future__ import annotations

from collections import OrderedDict, defaultdict

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, require_torch, torch
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
            self.shortlist_enable_hard_negative_emphasis = bool(shortlist_enable_hard_negative_emphasis)
            self.shortlist_hard_negative_rank_min = max(1, int(shortlist_hard_negative_rank_min))
            self.shortlist_hard_negative_rank_max = max(self.shortlist_hard_negative_rank_min, int(shortlist_hard_negative_rank_max))
            self.shortlist_hard_negative_loss_weight = max(0.0, float(shortlist_hard_negative_loss_weight))
            self.shortlist_hard_negative_mode = str(shortlist_hard_negative_mode or "top_false").strip().lower()
            self.source_vocab = tuple(self.SOURCE_VOCAB)
            self.source_to_index = {name: idx for idx, name in enumerate(self.source_vocab)}
            self._active_split_name = "unknown"
            if self.shortlist_enable_hard_negative_emphasis:
                raise NotImplementedError(
                    "shortlist hard-negative emphasis is intentionally unsupported in this branch because "
                    "the shortlist provider remains frozen."
                )
            self.trainable_module_summary = [
                {
                    "name": "winner_head_v2_rebuild_multisite_pairwise",
                    "lr": float(self.learning_rate),
                    "param_count": int(sum(param.numel() for param in self.winner_head.parameters() if param.requires_grad)),
                }
            ]
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
                    ("shortlist_enable_hard_negative_emphasis", bool(self.shortlist_enable_hard_negative_emphasis)),
                    ("shortlist_hard_negative_rank_window", [int(self.shortlist_hard_negative_rank_min), int(self.shortlist_hard_negative_rank_max)]),
                    ("shortlist_hard_negative_loss_weight", float(self.shortlist_hard_negative_loss_weight)),
                    ("shortlist_hard_negative_mode", str(self.shortlist_hard_negative_mode)),
                    ("source_vocab", list(self.source_vocab)),
                ]
            )

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
            total_loss = float(self.winner_v2_loss_weight) * winner_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Non-finite two-head multisite-pairwise total loss detected")
            metrics = {
                "winner_loss": float(winner_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                **shortlist_metrics,
                **winner_loss_stats,
            }
            return total_loss, shortlist_scores, examples, metrics

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
            metrics["shortlist_enable_hard_negative_emphasis"] = bool(self.shortlist_enable_hard_negative_emphasis)
            metrics["winner_pairwise_example_count"] = float(pairwise_example_count)
            metrics["winner_pairwise_hard_false_count"] = float(pairwise_hard_false_count)
            metrics["multisite_pairwise_restore_summary"] = dict(self.restore_summary)
            metrics["multisite_pairwise_source_conditioning_enabled"] = bool(
                self.winner_use_source_embedding or self.winner_use_source_bias
            )
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
