from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, F, require_torch, torch
from enzyme_software.liquid_nn_v2.training.hard_negative_mining import (
    _bfs_shortest_paths,
    _build_local_adjacency,
    _valid_3d_coordinates,
    mine_hard_negative_pairs,
)
from enzyme_software.liquid_nn_v2.training.metrics import compute_site_metrics_v2, compute_sourcewise_recall_at_k
from enzyme_software.liquid_nn_v2.training.pairwise_probe import apply_candidate_mask_to_site_logits
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs, move_to_device


if TORCH_AVAILABLE:
    def _graph_sources_from_metadata(metadata: List[dict]) -> List[str]:
        sources = []
        for entry in metadata:
            source = ""
            if isinstance(entry, dict):
                source = str(entry.get("source") or entry.get("data_source") or "").strip().lower()
            sources.append(source or "unknown")
        return sources


    @dataclass
    class TwoHeadShortlistWinnerV2Trainer:
        model: object
        winner_head: object
        learning_rate: float = 1.0e-4
        weight_decay: float = 1.0e-4
        max_grad_norm: float = 5.0
        frozen_shortlist_topk: int = 6
        winner_v2_use_existing_candidate_features: bool = True
        winner_v2_use_score_gap_features: bool = True
        winner_v2_use_rank_features: bool = True
        winner_v2_use_pairwise_features: bool = True
        winner_v2_use_graph_local_features: bool = True
        winner_v2_use_3d_local_features: bool = True
        winner_v2_train_only_on_hits: bool = True
        winner_v2_loss_weight: float = 1.0
        shortlist_checkpoint_path: str = ""
        device: Optional[torch.device] = None

        def __post_init__(self):
            self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.winner_head.to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.winner_head.parameters():
                param.requires_grad = True
            self.optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [param for param in self.winner_head.parameters() if param.requires_grad],
                        "lr": float(self.learning_rate),
                        "weight_decay": float(self.weight_decay),
                    }
                ]
            )
            self.trainable_module_summary = [
                {
                    "name": "winner_head_v2",
                    "lr": float(self.learning_rate),
                    "param_count": int(sum(param.numel() for param in self.winner_head.parameters() if param.requires_grad)),
                }
            ]
            self.frozen_module_summary = [
                "frozen_shortlist_provider",
                "backbone",
                "base_lnn.impl.site_head",
                "pairwise_teacher",
                "tournament",
                "winner_branches",
            ]

        def _prepare_batch(self, batch_or_graphs):
            if isinstance(batch_or_graphs, dict):
                return move_to_device(batch_or_graphs, self.device)
            return move_to_device(collate_molecule_graphs(batch_or_graphs), self.device)

        def _candidate_mask(self, batch: Dict[str, object]):
            return batch.get("candidate_train_mask", batch.get("candidate_mask"))

        def _supervision_mask(self, batch: Dict[str, object]):
            return batch.get("effective_site_supervision_mask", batch.get("site_supervision_mask"))

        def _forward_shortlist_provider(self, batch: Dict[str, object]):
            self.model.eval()
            with torch.no_grad():
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
            feature_parts = [atom_features[selected_indices]]
            top_embedding = atom_features[selected_indices[0]].unsqueeze(0).expand(int(selected_indices.numel()), -1)
            top_score = selected_scores[0]
            if bool(self.winner_v2_use_existing_candidate_features):
                feature_parts.append(selected_scores.unsqueeze(-1))
            if bool(self.winner_v2_use_score_gap_features):
                feature_parts.append((top_score - selected_scores).unsqueeze(-1))
            if bool(self.winner_v2_use_rank_features):
                if int(selected_indices.numel()) > 1:
                    rank_feature = torch.arange(
                        int(selected_indices.numel()),
                        device=selected_scores.device,
                        dtype=selected_scores.dtype,
                    ) / float(int(selected_indices.numel()) - 1)
                else:
                    rank_feature = torch.zeros(1, device=selected_scores.device, dtype=selected_scores.dtype)
                feature_parts.append(rank_feature.unsqueeze(-1))
            if bool(self.winner_v2_use_pairwise_features):
                candidate_embeddings = atom_features[selected_indices]
                feature_parts.extend(
                    [
                        top_embedding,
                        candidate_embeddings - top_embedding,
                        candidate_embeddings * top_embedding,
                    ]
                )
            if bool(self.winner_v2_use_graph_local_features):
                adjacency = _build_local_adjacency(mol_atom_indices, edge_index)
                local_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(mol_atom_indices.tolist())}
                start_local = int(local_lookup[int(selected_indices[0].item())])
                graph_distances = _bfs_shortest_paths(
                    adjacency,
                    start_local,
                    device=selected_scores.device,
                    dtype=selected_scores.dtype,
                )
                candidate_graph_distances = graph_distances[
                    torch.tensor(
                        [local_lookup[int(idx.item())] for idx in selected_indices],
                        device=selected_scores.device,
                        dtype=torch.long,
                    )
                ]
                graph_feature = torch.where(
                    torch.isfinite(candidate_graph_distances),
                    1.0 / (1.0 + candidate_graph_distances),
                    torch.zeros_like(candidate_graph_distances),
                )
                feature_parts.append(graph_feature.unsqueeze(-1))
            if bool(self.winner_v2_use_3d_local_features):
                mol_coords = atom_coordinates[mol_atom_indices] if atom_coordinates is not None else None
                if _valid_3d_coordinates(mol_coords):
                    local_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(mol_atom_indices.tolist())}
                    top_coord = mol_coords[local_lookup[int(selected_indices[0].item())] : local_lookup[int(selected_indices[0].item())] + 1]
                    selected_coords = mol_coords[
                        torch.tensor(
                            [local_lookup[int(idx.item())] for idx in selected_indices],
                            device=selected_scores.device,
                            dtype=torch.long,
                        )
                    ]
                    euclidean = torch.norm(selected_coords - top_coord, dim=-1)
                    spatial_feature = 1.0 / (1.0 + euclidean)
                else:
                    spatial_feature = torch.zeros_like(selected_scores)
                feature_parts.append(spatial_feature.unsqueeze(-1))
            winner_features = torch.cat(feature_parts, dim=-1)
            if not bool(torch.isfinite(winner_features).all()):
                raise FloatingPointError("Non-finite winner v2 features detected")
            return winner_features

        def _build_winner_examples(self, atom_features, shortlist_scores, batch: Dict[str, object]):
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
            shortlist_hits = 0
            shortlist_total = 0
            candidate_counts: List[float] = []
            skipped_empty = 0
            for mol_idx in range(num_molecules):
                mol_valid = (batch_index == mol_idx) & valid
                if not bool(mol_valid.any()):
                    skipped_empty += 1
                    continue
                mol_indices = torch.nonzero(mol_valid, as_tuple=False).view(-1)
                mol_scores = shortlist_scores[mol_indices]
                candidate_count = int(mol_indices.numel())
                k = min(int(self.frozen_shortlist_topk), candidate_count)
                if k <= 0:
                    skipped_empty += 1
                    continue
                order = torch.argsort(mol_scores, descending=True)[:k]
                selected_indices = mol_indices[order]
                selected_scores = mol_scores[order]
                selected_labels = site_labels[selected_indices]
                shortlist_total += 1
                hit = bool(selected_labels.any().item())
                shortlist_hits += int(hit)
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
                if hit:
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
                        "hit": hit,
                        "trainable": bool(hit or (not self.winner_v2_train_only_on_hits)),
                        "target_index": target_index,
                        "source": source,
                    }
                )
            metrics = {
                "shortlist_hit_fraction": float(shortlist_hits) / float(shortlist_total) if shortlist_total > 0 else 0.0,
                "shortlist_candidate_count_mean": float(sum(candidate_counts) / len(candidate_counts)) if candidate_counts else 0.0,
                "winner_eval_molecule_count": float(sum(1 for ex in examples if ex["hit"])),
                "winner_trainable_molecule_count": float(sum(1 for ex in examples if ex["trainable"] and ex["target_index"] >= 0)),
                "skipped_empty_shortlist_molecules": float(skipped_empty),
            }
            return examples, metrics

        def _winner_loss(self, examples):
            trainable = [ex for ex in examples if ex["trainable"] and ex["target_index"] >= 0]
            if not trainable:
                zero = next(self.winner_head.parameters()).sum() * 0.0
                return zero
            losses = []
            for example in trainable:
                logits = self.winner_head(example["winner_features"]).view(-1)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("Non-finite winner v2 logits detected")
                target = torch.tensor([int(example["target_index"])], device=logits.device, dtype=torch.long)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))
            loss = torch.stack(losses).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Non-finite winner v2 loss detected")
            return loss

        def _winner_eval_metrics(self, examples):
            hit_examples = [ex for ex in examples if ex["hit"]]
            winner_eval_count = len(hit_examples)
            winner_top1 = 0
            winner_top2 = 0
            winner_top3 = 0
            end_to_end_top1 = 0
            end_to_end_top3 = 0
            source_rows = defaultdict(lambda: {"n": 0, "shortlist_hit": 0.0, "winner_hit": 0.0, "end_to_end_top1": 0.0})
            total_examples = len(examples)
            for example in examples:
                logits = self.winner_head(example["winner_features"]).view(-1)
                order = torch.argsort(logits, descending=True)
                labels = example["selected_labels"] > 0.5
                top1_hit = bool(labels[order[0]].item()) if int(order.numel()) > 0 else False
                top3_hit = bool(labels[order[: min(3, int(order.numel()))]].any().item()) if int(order.numel()) > 0 else False
                if top1_hit:
                    end_to_end_top1 += 1
                if top3_hit:
                    end_to_end_top3 += 1
                source_row = source_rows[str(example["source"])]
                source_row["n"] += 1
                source_row["shortlist_hit"] += float(example["hit"])
                source_row["end_to_end_top1"] += float(top1_hit)
                if example["hit"]:
                    winner_top1 += int(top1_hit)
                    winner_top2 += int(bool(labels[order[: min(2, int(order.numel()))]].any().item()))
                    winner_top3 += int(top3_hit)
                    source_row["winner_hit"] += float(top1_hit)
            source_breakdown = {}
            for name, row in sorted(source_rows.items()):
                n = int(row["n"])
                hit_n = sum(1 for ex in examples if str(ex["source"]) == name and ex["hit"])
                source_breakdown[name] = {
                    "n": n,
                    "shortlist_recall_at_6": float(row["shortlist_hit"]) / float(n) if n > 0 else 0.0,
                    "winner_acc_given_hit": float(row["winner_hit"]) / float(hit_n) if hit_n > 0 else 0.0,
                    "end_to_end_top1": float(row["end_to_end_top1"]) / float(n) if n > 0 else 0.0,
                }
            return {
                "winner_acc_given_hit": float(winner_top1) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_top2_given_hit": float(winner_top2) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_top3_given_hit": float(winner_top3) / float(winner_eval_count) if winner_eval_count > 0 else 0.0,
                "winner_eval_molecule_count": float(winner_eval_count),
                "end_to_end_top1": float(end_to_end_top1) / float(total_examples) if total_examples > 0 else 0.0,
                "end_to_end_top3": float(end_to_end_top3) / float(total_examples) if total_examples > 0 else 0.0,
                "end_to_end_hit_then_win_fraction": float(end_to_end_top1) / float(total_examples) if total_examples > 0 else 0.0,
                "source_breakdown": source_breakdown,
            }

        def _run_batch(self, batch: Dict[str, object]):
            outputs = self._forward_shortlist_provider(batch)
            atom_features = outputs.get("atom_features")
            shortlist_logits = outputs.get("site_logits")
            if atom_features is None:
                raise RuntimeError("two_head_shortlist_winner_v2 requires model outputs['atom_features']")
            if shortlist_logits is None:
                raise RuntimeError("two_head_shortlist_winner_v2 requires model outputs['site_logits']")
            shortlist_logits = shortlist_logits.view(-1)
            config = getattr(self.model, "config", None)
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
            winner_loss = self._winner_loss(examples)
            total_loss = float(self.winner_v2_loss_weight) * winner_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError("Non-finite two-head v2 total loss detected")
            metrics = {
                "winner_loss": float(winner_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
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
            if not shortlist_scores:
                raise RuntimeError("two_head_shortlist_winner_v2 received zero valid batches")
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
            source_recall = compute_sourcewise_recall_at_k(
                merged_site_scores,
                merged_site_labels,
                merged_site_batch,
                graph_sources,
                k=6,
                supervision_mask=merged_site_supervision_mask,
                ranking_mask=merged_candidate_mask,
            )
            metrics = {
                "shortlist_top1_acc": float(shortlist_metrics.get("site_top1_acc_all_molecules", 0.0)),
                "shortlist_top2_acc": float(shortlist_metrics.get("site_top2_acc_all_molecules", 0.0)),
                "shortlist_top3_acc": float(shortlist_metrics.get("site_top3_acc_all_molecules", 0.0)),
                "shortlist_recall_at_6": float(shortlist_metrics.get("recall_at_6", 0.0)),
                "shortlist_recall_at_12": float(shortlist_metrics.get("recall_at_12", 0.0)),
                "shortlist_true_site_rank_mean": float(shortlist_metrics.get("true_site_rank_mean", 0.0)),
                "shortlist_top_score_hard_neg_rank_mean": float(hard_neg_stats.get("top_score_hard_neg_rank_mean", 0.0)),
                "shortlist_top_score_margin_mean": float(hard_neg_stats.get("top_score_margin_mean", 0.0)),
                "shortlist_top_score_true_beats_fraction": float(hard_neg_stats.get("top_score_true_beats_fraction", 0.0)),
                "shortlist_source_recall_at_6": source_recall,
                "frozen_shortlist_checkpoint_path": str(self.shortlist_checkpoint_path or ""),
            }
            if batch_metrics_rows:
                keys = sorted({key for row in batch_metrics_rows for key in row.keys()})
                for key in keys:
                    metrics[key] = float(sum(float(row.get(key, 0.0)) for row in batch_metrics_rows) / len(batch_metrics_rows))
            metrics.update(self._winner_eval_metrics(winner_examples))
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
                if float(batch_metrics.get("winner_trainable_molecule_count", 0.0)) > 0.0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [param for group in self.optimizer.param_groups for param in group["params"]],
                        float(self.max_grad_norm),
                    )
                    self.optimizer.step()
                else:
                    zero_hit_batches += 1
                batch_metrics["winner_zero_hit_batches"] = 1.0 if float(batch_metrics.get("winner_trainable_molecule_count", 0.0)) <= 0.0 else 0.0
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

        def evaluate_loader(self, loader):
            self.model.eval()
            self.winner_head.eval()
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
            with torch.no_grad():
                for raw_batch in loader:
                    if raw_batch is None:
                        continue
                    batch = self._prepare_batch(raw_batch)
                    _loss, batch_shortlist_scores, batch_winner_examples, batch_metrics = self._run_batch(batch)
                    batch_metrics["winner_zero_hit_batches"] = 1.0 if float(batch_metrics.get("winner_trainable_molecule_count", 0.0)) <= 0.0 else 0.0
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
            return self._finalize_epoch_metrics(
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
else:  # pragma: no cover
    @dataclass
    class TwoHeadShortlistWinnerV2Trainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
