from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:
    @dataclass
    class PairwiseSiteTournamentTrainer:
        model: object
        learning_rate: float = 1.0e-3
        weight_decay: float = 1.0e-4
        pair_loss_weight: float = 1.0
        site_loss_weight: float = 0.25
        antisymmetry_weight: float = 0.10
        compare_top_n: int = 0
        shortlist_k: int = 6
        device: Optional[torch.device] = None

        def __post_init__(self):
            self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )

        def _move(self, batch: Dict[str, object]) -> Dict[str, object]:
            moved = {}
            for key, value in batch.items():
                moved[key] = value.to(self.device) if hasattr(value, "to") else value
            return moved

        def _loss(self, outputs, target_mask, candidate_mask, proposal_scores):
            pair_logits = outputs["pair_logits"]
            comparison_mask = outputs["comparison_mask"]
            final_scores = outputs["final_scores"]
            target = target_mask.float()
            valid = candidate_mask > 0.5

            left = target.unsqueeze(2)
            right = target.unsqueeze(1)
            pair_supervision = comparison_mask & (left != right)
            pair_labels = (left > right).float()
            if bool(pair_supervision.any()):
                pair_loss = F.binary_cross_entropy_with_logits(pair_logits[pair_supervision], pair_labels[pair_supervision])
                pair_pred = (torch.sigmoid(pair_logits[pair_supervision]) >= 0.5).float()
                pair_accuracy = float((pair_pred == pair_labels[pair_supervision]).float().mean().item())
            else:
                pair_loss = final_scores.sum() * 0.0
                pair_accuracy = 0.0

            site_target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
            site_ce = -(site_target * F.log_softmax(final_scores, dim=1)).sum(dim=1).mean()

            antisym = pair_logits + pair_logits.transpose(1, 2)
            antisym_mask = comparison_mask & comparison_mask.transpose(1, 2)
            antisymmetry = antisym[antisym_mask].pow(2).mean() if bool(antisym_mask.any()) else final_scores.sum() * 0.0

            total = (
                float(self.pair_loss_weight) * pair_loss
                + float(self.site_loss_weight) * site_ce
                + float(self.antisymmetry_weight) * antisymmetry
            )

            valid_final = final_scores.masked_fill(~valid, float("-inf"))
            top2_vals = torch.topk(valid_final, k=min(2, int(valid_final.shape[1])), dim=1).values
            top1_gap = (
                (top2_vals[:, 0] - top2_vals[:, 1]).mean()
                if int(top2_vals.shape[1]) >= 2
                else top2_vals[:, 0].mean()
            )
            prop_valid = proposal_scores.masked_fill(~valid, float("-inf"))
            prop_top2_vals = torch.topk(prop_valid, k=min(2, int(prop_valid.shape[1])), dim=1).values
            proposal_gap = (
                (prop_top2_vals[:, 0] - prop_top2_vals[:, 1]).mean()
                if int(prop_top2_vals.shape[1]) >= 2
                else prop_top2_vals[:, 0].mean()
            )
            tournament_margin = outputs["tournament_margin"]
            rival_count = outputs["comparison_mask"].sum(dim=-1)
            return total, {
                "pairwise_bce": float(pair_loss.detach().item()),
                "pairwise_accuracy": float(pair_accuracy),
                "site_ce": float(site_ce.detach().item()),
                "antisymmetry": float(antisymmetry.detach().item()),
                "tournament_total_loss": float(total.detach().item()),
                "tournament_margin_mean": float(tournament_margin[valid].mean().detach().item()) if bool(valid.any()) else 0.0,
                "tournament_margin_abs_mean": float(tournament_margin[valid].abs().mean().detach().item()) if bool(valid.any()) else 0.0,
                "local_rival_count_mean": float(rival_count[valid].float().mean().detach().item()) if bool(valid.any()) else 0.0,
                "tournament_top1_gap_mean": float(top1_gap.detach().item()),
                "proposal_top1_gap_mean": float(proposal_gap.detach().item()),
            }

        def train_epoch(self, loader) -> Dict[str, float]:
            self.model.train()
            history = []
            for raw_batch in loader:
                batch = self._move(raw_batch)
                outputs = self.model(
                    batch["candidate_features"],
                    batch["candidate_mask"],
                    batch["proposal_scores"],
                    compare_top_n=int(self.compare_top_n),
                    candidate_local_rival_mask=batch.get("candidate_local_rival_mask"),
                    candidate_graph_distance=batch.get("candidate_graph_distance"),
                    candidate_3d_distance=batch.get("candidate_3d_distance"),
                    candidate_same_ring_system=batch.get("candidate_same_ring_system"),
                    candidate_same_topology_role=batch.get("candidate_same_topology_role"),
                    candidate_same_chem_family=batch.get("candidate_same_chem_family"),
                    candidate_branch_bulk=batch.get("candidate_branch_bulk"),
                    candidate_exposed_span=batch.get("candidate_exposed_span"),
                    candidate_anti_score=batch.get("candidate_anti_score"),
                )
                loss, stats = self._loss(
                    outputs,
                    batch["target_mask"],
                    batch["candidate_mask"],
                    batch["proposal_scores"],
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                history.append(stats)
            if not history:
                return {"tournament_total_loss": 0.0}
            keys = sorted({key for row in history for key in row.keys()})
            return {key: float(sum(float(row.get(key, 0.0)) for row in history) / len(history)) for key in keys}

        @staticmethod
        def _split_metrics(final_scores, target_mask, candidate_mask, proposal_top1_is_true, *, total_molecules: int, shortlist_k: int, sources=None):
            n = int(final_scores.shape[0])
            corrected = 0.0
            harmed = 0.0
            top1_hits = 0.0
            top2_hits = 0.0
            top3_hits = 0.0
            shortlist_hits = 0.0
            proposal_top1_hits = 0.0
            source_rows = defaultdict(lambda: {"n": 0, "top1": 0.0, "top3": 0.0, "shortlist": 0.0})
            top1_gap = []
            tournament_margin = []
            for idx in range(n):
                valid = candidate_mask[idx] > 0.5
                if not bool(valid.any()):
                    continue
                scores = final_scores[idx, valid]
                labels = target_mask[idx, valid] > 0.5
                order = torch.argsort(scores, descending=True)
                proposal_hit = float(proposal_top1_is_true[idx].item()) > 0.5
                proposal_top1_hits += float(proposal_hit)
                top1_hit = bool(labels[order[0]].item())
                top2_hit = bool(labels[order[: min(2, int(order.shape[0]))]].any().item())
                top3_hit = bool(labels[order[: min(3, int(order.shape[0]))]].any().item())
                shortlist_hit = bool(labels[order[: min(int(shortlist_k), int(order.shape[0]))]].any().item())
                corrected += float((not proposal_hit) and top1_hit)
                harmed += float(proposal_hit and (not top1_hit))
                top1_hits += float(top1_hit)
                top2_hits += float(top2_hit)
                top3_hits += float(top3_hit)
                shortlist_hits += float(shortlist_hit)
                chosen_scores = scores[order]
                if int(chosen_scores.shape[0]) >= 2:
                    top1_gap.append(float((chosen_scores[0] - chosen_scores[1]).item()))
                src = str((sources or [""] * n)[idx])
                row = source_rows[src]
                row["n"] += 1
                row["top1"] += float(top1_hit)
                row["top3"] += float(top3_hit)
                row["shortlist"] += float(shortlist_hit)
            conditional_count = float(n) if n > 0 else 1.0
            source_breakdown = {
                name: {
                    "n": int(row["n"]),
                    "top1_acc_given_cache": float(row["top1"]) / float(row["n"]) if row["n"] > 0 else 0.0,
                    "top3_acc_given_cache": float(row["top3"]) / float(row["n"]) if row["n"] > 0 else 0.0,
                    "proposal_molecule_recall_at_k": float(row["shortlist"]) / float(row["n"]) if row["n"] > 0 else 0.0,
                }
                for name, row in sorted(source_rows.items())
            }
            return {
                "tournament_top1_acc_given_cache": float(top1_hits) / conditional_count,
                "tournament_top2_acc_given_cache": float(top2_hits) / conditional_count,
                "tournament_top3_acc_given_cache": float(top3_hits) / conditional_count,
                "proposal_top1_acc_given_cache": float(proposal_top1_hits) / conditional_count,
                "proposal_molecule_recall_at_k": float(shortlist_hits) / float(total_molecules) if total_molecules > 0 else 0.0,
                "end_to_end_top1": float(top1_hits) / float(total_molecules) if total_molecules > 0 else 0.0,
                "tournament_corrected_count": float(corrected),
                "tournament_harmed_count": float(harmed),
                "tournament_corrected_fraction": float(corrected) / conditional_count,
                "tournament_harmed_fraction": float(harmed) / conditional_count,
                "candidate_set_size_mean": float(candidate_mask.sum(dim=1).float().mean().item()) if candidate_mask.numel() else 0.0,
                "tournament_top1_gap_mean": float(sum(top1_gap) / len(top1_gap)) if top1_gap else 0.0,
                "source_breakdown": source_breakdown,
            }

        def evaluate(self, loader, *, split_summary: Optional[Dict[str, object]] = None) -> Dict[str, object]:
            self.model.eval()
            final_rows = []
            target_rows = []
            mask_rows = []
            proposal_top1_rows = []
            sources = []
            margin_history = []
            rival_history = []
            with torch.no_grad():
                for raw_batch in loader:
                    batch = self._move(raw_batch)
                    outputs = self.model(
                        batch["candidate_features"],
                        batch["candidate_mask"],
                        batch["proposal_scores"],
                        compare_top_n=int(self.compare_top_n),
                        candidate_local_rival_mask=batch.get("candidate_local_rival_mask"),
                        candidate_graph_distance=batch.get("candidate_graph_distance"),
                        candidate_3d_distance=batch.get("candidate_3d_distance"),
                        candidate_same_ring_system=batch.get("candidate_same_ring_system"),
                        candidate_same_topology_role=batch.get("candidate_same_topology_role"),
                        candidate_same_chem_family=batch.get("candidate_same_chem_family"),
                        candidate_branch_bulk=batch.get("candidate_branch_bulk"),
                        candidate_exposed_span=batch.get("candidate_exposed_span"),
                        candidate_anti_score=batch.get("candidate_anti_score"),
                    )
                    final_rows.append(outputs["final_scores"].detach().cpu())
                    target_rows.append(batch["target_mask"].detach().cpu())
                    mask_rows.append(batch["candidate_mask"].detach().cpu())
                    proposal_top1_rows.append(batch["proposal_top1_is_true"].detach().cpu())
                    valid = batch["candidate_mask"] > 0.5
                    if bool(valid.any()):
                        margin_history.append(float(outputs["tournament_margin"][valid].abs().mean().detach().cpu().item()))
                        rival_history.append(float(outputs["comparison_mask"].sum(dim=-1)[valid].float().mean().detach().cpu().item()))
                    sources.extend(list(raw_batch.get("source") or []))
            if not final_rows:
                return {}
            final_scores = torch.cat(final_rows, dim=0)
            target_mask = torch.cat(target_rows, dim=0)
            candidate_mask = torch.cat(mask_rows, dim=0)
            proposal_top1_is_true = torch.cat(proposal_top1_rows, dim=0)
            metrics = self._split_metrics(
                final_scores,
                target_mask,
                candidate_mask,
                proposal_top1_is_true,
                total_molecules=int((split_summary or {}).get("total_molecules", int(final_scores.shape[0]))),
                shortlist_k=int(self.shortlist_k),
                sources=sources,
            )
            metrics["tournament_margin_abs_mean"] = float(sum(margin_history) / len(margin_history)) if margin_history else 0.0
            metrics["local_rival_count_mean"] = float(sum(rival_history) / len(rival_history)) if rival_history else 0.0
            return metrics

        def rerank_split(self, loader, *, shortlist_k: int | None = None, total_molecules: int | None = None):
            self.model.eval()
            rows = []
            processed_rows = 0
            proposal_hit_molecules = 0
            with torch.no_grad():
                for raw_batch in loader:
                    batch = self._move(raw_batch)
                    outputs = self.model(
                        batch["candidate_features"],
                        batch["candidate_mask"],
                        batch["proposal_scores"],
                        compare_top_n=int(self.compare_top_n),
                        candidate_local_rival_mask=batch.get("candidate_local_rival_mask"),
                        candidate_graph_distance=batch.get("candidate_graph_distance"),
                        candidate_3d_distance=batch.get("candidate_3d_distance"),
                        candidate_same_ring_system=batch.get("candidate_same_ring_system"),
                        candidate_same_topology_role=batch.get("candidate_same_topology_role"),
                        candidate_same_chem_family=batch.get("candidate_same_chem_family"),
                        candidate_branch_bulk=batch.get("candidate_branch_bulk"),
                        candidate_exposed_span=batch.get("candidate_exposed_span"),
                        candidate_anti_score=batch.get("candidate_anti_score"),
                    )
                    final_scores = outputs["final_scores"].detach().cpu()
                    tournament_margin = outputs["tournament_margin"].detach().cpu()
                    candidate_features = batch["candidate_features"].detach().cpu()
                    candidate_mask = batch["candidate_mask"].detach().cpu()
                    target_mask = batch["target_mask"].detach().cpu()
                    candidate_atom_indices = batch["candidate_atom_indices"].detach().cpu()
                    proposal_scores = batch["proposal_scores"].detach().cpu()
                    for idx in range(int(final_scores.shape[0])):
                        valid = candidate_mask[idx] > 0.5
                        if not bool(valid.any()):
                            processed_rows += 1
                            continue
                        order = torch.argsort(final_scores[idx, valid], descending=True)
                        valid_idx = torch.nonzero(valid, as_tuple=False).view(-1)
                        keep_idx = valid_idx[order]
                        if shortlist_k is not None and int(shortlist_k) > 0:
                            keep_idx = keep_idx[: min(int(shortlist_k), int(keep_idx.shape[0]))]
                        row_target = target_mask[idx, keep_idx]
                        processed_rows += 1
                        proposal_hit = bool(row_target.any().item())
                        proposal_hit_molecules += int(proposal_hit)
                        if not proposal_hit:
                            continue
                        rows.append(
                            {
                                "molecule_id": str(raw_batch["molecule_id"][idx]),
                                "canonical_smiles": str(raw_batch["canonical_smiles"][idx]),
                                "source": str(raw_batch["source"][idx]),
                                "primary_cyp": str(raw_batch["primary_cyp"][idx]),
                                "candidate_features": candidate_features[idx, keep_idx].clone(),
                                "candidate_mask": torch.ones((int(keep_idx.shape[0]),), dtype=torch.float32),
                                "target_mask": row_target.clone(),
                                "candidate_atom_indices": candidate_atom_indices[idx, keep_idx].clone(),
                                "proposal_scores": final_scores[idx, keep_idx].clone(),
                                "proposal_top1_index": 0,
                                "proposal_top1_is_true": bool(row_target[0].item() > 0.5),
                                "true_site_atoms": [int(v) for v in list(raw_batch["true_site_atoms"][idx])],
                                "base_proposal_scores": proposal_scores[idx, keep_idx].clone(),
                                "tournament_margin": tournament_margin[idx, keep_idx].clone(),
                            }
                        )
            total_count = int(total_molecules) if total_molecules is not None and int(total_molecules) > 0 else int(processed_rows)
            return {
                "samples": rows,
                "summary": {
                    "total_molecules": int(total_count),
                    "proposal_hit_molecules": int(proposal_hit_molecules),
                    "proposal_molecule_recall_at_k": float(proposal_hit_molecules) / float(total_count) if total_count > 0 else 0.0,
                    "conditional_sample_count": int(len(rows)),
                    "feature_dim": int(rows[0]["candidate_features"].shape[-1]) if rows else 0,
                },
            }

        def save_checkpoint(self, path: str | Path, *, feature_dim: int, hidden_dim: int, extra: Optional[Dict[str, object]] = None) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "feature_dim": int(feature_dim),
                    "hidden_dim": int(hidden_dim),
                    "extra": dict(extra or {}),
                },
                path,
            )
else:  # pragma: no cover
    @dataclass
    class PairwiseSiteTournamentTrainer:  # type: ignore[override]
        model: object

        def __post_init__(self):
            require_torch()
