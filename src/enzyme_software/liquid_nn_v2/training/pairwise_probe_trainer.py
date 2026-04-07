from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.training.metrics import _safe_binary_auc
from enzyme_software.liquid_nn_v2.training.pairwise_probe import (
    apply_candidate_mask_to_site_logits,
    build_pairwise_probe_examples,
    zero_pairwise_probe_metrics,
)
from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs, move_to_device


if TORCH_AVAILABLE:
    @dataclass
    class PairwiseProbeTrainer:
        """Frozen-backbone diagnostic probe over (true atom, top-score hard negative) pairs.

        Atom embeddings come from ``outputs["atom_features"]`` and proposer scores come
        from the masked ``outputs["site_logits"]`` path. The wrapped model runs under
        ``torch.no_grad()`` so Stage 1 trains only the PairwiseHead.
        """

        model: object
        pairwise_head: object
        learning_rate: float = 1.0e-3
        weight_decay: float = 1.0e-4
        max_grad_norm: float = 5.0
        max_pairs_per_batch: Optional[int] = None
        freeze_backbone: bool = True
        freeze_proposer: bool = True
        device: Optional[torch.device] = None

        def __post_init__(self):
            self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.pairwise_head.to(self.device)
            if self.freeze_backbone or self.freeze_proposer:
                for param in self.model.parameters():
                    param.requires_grad = False
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
            self.optimizer = torch.optim.AdamW(
                [param for param in self.pairwise_head.parameters() if param.requires_grad],
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )

        def _prepare_batch(self, batch_or_graphs):
            if isinstance(batch_or_graphs, dict):
                return move_to_device(batch_or_graphs, self.device)
            return move_to_device(collate_molecule_graphs(batch_or_graphs), self.device)

        def _build_batch_pairs(self, batch: Dict[str, object]):
            with torch.no_grad():
                outputs = self.model(batch)
                config = getattr(self.model, "config", None)
                site_logits = apply_candidate_mask_to_site_logits(
                    outputs.get("site_logits"),
                    batch.get("candidate_mask", batch.get("candidate_train_mask")),
                    mask_mode=str(getattr(config, "candidate_mask_mode", "hard") or "hard"),
                    logit_bias=float(getattr(config, "candidate_mask_logit_bias", 2.0)),
                )
            atom_features = outputs.get("atom_features")
            if atom_features is None or site_logits is None:
                raise RuntimeError("Pairwise probe requires model outputs['atom_features'] and outputs['site_logits']")
            pair_features, pair_labels, metadata = build_pairwise_probe_examples(
                atom_embeddings=atom_features.detach(),
                site_logits=site_logits.detach(),
                site_labels=batch["site_labels"],
                batch_index=batch["batch"],
                supervision_mask=batch.get("site_supervision_mask"),
                candidate_mask=batch.get("candidate_mask", batch.get("candidate_train_mask")),
                max_pairs=self.max_pairs_per_batch,
            )
            expected_dim = (4 * int(atom_features.size(-1))) + 2
            if int(pair_features.size(-1)) != int(expected_dim):
                raise ValueError(
                    f"Expected pairwise probe feature dim {expected_dim}, got {int(pair_features.size(-1))}"
                )
            return pair_features, pair_labels, metadata

        def _compute_batch_metrics(self, pair_logits, pair_labels, metadata: Dict[str, float]):
            probabilities = torch.sigmoid(pair_logits)
            pair_count = int(pair_labels.numel())
            accuracy = float(((probabilities >= 0.5).float() == pair_labels).float().mean().item()) if pair_count else 0.0
            auc = float(_safe_binary_auc(pair_labels.detach().cpu().numpy(), probabilities.detach().cpu().numpy())) if pair_count else 0.5
            return {
                "pairwise_accuracy": accuracy,
                "pairwise_auc": auc,
                "pair_count": float(pair_count),
                "molecules_with_pairs": float(metadata.get("molecules_with_pairs", 0.0)),
                "hard_neg_rank_mean": float(metadata.get("hard_neg_rank_mean", 0.0)),
                "score_gap_mean": float(metadata.get("score_gap_mean", 0.0)),
                "true_beats_fraction_before_pairwise": float(metadata.get("true_beats_fraction_before_pairwise", 0.0)),
                "pairwise_mean_probability": float(probabilities.mean().item()) if pair_count else 0.0,
                "pairwise_positive_rate": float((probabilities >= 0.5).float().mean().item()) if pair_count else 0.0,
            }

        def _merge_epoch_stats(self, accum: Dict[str, float]) -> Dict[str, float]:
            total_pairs = float(accum.get("pair_count", 0.0))
            zero_pair_batches = float(accum.get("zero_pair_batches", 0.0))
            if total_pairs <= 0.0:
                stats = zero_pairwise_probe_metrics()
                stats["zero_pair_batches"] = zero_pair_batches
                return stats
            return {
                "pairwise_loss": float(accum.get("pairwise_loss_sum", 0.0)) / total_pairs,
                "pairwise_accuracy": float(accum.get("pairwise_accuracy_sum", 0.0)) / total_pairs,
                "pairwise_auc": float(accum.get("pairwise_auc_sum", 0.0)) / total_pairs,
                "pair_count": total_pairs,
                "molecules_with_pairs": float(accum.get("molecules_with_pairs", 0.0)),
                "hard_neg_rank_mean": float(accum.get("hard_neg_rank_sum", 0.0)) / total_pairs,
                "score_gap_mean": float(accum.get("score_gap_sum", 0.0)) / total_pairs,
                "true_beats_fraction_before_pairwise": float(accum.get("true_beats_sum", 0.0)) / total_pairs,
                "pairwise_mean_probability": float(accum.get("pairwise_probability_sum", 0.0)) / total_pairs,
                "pairwise_positive_rate": float(accum.get("pairwise_positive_rate_sum", 0.0)) / total_pairs,
                "zero_pair_batches": zero_pair_batches,
            }

        def train_loader_epoch(self, loader) -> Dict[str, float]:
            self.model.eval()
            self.pairwise_head.train()
            accum: Dict[str, float] = {"zero_pair_batches": 0.0}
            for raw_batch in loader:
                if raw_batch is None:
                    continue
                batch = self._prepare_batch(raw_batch)
                pair_features, pair_labels, metadata = self._build_batch_pairs(batch)
                if int(pair_labels.numel()) == 0:
                    accum["zero_pair_batches"] = float(accum.get("zero_pair_batches", 0.0)) + 1.0
                    continue
                pair_logits = self.pairwise_head(pair_features).view(-1)
                loss = self.loss_fn(pair_logits, pair_labels)
                if not bool(torch.isfinite(pair_features).all()):
                    raise FloatingPointError("Non-finite pairwise probe features detected in training")
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite pairwise probe loss detected")
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pairwise_head.parameters(), float(self.max_grad_norm))
                self.optimizer.step()

                batch_metrics = self._compute_batch_metrics(pair_logits.detach(), pair_labels.detach(), metadata)
                pair_count = float(batch_metrics["pair_count"])
                accum["pair_count"] = float(accum.get("pair_count", 0.0)) + pair_count
                accum["molecules_with_pairs"] = float(accum.get("molecules_with_pairs", 0.0)) + float(batch_metrics["molecules_with_pairs"])
                accum["pairwise_loss_sum"] = float(accum.get("pairwise_loss_sum", 0.0)) + (float(loss.detach().item()) * pair_count)
                accum["pairwise_accuracy_sum"] = float(accum.get("pairwise_accuracy_sum", 0.0)) + (float(batch_metrics["pairwise_accuracy"]) * pair_count)
                accum["pairwise_auc_sum"] = float(accum.get("pairwise_auc_sum", 0.0)) + (float(batch_metrics["pairwise_auc"]) * pair_count)
                accum["hard_neg_rank_sum"] = float(accum.get("hard_neg_rank_sum", 0.0)) + (float(batch_metrics["hard_neg_rank_mean"]) * pair_count)
                accum["score_gap_sum"] = float(accum.get("score_gap_sum", 0.0)) + (float(batch_metrics["score_gap_mean"]) * pair_count)
                accum["true_beats_sum"] = float(accum.get("true_beats_sum", 0.0)) + (float(batch_metrics["true_beats_fraction_before_pairwise"]) * pair_count)
                accum["pairwise_probability_sum"] = float(accum.get("pairwise_probability_sum", 0.0)) + (float(batch_metrics["pairwise_mean_probability"]) * pair_count)
                accum["pairwise_positive_rate_sum"] = float(accum.get("pairwise_positive_rate_sum", 0.0)) + (float(batch_metrics["pairwise_positive_rate"]) * pair_count)
            return self._merge_epoch_stats(accum)

        def evaluate_loader(self, loader) -> Dict[str, float]:
            self.model.eval()
            self.pairwise_head.eval()
            accum: Dict[str, float] = {"zero_pair_batches": 0.0}
            with torch.no_grad():
                for raw_batch in loader:
                    if raw_batch is None:
                        continue
                    batch = self._prepare_batch(raw_batch)
                    pair_features, pair_labels, metadata = self._build_batch_pairs(batch)
                    if int(pair_labels.numel()) == 0:
                        accum["zero_pair_batches"] = float(accum.get("zero_pair_batches", 0.0)) + 1.0
                        continue
                    pair_logits = self.pairwise_head(pair_features).view(-1)
                    loss = self.loss_fn(pair_logits, pair_labels)
                    batch_metrics = self._compute_batch_metrics(pair_logits, pair_labels, metadata)
                    pair_count = float(batch_metrics["pair_count"])
                    accum["pair_count"] = float(accum.get("pair_count", 0.0)) + pair_count
                    accum["molecules_with_pairs"] = float(accum.get("molecules_with_pairs", 0.0)) + float(batch_metrics["molecules_with_pairs"])
                    accum["pairwise_loss_sum"] = float(accum.get("pairwise_loss_sum", 0.0)) + (float(loss.detach().item()) * pair_count)
                    accum["pairwise_accuracy_sum"] = float(accum.get("pairwise_accuracy_sum", 0.0)) + (float(batch_metrics["pairwise_accuracy"]) * pair_count)
                    accum["pairwise_auc_sum"] = float(accum.get("pairwise_auc_sum", 0.0)) + (float(batch_metrics["pairwise_auc"]) * pair_count)
                    accum["hard_neg_rank_sum"] = float(accum.get("hard_neg_rank_sum", 0.0)) + (float(batch_metrics["hard_neg_rank_mean"]) * pair_count)
                    accum["score_gap_sum"] = float(accum.get("score_gap_sum", 0.0)) + (float(batch_metrics["score_gap_mean"]) * pair_count)
                    accum["true_beats_sum"] = float(accum.get("true_beats_sum", 0.0)) + (float(batch_metrics["true_beats_fraction_before_pairwise"]) * pair_count)
                    accum["pairwise_probability_sum"] = float(accum.get("pairwise_probability_sum", 0.0)) + (float(batch_metrics["pairwise_mean_probability"]) * pair_count)
                    accum["pairwise_positive_rate_sum"] = float(accum.get("pairwise_positive_rate_sum", 0.0)) + (float(batch_metrics["pairwise_positive_rate"]) * pair_count)
            return self._merge_epoch_stats(accum)
else:  # pragma: no cover
    @dataclass
    class PairwiseProbeTrainer:  # type: ignore[override]
        model: object
        pairwise_head: object

        def __post_init__(self):
            require_torch()
