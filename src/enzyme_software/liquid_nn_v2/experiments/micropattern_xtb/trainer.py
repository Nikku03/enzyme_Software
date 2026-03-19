from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from enzyme_software.losses import ListMLELoss, MIRankLoss
from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.config import MicroPatternXTBConfig
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.metrics import compute_reranker_metrics
from enzyme_software.liquid_nn_v2.training.utils import move_to_device


if TORCH_AVAILABLE:
    @dataclass
    class MicroPatternTrainer:
        model: object
        config: MicroPatternXTBConfig
        device: Optional[torch.device] = None

        def __post_init__(self):
            self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
            )
            self.mirank = MIRankLoss(
                margin=max(0.1, float(self.config.pairwise_margin)),
                hard_negative_fraction=self.config.hard_negative_fraction,
            )
            self.listmle = ListMLELoss()

        def step_scheduler(self, val_metric: float) -> None:
            self.scheduler.step(val_metric)

        def maybe_unfreeze(self, epoch: int) -> None:
            if not self.config.freeze_base_model:
                return
            if int(self.config.unfreeze_after_epochs) <= 0 or epoch < int(self.config.unfreeze_after_epochs):
                return
            if any(param.requires_grad for param in self.model.base_model.parameters()):
                return
            self.model.set_base_trainable(True)
            finetune_lr = self.config.finetune_learning_rate or self.config.learning_rate * 0.25
            self.optimizer = torch.optim.AdamW(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=finetune_lr,
                weight_decay=self.config.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
            )

        def _prepare_batch(self, batch):
            return move_to_device(batch, self.device)

        def compute_loss(self, outputs: Dict[str, object]) -> tuple[torch.Tensor, Dict[str, float]]:
            valid = outputs["candidate_valid"]
            positive = outputs["candidate_positive"]
            scores = outputs["reranked_candidate_scores"]
            masked_scores = scores.masked_fill(~valid, float("-inf"))
            valid_rows = valid.any(dim=1) & positive.any(dim=1)
            if not bool(valid_rows.any()):
                zero = scores.sum() * 0.0
                stats = compute_reranker_metrics(outputs, outputs)
                stats.update(outputs.get("stats", {}))
                stats["loss"] = 0.0
                return zero, stats
            masked_scores = masked_scores[valid_rows]
            positive = positive[valid_rows]
            pos_distribution = positive.float()
            pos_distribution = pos_distribution / pos_distribution.sum(dim=1, keepdim=True).clamp(min=1.0)
            log_probs = torch.log_softmax(masked_scores, dim=1).clamp(min=-1e9)
            candidate_ce = -(pos_distribution * log_probs).sum(dim=1).mean()
            mirank_terms = []
            listmle_terms = []
            pairwise_terms = []
            for row_scores, row_valid, row_positive in zip(scores[valid_rows], valid[valid_rows], positive[valid_rows]):
                local_scores = row_scores[row_valid]
                local_positive = row_positive[row_valid]
                pos_idx = torch.nonzero(local_positive, as_tuple=False).view(-1).tolist()
                if not pos_idx:
                    continue
                listmle_terms.append(self.listmle(local_scores, pos_idx))
                if len(pos_idx) >= int(local_scores.numel()):
                    continue
                mirank_terms.append(self.mirank(local_scores, pos_idx))
                pos_scores = local_scores[pos_idx]
                neg_mask = torch.ones((int(local_scores.numel()),), dtype=torch.bool, device=local_scores.device)
                neg_mask[pos_idx] = False
                neg_scores = local_scores[neg_mask]
                pairwise_terms.append(torch.relu(self.config.pairwise_margin - (pos_scores.max() - neg_scores.max())))
            zero = scores.sum() * 0.0
            mirank_loss = torch.stack(mirank_terms).mean() if mirank_terms else zero
            listmle_loss = torch.stack(listmle_terms).mean() if listmle_terms else zero
            pairwise_margin = torch.stack(pairwise_terms).mean() if pairwise_terms else zero
            total = (
                self.config.candidate_ce_weight * candidate_ce
                + self.config.pairwise_margin_weight * pairwise_margin
                + self.config.mirank_weight * mirank_loss
                + self.config.listmle_weight * listmle_loss
            )
            stats = compute_reranker_metrics(outputs, outputs)
            stats.update(outputs.get("stats", {}))
            stats["candidate_ce_loss"] = float(candidate_ce.item())
            stats["pairwise_margin_loss"] = float(pairwise_margin.item())
            stats["mirank_loss"] = float(mirank_loss.item())
            stats["listmle_loss"] = float(listmle_loss.item())
            stats["loss"] = float(total.item())
            return total, stats

        def train_epoch(self, loader, epoch: int) -> Dict[str, float]:
            self.maybe_unfreeze(epoch)
            self.model.train()
            history = []
            for raw_batch in loader:
                if raw_batch is None:
                    continue
                batch = self._prepare_batch(raw_batch)
                outputs = self.model(batch)
                loss, stats = self.compute_loss(outputs)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad and p.grad is not None], 1.0)
                self.optimizer.step()
                history.append(stats)
            if not history:
                return {"loss": float("inf")}
            keys = sorted({k for item in history for k in item.keys()})
            return {k: float(sum(float(item.get(k, 0.0)) for item in history) / len(history)) for k in keys}

        def evaluate(self, loader) -> Dict[str, float]:
            self.model.eval()
            history = []
            with torch.no_grad():
                for raw_batch in loader:
                    if raw_batch is None:
                        continue
                    batch = self._prepare_batch(raw_batch)
                    outputs = self.model(batch)
                    _, stats = self.compute_loss(outputs)
                    history.append(stats)
            if not history:
                return {"loss": float("inf")}
            keys = sorted({k for item in history for k in item.keys()})
            return {k: float(sum(float(item.get(k, 0.0)) for item in history) / len(history)) for k in keys}
else:  # pragma: no cover
    class MicroPatternTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
