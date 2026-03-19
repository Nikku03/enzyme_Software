from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.training.utils import move_to_device
from .config import RecursiveMetabolismConfig
from .utils import initialized_state_dict


if TORCH_AVAILABLE:
    @dataclass
    class RecursiveMetabolismTrainer:
        model: object
        config: RecursiveMetabolismConfig
        device: Optional[torch.device] = None

        def __post_init__(self):
            self.device = self.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.optimizer = torch.optim.AdamW(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        def maybe_unfreeze(self, epoch: int) -> None:
            if not self.config.freeze_base_model:
                return
            if int(self.config.unfreeze_after_epochs) <= 0 or epoch < int(self.config.unfreeze_after_epochs):
                return
            if any(param.requires_grad for param in self.model.base_model.parameters()):
                return
            self.model.set_base_trainable(True)
            lr = self.config.finetune_learning_rate or (self.config.learning_rate * 0.25)
            self.optimizer = torch.optim.AdamW(
                [param for param in self.model.parameters() if param.requires_grad],
                lr=lr,
                weight_decay=self.config.weight_decay,
            )

        def _prepare_batch(self, batch):
            return move_to_device(batch, self.device)

        def compute_loss_and_metrics(self, outputs: Dict[str, object], batch: Dict[str, object]) -> tuple[torch.Tensor, Dict[str, float]]:
            site_logits = outputs["recursive_site_logits"]
            base_site_logits = outputs["base_site_logits"]
            site_labels = batch["site_labels"].squeeze(-1)
            batch_index = batch["batch"]
            num_graphs = int(batch["num_graphs"])

            losses = []
            top1 = top3 = base_top1 = base_top3 = 0
            weighted_margin = []
            for graph_idx in range(num_graphs):
                atom_mask = batch_index == graph_idx
                graph_logits = site_logits[atom_mask].squeeze(-1)
                graph_base_logits = base_site_logits[atom_mask].squeeze(-1)
                graph_labels = site_labels[atom_mask]
                positives = torch.nonzero(graph_labels > 0.5, as_tuple=False).view(-1)
                if positives.numel() == 0:
                    continue
                target = positives[0].unsqueeze(0)
                weight = (
                    batch["graph_source_weights"][graph_idx]
                    * (self.config.step_weight_decay ** int(batch["graph_step_numbers"][graph_idx].item()))
                )
                graph_loss = torch.nn.functional.cross_entropy(graph_logits.unsqueeze(0), target)
                losses.append(weight * graph_loss)

                ranking = torch.argsort(graph_logits, descending=True)
                base_ranking = torch.argsort(graph_base_logits, descending=True)
                target_idx = int(target.item())
                top1 += int(int(ranking[0].item()) == target_idx)
                top3 += int(any(int(v.item()) == target_idx for v in ranking[: min(3, ranking.numel())]))
                base_top1 += int(int(base_ranking[0].item()) == target_idx)
                base_top3 += int(any(int(v.item()) == target_idx for v in base_ranking[: min(3, base_ranking.numel())]))
                if graph_logits.numel() > 1:
                    wrong_mask = torch.ones_like(graph_logits, dtype=torch.bool)
                    wrong_mask[target_idx] = False
                    best_wrong = graph_logits[wrong_mask].max()
                    weighted_margin.append(float((graph_logits[target_idx] - best_wrong).detach().item()))

            if not losses:
                zero = site_logits.sum() * 0.0
                return zero, {"loss": 0.0}
            loss = torch.stack(losses).mean()
            valid_graphs = len(losses)
            metrics = {
                "loss": float(loss.detach().item()),
                "top1_acc": float(top1 / valid_graphs),
                "top3_acc": float(top3 / valid_graphs),
                "base_top1_acc": float(base_top1 / valid_graphs),
                "base_top3_acc": float(base_top3 / valid_graphs),
                "margin_mean": float(sum(weighted_margin) / len(weighted_margin)) if weighted_margin else 0.0,
            }
            metrics.update(outputs.get("diagnostics", {}))
            return loss, metrics

        def _run_epoch(self, loader, *, train: bool, epoch: int = 0) -> Dict[str, float]:
            if train:
                self.maybe_unfreeze(epoch)
                self.model.train()
            else:
                self.model.eval()
            history = []
            context = torch.enable_grad() if train else torch.no_grad()
            with context:
                for raw_batch in loader:
                    if raw_batch is None:
                        continue
                    batch = self._prepare_batch(raw_batch)
                    outputs = self.model(batch)
                    loss, metrics = self.compute_loss_and_metrics(outputs, batch)
                    if train:
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [param for param in self.model.parameters() if param.requires_grad and param.grad is not None],
                            1.0,
                        )
                        self.optimizer.step()
                    history.append(metrics)
            if not history:
                return {"loss": float("inf")}
            keys = sorted({key for item in history for key in item.keys()})
            return {key: float(sum(float(item.get(key, 0.0)) for item in history) / len(history)) for key in keys}

        def train_epoch(self, loader, epoch: int) -> Dict[str, float]:
            return self._run_epoch(loader, train=True, epoch=epoch)

        def evaluate(self, loader) -> Dict[str, float]:
            return self._run_epoch(loader, train=False)

        def save_checkpoint(self, path: str | Path, payload: Dict[str, object]) -> None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, out)
else:  # pragma: no cover
    class RecursiveMetabolismTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
