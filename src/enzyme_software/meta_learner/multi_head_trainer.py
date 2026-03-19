from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import DataLoader

from enzyme_software.losses import CombinedSiteRankingLoss, SiteRankingLossV2
from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.meta_learner.config import MetaLearnerConfig
from enzyme_software.meta_learner.meta_evaluator import evaluate_meta_predictions
from enzyme_software.meta_learner.meta_trainer import MetaLearnerDataset, collate_meta_batch


if TORCH_AVAILABLE:
    @dataclass
    class MultiHeadTrainer:
        model: object
        train_dataset: MetaLearnerDataset
        val_dataset: MetaLearnerDataset
        config: MetaLearnerConfig
        device: Optional[torch.device] = None
        site_loss_weight: float = 1.0
        cyp_loss_weight: float = 0.5
        attention_entropy_weight: float = 0.01

        def __post_init__(self):
            self.device = self.device or (
                torch.device("mps")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.model.to(self.device)
            self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, collate_fn=collate_meta_batch)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, collate_fn=collate_meta_batch)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=5)
            self.history: List[Dict[str, object]] = []
            self.best_val_top1 = -1.0
            if float(self.config.listmle_weight) > 0.0 or float(self.config.focal_weight) > 0.0:
                self.site_loss_fn = CombinedSiteRankingLoss(
                    mirank_weight=self.config.mirank_weight,
                    listmle_weight=self.config.listmle_weight,
                    bce_weight=self.config.bce_weight,
                    focal_weight=self.config.focal_weight,
                    margin=self.config.ranking_margin,
                    hard_negative_fraction=self.config.hard_negative_fraction,
                )
            else:
                self.site_loss_fn = SiteRankingLossV2(
                    mirank_weight=self.config.mirank_weight,
                    bce_weight=self.config.bce_weight,
                    margin=self.config.ranking_margin,
                    hard_negative_fraction=self.config.hard_negative_fraction,
                )

        def _move(self, batch: Dict[str, object]) -> Dict[str, object]:
            out: Dict[str, object] = {}
            for key, value in batch.items():
                out[key] = value.to(self.device) if hasattr(value, "to") else value
            return out

        @staticmethod
        def _entropy_target(attention: torch.Tensor, target: float = 0.9) -> torch.Tensor:
            entropy = -(attention * torch.log(attention.clamp_min(1.0e-8))).sum()
            return torch.abs(entropy - target)

        def train_epoch(self) -> Dict[str, float]:
            self.model.train()
            total_loss = 0.0
            total_site_loss = 0.0
            total_cyp_loss = 0.0
            total_entropy = 0.0
            n = 0
            site_loss_stats = {"mirank": 0.0, "bce": 0.0, "total": 0.0}
            for raw in self.train_loader:
                batch = self._move(raw)
                site_logits, cyp_logits, stats = self.model(
                    batch["atom_features"],
                    batch["global_features"],
                    batch["site_scores_raw"],
                    batch["cyp_probs_raw"],
                )
                if bool(batch["site_supervised"].item()):
                    site_loss, site_loss_stats = self.site_loss_fn(site_logits, batch["site_labels"])
                else:
                    site_loss = site_logits.sum() * 0.0
                    site_loss_stats = {"mirank": 0.0, "bce": 0.0, "total": 0.0}
                cyp_loss = torch.nn.functional.cross_entropy(cyp_logits.unsqueeze(0), batch["cyp_label"].view(1))
                entropy_loss = self._entropy_target(stats["site_attention"]) + self._entropy_target(stats["cyp_attention"])
                loss = (
                    self.site_loss_weight * site_loss
                    + self.cyp_loss_weight * cyp_loss
                    + self.attention_entropy_weight * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += float(loss.item())
                total_site_loss += float(site_loss.item())
                total_cyp_loss += float(cyp_loss.item())
                total_entropy += float(entropy_loss.item())
                n += 1
            denom = max(1, n)
            return {
                "loss": total_loss / denom,
                "site_loss": total_site_loss / denom,
                "cyp_loss": total_cyp_loss / denom,
                "entropy_loss": total_entropy / denom,
                "site_mirank": float(site_loss_stats.get("mirank", 0.0)),
                "site_bce": float(site_loss_stats.get("bce", 0.0)),
            }

        def validate(self) -> Dict[str, float | list[float]]:
            self.model.eval()
            n = 0
            n_site = 0
            top1 = 0.0
            top3 = 0.0
            cyp = 0.0
            site_attn_sum = None
            cyp_attn_sum = None
            with torch.no_grad():
                for raw in self.val_loader:
                    batch = self._move(raw)
                    site_logits, cyp_logits, stats = self.model(
                        batch["atom_features"],
                        batch["global_features"],
                        batch["site_scores_raw"],
                        batch["cyp_probs_raw"],
                    )
                    cyp += float(int(torch.argmax(cyp_logits.detach()).item()) == int(batch["cyp_label"].item()))
                    if bool(batch["site_supervised"].item()):
                        metrics = evaluate_meta_predictions(site_logits, batch["site_labels"], cyp_logits, int(batch["cyp_label"].item()))
                        top1 += metrics["site_top1"]
                        top3 += metrics["site_top3"]
                        n_site += 1
                    site_attn_sum = stats["site_attention"].detach().cpu() if site_attn_sum is None else site_attn_sum + stats["site_attention"].detach().cpu()
                    cyp_attn_sum = stats["cyp_attention"].detach().cpu() if cyp_attn_sum is None else cyp_attn_sum + stats["cyp_attention"].detach().cpu()
                    n += 1
            denom = max(1, n)
            site_denom = max(1, n_site)
            out: Dict[str, float | list[float]] = {
                "site_top1": top1 / site_denom,
                "site_top3": top3 / site_denom,
                "cyp_acc": cyp / denom,
                "n_samples": float(n),
                "n_site_samples": float(n_site),
            }
            if n > 0:
                out["site_attention"] = (site_attn_sum / float(n)).tolist()
                out["cyp_attention"] = (cyp_attn_sum / float(n)).tolist()
            return out

        def _save_progress(self, *, best_state, last_val_metrics: Optional[Dict[str, object]] = None, status: str = "running") -> Dict[str, object]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            latest_path = Path(self.config.checkpoint_dir) / "multihead_meta_learner_latest.pt"
            best_path = Path(self.config.checkpoint_dir) / "multihead_meta_learner_best.pt"
            archive_path = Path(self.config.checkpoint_dir) / f"multihead_meta_learner_{timestamp}.pt"
            report_path = Path(self.config.artifact_dir) / f"multihead_meta_learner_report_{timestamp}.json"
            payload = {
                "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "config": self.config.__dict__,
                "best_val_top1": self.best_val_top1,
                "final_site_attention": (last_val_metrics or {}).get("site_attention", []),
                "final_cyp_attention": (last_val_metrics or {}).get("cyp_attention", []),
                "history": self.history,
                "status": status,
            }
            torch.save(payload, latest_path)
            if best_state is not None:
                best_payload = dict(payload)
                best_payload["model_state_dict"] = best_state
                best_payload["status"] = f"{status}_best"
                torch.save(best_payload, best_path)
            torch.save(payload, archive_path)
            report_path.write_text(
                json.dumps(
                    {
                        "best_val_top1": self.best_val_top1,
                        "final_site_attention": payload["final_site_attention"],
                        "final_cyp_attention": payload["final_cyp_attention"],
                        "history": self.history,
                        "status": status,
                    },
                    indent=2,
                )
            )
            return {
                "payload": payload,
                "latest_path": str(latest_path),
                "best_path": str(best_path),
                "archive_path": str(archive_path),
                "report_path": str(report_path),
            }

        def train(self) -> Dict[str, object]:
            self.config.ensure_dirs()
            patience_left = self.config.patience
            best_state = None
            last_val_metrics: Dict[str, object] = {}
            try:
                for epoch in range(1, int(self.config.epochs) + 1):
                    train_metrics = self.train_epoch()
                    val_metrics = self.validate()
                    last_val_metrics = val_metrics
                    self.scheduler.step(float(val_metrics["site_top1"]))
                    self.history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
                    site_attn = val_metrics.get("site_attention") or []
                    cyp_attn = val_metrics.get("cyp_attention") or []
                    print(
                        f"Epoch {epoch:3d} | loss={train_metrics['loss']:.4f} | "
                        f"site_top1={float(val_metrics['site_top1']):.3f} | site_top3={float(val_metrics['site_top3']):.3f} | "
                        f"cyp={float(val_metrics['cyp_acc']):.3f} | "
                        f"site_attn=[{','.join(f'{float(x):.2f}' for x in site_attn)}] | "
                        f"cyp_attn=[{','.join(f'{float(x):.2f}' for x in cyp_attn)}]",
                        flush=True,
                    )
                    if float(val_metrics["site_top1"]) > self.best_val_top1:
                        self.best_val_top1 = float(val_metrics["site_top1"])
                        best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                        patience_left = self.config.patience
                    else:
                        patience_left -= 1
                    self._save_progress(best_state=best_state, last_val_metrics=last_val_metrics, status="running")
                    if patience_left <= 0:
                        break
            except KeyboardInterrupt:
                saved = self._save_progress(best_state=best_state, last_val_metrics=last_val_metrics, status="interrupted")
                print(f"Interrupted. Saved latest checkpoint: {saved['latest_path']}", flush=True)
                print(f"Saved best checkpoint: {saved['best_path']}", flush=True)
                print(f"Saved report: {saved['report_path']}", flush=True)
                return saved["payload"]
            if best_state is not None:
                self.model.load_state_dict(best_state, strict=False)
            saved = self._save_progress(best_state=best_state, last_val_metrics=last_val_metrics, status="completed")
            return saved["payload"]
else:  # pragma: no cover
    class MultiHeadTrainer:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            require_torch()
