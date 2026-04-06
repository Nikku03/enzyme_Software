from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from enzyme_software.liquid_nn_v2._compat import F, TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.training.metrics import compute_candidate_winner_metrics


if TORCH_AVAILABLE:
    @dataclass
    class CandidateWinnerTrainer:
        model: object
        learning_rate: float = 1.0e-3
        weight_decay: float = 1.0e-4
        margin_weight: float = 0.25
        margin_value: float = 0.30
        do_no_harm_weight: float = 0.20
        do_no_harm_margin: float = 0.15
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

        def _loss(self, logits, target_mask, candidate_mask, *, proposal_scores=None, proposal_top1_index=None, proposal_top1_is_true=None):
            target = target_mask.float()
            target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
            ce = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            margin_losses = []
            for row_logits, row_target, row_mask in zip(logits, target_mask, candidate_mask):
                valid = row_mask > 0.5
                if not bool(valid.any()):
                    continue
                pos = (row_target > 0.5) & valid
                neg = (row_target <= 0.5) & valid
                if not bool(pos.any()) or not bool(neg.any()):
                    continue
                best_pos = row_logits[pos].max()
                margin_losses.append(torch.relu(float(self.margin_value) - (best_pos - row_logits[neg])).mean())
            margin = torch.stack(margin_losses).mean() if margin_losses else logits.sum() * 0.0
            do_no_harm_losses = []
            if proposal_top1_index is not None and proposal_top1_is_true is not None:
                for row_logits, row_mask, row_prop_idx, row_prop_true in zip(logits, candidate_mask, proposal_top1_index, proposal_top1_is_true):
                    if float(row_prop_true.item()) <= 0.5:
                        continue
                    valid = row_mask > 0.5
                    if not bool(valid.any()):
                        continue
                    prop_idx = int(row_prop_idx.item())
                    if prop_idx < 0 or prop_idx >= int(row_logits.shape[0]) or not bool(valid[prop_idx].item()):
                        continue
                    other_mask = valid.clone()
                    other_mask[prop_idx] = False
                    if not bool(other_mask.any()):
                        continue
                    proposal_logit = row_logits[prop_idx]
                    best_other = row_logits[other_mask].max()
                    do_no_harm_losses.append(torch.relu(float(self.do_no_harm_margin) - (proposal_logit - best_other)))
            do_no_harm = torch.stack(do_no_harm_losses).mean() if do_no_harm_losses else logits.sum() * 0.0
            total = ce + (float(self.margin_weight) * margin) + (float(self.do_no_harm_weight) * do_no_harm)
            valid_logits = logits.masked_fill(candidate_mask <= 0.5, float("-inf"))
            top2_vals = torch.topk(valid_logits, k=min(2, int(valid_logits.shape[1])), dim=1).values
            top1_gap = (
                (top2_vals[:, 0] - top2_vals[:, 1]).mean()
                if int(top2_vals.shape[1]) >= 2
                else valid_logits[:, 0].mean()
            )
            winner_probs = torch.softmax(valid_logits, dim=1)
            winner_top1_prob = torch.max(winner_probs, dim=1).values.mean()
            proposal_top1_gap = logits.sum() * 0.0
            if proposal_scores is not None:
                valid_prop = proposal_scores.masked_fill(candidate_mask <= 0.5, float("-inf"))
                prop_top2_vals = torch.topk(valid_prop, k=min(2, int(valid_prop.shape[1])), dim=1).values
                proposal_top1_gap = (
                    (prop_top2_vals[:, 0] - prop_top2_vals[:, 1]).mean()
                    if int(prop_top2_vals.shape[1]) >= 2
                    else prop_top2_vals[:, 0].mean()
                )
            return total, {
                "candidate_set_ce": float(ce.detach().item()),
                "candidate_set_margin": float(margin.detach().item()),
                "candidate_set_do_no_harm": float(do_no_harm.detach().item()),
                "candidate_set_total_loss": float(total.detach().item()),
                "winner_top1_gap_mean": float(top1_gap.detach().item()),
                "winner_top1_prob_mean": float(winner_top1_prob.detach().item()),
                "proposal_top1_gap_mean": float(proposal_top1_gap.detach().item()),
                "do_no_harm_active_fraction": float((proposal_top1_is_true > 0.5).float().mean().item()) if proposal_top1_is_true is not None else 0.0,
            }

        def train_epoch(self, loader) -> Dict[str, float]:
            self.model.train()
            history = []
            for raw_batch in loader:
                batch = self._move(raw_batch)
                outputs = self.model(batch["candidate_features"], batch["candidate_mask"])
                loss, stats = self._loss(
                    outputs["winner_logits"],
                    batch["target_mask"],
                    batch["candidate_mask"],
                    proposal_scores=batch.get("proposal_scores"),
                    proposal_top1_index=batch.get("proposal_top1_index"),
                    proposal_top1_is_true=batch.get("proposal_top1_is_true"),
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                history.append(stats)
            if not history:
                return {"candidate_set_total_loss": 0.0}
            keys = sorted({key for row in history for key in row.keys()})
            return {key: float(sum(float(row.get(key, 0.0)) for row in history) / len(history)) for key in keys}

        def evaluate(self, loader, *, split_summary: Optional[Dict[str, object]] = None) -> Dict[str, object]:
            self.model.eval()
            logits_rows = []
            target_rows = []
            mask_rows = []
            proposal_score_rows = []
            proposal_top1_index_rows = []
            proposal_top1_rows = []
            sources = []
            with torch.no_grad():
                for raw_batch in loader:
                    batch = self._move(raw_batch)
                    outputs = self.model(batch["candidate_features"], batch["candidate_mask"])
                    logits_rows.append(outputs["winner_logits"].detach().cpu())
                    target_rows.append(batch["target_mask"].detach().cpu())
                    mask_rows.append(batch["candidate_mask"].detach().cpu())
                    proposal_score_rows.append(batch["proposal_scores"].detach().cpu())
                    proposal_top1_index_rows.append(batch["proposal_top1_index"].detach().cpu())
                    proposal_top1_rows.append(batch["proposal_top1_is_true"].detach().cpu())
                    sources.extend(list(raw_batch.get("source") or []))
            if not logits_rows:
                return {}
            winner_logits = torch.cat(logits_rows, dim=0)
            target_mask = torch.cat(target_rows, dim=0)
            candidate_mask = torch.cat(mask_rows, dim=0)
            proposal_scores = torch.cat(proposal_score_rows, dim=0)
            proposal_top1_index = torch.cat(proposal_top1_index_rows, dim=0)
            proposal_top1_is_true = torch.cat(proposal_top1_rows, dim=0)
            metrics = compute_candidate_winner_metrics(
                winner_logits,
                target_mask,
                candidate_mask,
                proposal_top1_is_true=proposal_top1_is_true,
                total_molecule_count=int((split_summary or {}).get("total_molecules", int(target_mask.shape[0]))),
                proposal_hit_count=int((split_summary or {}).get("proposal_hit_molecules", int(target_mask.shape[0]))),
            )
            source_rows = defaultdict(lambda: {"n": 0, "winner_acc_given_proposal": 0.0})
            pred_idx = torch.argmax(winner_logits, dim=1)
            for idx, source in enumerate(sources):
                row = source_rows[str(source)]
                row["n"] += 1
                row["winner_acc_given_proposal"] += float(target_mask[idx, int(pred_idx[idx].item())].item() > 0.5)
            source_breakdown = {
                name: {
                    "n": int(row["n"]),
                    "winner_acc_given_proposal": float(row["winner_acc_given_proposal"]) / float(row["n"]) if row["n"] > 0 else 0.0,
                }
                for name, row in sorted(source_rows.items())
            }
            valid_logits = winner_logits.masked_fill(candidate_mask <= 0.5, float("-inf"))
            top2_vals = torch.topk(valid_logits, k=min(2, int(valid_logits.shape[1])), dim=1).values
            winner_top1_gap_mean = float(
                (top2_vals[:, 0] - top2_vals[:, 1]).mean().item()
                if int(top2_vals.shape[1]) >= 2
                else top2_vals[:, 0].mean().item()
            )
            valid_prop = proposal_scores.masked_fill(candidate_mask <= 0.5, float("-inf"))
            prop_top2_vals = torch.topk(valid_prop, k=min(2, int(valid_prop.shape[1])), dim=1).values
            proposal_top1_gap_mean = float(
                (prop_top2_vals[:, 0] - prop_top2_vals[:, 1]).mean().item()
                if int(prop_top2_vals.shape[1]) >= 2
                else prop_top2_vals[:, 0].mean().item()
            )
            winner_probs = torch.softmax(valid_logits, dim=1)
            winner_top1_prob_mean = float(torch.max(winner_probs, dim=1).values.mean().item())
            pred_idx = torch.argmax(winner_logits, dim=1)
            proposal_preserved = []
            proposal_harm_margin = []
            for idx in range(int(winner_logits.shape[0])):
                prop_true = float(proposal_top1_is_true[idx].item()) > 0.5
                if not prop_true:
                    continue
                prop_idx = int(proposal_top1_index[idx].item())
                proposal_preserved.append(float(int(pred_idx[idx].item()) == prop_idx))
                other_mask = candidate_mask[idx] > 0.5
                if prop_idx < 0 or prop_idx >= int(other_mask.shape[0]) or not bool(other_mask[prop_idx].item()):
                    continue
                if bool(other_mask.any().item()) and int(other_mask.sum().item()) > 1:
                    other_mask[prop_idx] = False
                    if bool(other_mask.any().item()):
                        proposal_harm_margin.append(float((winner_logits[idx, prop_idx] - winner_logits[idx, other_mask].max()).item()))
            return {
                **metrics,
                "winner_top1_gap_mean": winner_top1_gap_mean,
                "winner_top1_prob_mean": winner_top1_prob_mean,
                "proposal_top1_gap_mean": proposal_top1_gap_mean,
                "proposal_top1_preserved_fraction_on_correct_proposals": float(sum(proposal_preserved) / len(proposal_preserved)) if proposal_preserved else 0.0,
                "proposal_harm_margin_mean_on_correct_proposals": float(sum(proposal_harm_margin) / len(proposal_harm_margin)) if proposal_harm_margin else 0.0,
                "source_breakdown": source_breakdown,
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
    class CandidateWinnerTrainer:  # type: ignore[override]
        model: object

        def __post_init__(self):
            require_torch()
