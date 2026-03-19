from __future__ import annotations

from typing import Dict

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:
    def compute_reranker_metrics(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        valid = batch["candidate_valid"]
        positive = batch["candidate_positive"]
        scores = outputs["reranked_candidate_scores"]
        base_scores = outputs["base_candidate_scores"]
        valid_rows = valid.any(dim=1) & positive.any(dim=1)
        if not bool(valid_rows.any()):
            return {
                "candidate_accuracy": 0.0,
                "hard_negative_win_rate": 0.0,
                "true_margin_mean": 0.0,
                "base_top1": 0.0,
                "reranked_top1": 0.0,
                "base_top3": 0.0,
                "reranked_top3": 0.0,
            }
        scores = scores[valid_rows]
        base_scores = base_scores[valid_rows]
        valid = valid[valid_rows]
        positive = positive[valid_rows]

        masked_scores = scores.masked_fill(~valid, float("-inf"))
        masked_base = base_scores.masked_fill(~valid, float("-inf"))
        pred = masked_scores.argmax(dim=1)
        pred_base = masked_base.argmax(dim=1)
        candidate_accuracy = positive.gather(1, pred.unsqueeze(1)).float().mean()
        base_top1 = positive.gather(1, pred_base.unsqueeze(1)).float().mean()

        topk = min(3, scores.size(1))
        top3_idx = masked_scores.topk(topk, dim=1).indices
        base_top3_idx = masked_base.topk(topk, dim=1).indices
        reranked_top3 = positive.gather(1, top3_idx).any(dim=1).float().mean()
        base_top3 = positive.gather(1, base_top3_idx).any(dim=1).float().mean()

        pos_scores = masked_scores.masked_fill(~positive, float("-inf")).max(dim=1).values
        neg_scores = masked_scores.masked_fill(~(valid & ~positive), float("-inf")).max(dim=1).values
        margin = pos_scores - neg_scores
        hard_negative_win_rate = (margin > 0).float().mean()
        reranked_top2 = masked_scores.topk(min(2, scores.size(1)), dim=1).values
        base_top2 = masked_base.topk(min(2, scores.size(1)), dim=1).values
        reranked_margin = (reranked_top2[:, 0] - reranked_top2[:, 1]).mean() if reranked_top2.size(1) > 1 else reranked_top2[:, 0].mean()
        base_margin = (base_top2[:, 0] - base_top2[:, 1]).mean() if base_top2.size(1) > 1 else base_top2[:, 0].mean()
        reranked_probs = torch.softmax(masked_scores.masked_fill(~valid, float("-inf")), dim=1)
        reranked_entropy = -(reranked_probs * torch.log(reranked_probs.clamp_min(1e-8))).sum(dim=1).mean()
        return {
            "candidate_accuracy": float(candidate_accuracy.item()),
            "hard_negative_win_rate": float(hard_negative_win_rate.item()),
            "true_margin_mean": float(margin.mean().item()),
            "base_top1_top2_margin": float(base_margin.item()),
            "reranked_top1_top2_margin": float(reranked_margin.item()),
            "reranked_entropy": float(reranked_entropy.item()),
            "base_top1": float(base_top1.item()),
            "reranked_top1": float(candidate_accuracy.item()),
            "base_top3": float(base_top3.item()),
            "reranked_top3": float(reranked_top3.item()),
        }
else:  # pragma: no cover
    def compute_reranker_metrics(*args, **kwargs):
        require_torch()
