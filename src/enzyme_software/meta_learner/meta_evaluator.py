from __future__ import annotations

from typing import Dict

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:
    def evaluate_meta_predictions(site_scores: torch.Tensor, site_labels: torch.Tensor, cyp_logits: torch.Tensor, cyp_label: int) -> Dict[str, float]:
        site_scores = site_scores.detach().cpu().view(-1)
        site_labels = site_labels.detach().cpu().view(-1)
        positives = torch.nonzero(site_labels > 0.5, as_tuple=False).view(-1).tolist()
        ranking = torch.argsort(site_scores, descending=True).tolist()
        top1 = float(bool(ranking and ranking[0] in positives)) if positives else 0.0
        top3 = float(bool(any(idx in positives for idx in ranking[:3]))) if positives else 0.0
        pred_cyp = int(torch.argmax(cyp_logits.detach().cpu()).item())
        return {
            "site_top1": top1,
            "site_top3": top3,
            "cyp_acc": float(pred_cyp == int(cyp_label)),
        }
else:  # pragma: no cover
    def evaluate_meta_predictions(*args, **kwargs):
        require_torch()
