from __future__ import annotations

from typing import Dict

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch


if TORCH_AVAILABLE:
    def evaluate_cahml_predictions(site_scores: torch.Tensor, site_labels: torch.Tensor, cyp_logits: torch.Tensor, cyp_label: int, reaction_logits: torch.Tensor | None = None, reaction_label: int | None = None) -> Dict[str, float]:
        site_scores = site_scores.detach().cpu().view(-1)
        site_labels = site_labels.detach().cpu().view(-1)
        positives = torch.nonzero(site_labels > 0.5, as_tuple=False).view(-1).tolist()
        ranking = torch.argsort(site_scores, descending=True).tolist()
        out = {
            "site_top1": float(bool(ranking and ranking[0] in positives)) if positives else 0.0,
            "site_top3": float(bool(any(idx in positives for idx in ranking[:3]))) if positives else 0.0,
            "cyp_acc": float(int(torch.argmax(cyp_logits.detach().cpu()).item()) == int(cyp_label)),
        }
        if reaction_logits is not None and reaction_label is not None and int(reaction_label) >= 0:
            out["reaction_acc"] = float(int(torch.argmax(reaction_logits.detach().cpu()).item()) == int(reaction_label))
        return out
else:  # pragma: no cover
    def evaluate_cahml_predictions(*args, **kwargs):
        require_torch()
