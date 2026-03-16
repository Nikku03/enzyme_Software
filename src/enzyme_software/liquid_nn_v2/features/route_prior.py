from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES


ROUTE_TO_CYP = {
    "p450": ["CYP3A4", "CYP2D6", "CYP2C9", "CYP1A2"],
    "cyp_hydroxylation": ["CYP3A4", "CYP2D6", "CYP2C9"],
    "cyp_n_dealkylation": ["CYP3A4", "CYP2D6", "CYP1A2"],
    "cyp_o_dealkylation": ["CYP3A4", "CYP2C9", "CYP2C19"],
    "cyp_epoxidation": ["CYP3A4", "CYP1A2"],
    "cyp_oxidation": ["CYP2E1", "CYP1A2"],
    "monooxygenase": ["CYP3A4", "CYP1A2"],
    "amine_oxidase": ["CYP2D6", "CYP1A2"],
    "oxidoreductase": ["CYP1A2", "CYP3A4"],
}


def _normalize_route_posteriors(route_posteriors) -> Dict[str, float]:
    if route_posteriors is None:
        return {}
    if isinstance(route_posteriors, dict):
        return {str(key): float(value) for key, value in route_posteriors.items() if isinstance(value, (int, float))}
    normalized: Dict[str, float] = {}
    if isinstance(route_posteriors, Iterable):
        for item in route_posteriors:
            if not isinstance(item, dict):
                continue
            route = str(item.get("route") or item.get("route_id") or item.get("name") or "")
            value = item.get("posterior")
            if not isinstance(value, (int, float)):
                value = item.get("probability")
            if route and isinstance(value, (int, float)):
                normalized[route] = float(value)
    return normalized


if TORCH_AVAILABLE:
    def route_posteriors_to_cyp_prior(
        route_posteriors,
        cyp_order: Sequence[str] | None = None,
    ) -> torch.Tensor:
        cyp_order = tuple(cyp_order or MAJOR_CYP_CLASSES)
        route_probs = _normalize_route_posteriors(route_posteriors)
        cyp_scores = {cyp: 0.0 for cyp in cyp_order}

        for route, prob in route_probs.items():
            cyps = ROUTE_TO_CYP.get(str(route).lower(), [])
            matched = [cyp for cyp in cyps if cyp in cyp_scores]
            if not matched:
                continue
            share = float(prob) / float(len(matched))
            for cyp in matched:
                cyp_scores[cyp] += share

        total = sum(cyp_scores.values())
        if total <= 0.0:
            uniform = 1.0 / max(1, len(cyp_order))
            cyp_scores = {cyp: uniform for cyp in cyp_order}
        else:
            cyp_scores = {cyp: value / total for cyp, value in cyp_scores.items()}

        return torch.tensor([cyp_scores[cyp] for cyp in cyp_order], dtype=torch.float32)


    def combine_lnn_with_prior(
        lnn_logits: torch.Tensor,
        prior: torch.Tensor,
        prior_weight: float = 0.3,
    ) -> torch.Tensor:
        prior_weight = min(max(float(prior_weight), 0.0), 1.0)
        lnn_probs = torch.softmax(lnn_logits, dim=-1)

        if prior.dim() == 1:
            prior = prior.unsqueeze(0).expand_as(lnn_probs)
        prior = prior.to(device=lnn_logits.device, dtype=lnn_logits.dtype)
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=1.0e-8)

        log_lnn = torch.log(lnn_probs.clamp(min=1.0e-8))
        log_prior = torch.log(prior.clamp(min=1.0e-8))
        return (1.0 - prior_weight) * log_lnn + prior_weight * log_prior
else:  # pragma: no cover
    def route_posteriors_to_cyp_prior(*args, **kwargs):
        require_torch()

    def combine_lnn_with_prior(*args, **kwargs):
        require_torch()
