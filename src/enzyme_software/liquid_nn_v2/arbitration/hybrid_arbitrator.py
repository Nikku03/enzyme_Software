from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HybridPrediction:
    site_atoms: List[int]
    site_scores: List[float]
    site_source: str
    cyp: str
    cyp_confidence: float
    cyp_source: str
    route: str
    route_confidence: float
    agreement: bool
    arbitration_reason: str


class HybridArbitrator:
    """Confidence-weighted arbitration between manual engine and LNN outputs."""

    def __init__(self, lnn_weight: float = 0.6, manual_weight: float = 0.4, confidence_threshold: float = 0.7):
        self.lnn_weight = float(lnn_weight)
        self.manual_weight = float(manual_weight)
        self.confidence_threshold = float(confidence_threshold)

    def arbitrate(self, lnn_output: Dict, manual_output: Dict) -> HybridPrediction:
        lnn_sites = list(lnn_output.get("site_atoms") or [])
        lnn_scores = [float(v) for v in (lnn_output.get("site_scores") or [])]
        lnn_cyp = str(lnn_output.get("cyp") or "unknown")
        lnn_cyp_conf = float(lnn_output.get("cyp_confidence") or 0.0)

        manual_sites = [int(v) for v in (manual_output.get("predicted_sites") or [])]
        manual_route = str(manual_output.get("route") or "unknown")
        manual_route_conf = float(manual_output.get("route_confidence") or 0.0)
        manual_cyp = str(manual_output.get("primary_cyp") or self._route_to_primary_cyp(manual_route))

        site_agreement = self._check_site_agreement(lnn_sites, manual_sites)
        cyp_agreement = lnn_cyp == manual_cyp

        if site_agreement:
            final_sites = lnn_sites
            site_source = "hybrid"
            reason = "Both agree on top-site region"
        elif lnn_cyp_conf >= self.confidence_threshold:
            final_sites = lnn_sites
            site_source = "lnn"
            reason = "LNN site retained due to high confidence"
        else:
            final_sites = manual_sites[:3] if manual_sites else lnn_sites
            site_source = "manual"
            reason = "Manual fallback due to lower LNN confidence"

        if cyp_agreement:
            final_cyp = lnn_cyp
            cyp_source = "hybrid"
        elif lnn_cyp_conf >= manual_route_conf:
            final_cyp = lnn_cyp
            cyp_source = "lnn"
        else:
            final_cyp = manual_cyp
            cyp_source = "manual"

        return HybridPrediction(
            site_atoms=final_sites,
            site_scores=lnn_scores,
            site_source=site_source,
            cyp=final_cyp,
            cyp_confidence=max(lnn_cyp_conf, manual_route_conf),
            cyp_source=cyp_source,
            route=manual_route,
            route_confidence=manual_route_conf,
            agreement=site_agreement and cyp_agreement,
            arbitration_reason=reason,
        )

    @staticmethod
    def _check_site_agreement(lnn_sites: List[int], manual_sites: List[int]) -> bool:
        if not lnn_sites or not manual_sites:
            return False
        return int(lnn_sites[0]) in {int(v) for v in manual_sites[:3]}

    @staticmethod
    def _route_to_primary_cyp(route: str) -> str:
        mapping = {
            "p450": "CYP3A4",
            "cyp_hydroxylation": "CYP3A4",
            "cyp_n_dealkylation": "CYP3A4",
            "cyp_o_dealkylation": "CYP2C9",
            "cyp_epoxidation": "CYP1A2",
            "monooxygenase": "CYP3A4",
            "amine_oxidase": "CYP2D6",
        }
        return mapping.get(str(route).lower(), "CYP3A4")
