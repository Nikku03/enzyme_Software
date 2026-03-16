from __future__ import annotations
from typing import Dict, List, Optional

from .types import CPTResult, MechanismProfile


FIDELITY_WEIGHT = {
    "geometric_basic": 0.80,
    "geometric_with_sterics": 0.90,
    "geometric_robust": 0.92,
}


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _consistency_label(pass_frac: float) -> str:
    if pass_frac >= 0.80:
        return "high"
    if pass_frac >= 0.55:
        return "medium"
    return "low"


class UnifiedMechanismScorer:
    """
    Not a naive sum:
      - supporting evidence boosts feasibility
      - failing evidence penalizes more strongly
      - confidence depends on agreement + fidelity weights
    """

    def __init__(self, fail_penalty: float = 1.35):
        self.fail_penalty = fail_penalty

    def score(self, mechanism_id: str, evidence: List[CPTResult]) -> MechanismProfile:
        if not evidence:
            return MechanismProfile(
                mechanism_id=mechanism_id,
                feasibility_score=-6.0,
                confidence=0.15,
                consistency="low",
                primary_constraint="no_evidence",
                key_insight="No CPT evidence available yet.",
                evidence=[],
            )

        weighted = []
        for r in evidence:
            fidelity = r.data.get("fidelity", "geometric_basic")
            fw = FIDELITY_WEIGHT.get(fidelity, 0.75)
            w = _clip(r.confidence * fw, 0.05, 1.0)
            weighted.append((r, w))

        support = [(r, w) for r, w in weighted if r.passed]
        conflict = [(r, w) for r, w in weighted if not r.passed]

        if support:
            sup_score = sum(r.score * w for r, w in support) / (sum(w for _, w in support) + 1e-9)
        else:
            sup_score = 0.0

        if conflict:
            conf_score = sum((1.0 - r.score) * w for r, w in conflict) / (sum(w for _, w in conflict) + 1e-9)
        else:
            conf_score = 0.0

        feasibility = (10.0 * sup_score) - (10.0 * self.fail_penalty * conf_score)
        feasibility = _clip(feasibility, -10.0, 10.0)

        pass_frac = sum(1 for r in evidence if r.passed) / max(1, len(evidence))
        consistency = _consistency_label(pass_frac)

        avg_w = sum(w for _, w in weighted) / max(1, len(weighted))
        fatal = any((not r.passed) and r.data.get("fatal", False) for r in evidence)
        base_conf = 0.35 + 0.45 * avg_w + 0.20 * pass_frac
        if fatal:
            base_conf *= 0.65
        confidence = _clip(base_conf, 0.05, 0.99)

        primary_constraint = None
        if conflict:
            worst = max(conflict, key=lambda rw: rw[1] * (1.0 - rw[0].score))[0]
            primary_constraint = worst.cpt_id

        key_insight = self._insight(evidence, feasibility, primary_constraint)

        return MechanismProfile(
            mechanism_id=mechanism_id,
            feasibility_score=feasibility,
            confidence=confidence,
            consistency=consistency,
            primary_constraint=primary_constraint,
            key_insight=key_insight,
            evidence=evidence,
        )

    def _insight(self, evidence: List[CPTResult], feas: float, constraint: Optional[str]) -> str:
        if feas >= 5.0:
            return "Geometry is broadly favorable; proceed to higher-fidelity sterics and (next) electronics."
        if feas >= 0.0:
            return f"Borderline geometry; main bottleneck appears to be {constraint or 'one geometric constraint'}."
        return f"Unfavorable geometry; strongest limiting factor: {constraint or 'multiple geometric failures'}."
