from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from enzyme_software.mechanism_registry import MechanismSpec


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class ExpertVote:
    name: str
    score: float
    confidence: float
    reasons: List[str]
    required_inputs_missing: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": round(float(self.score), 4),
            "confidence": round(float(self.confidence), 4),
            "reasons": self.reasons,
            "required_inputs_missing": self.required_inputs_missing,
        }


def _extract_prior(shared_state: Dict[str, Any], route_id: str) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    physics = (shared_state.get("physics") or {}).get("priors") or {}
    prior = physics.get(route_id) or {}
    p_any = prior.get("route_prior_any_activity")
    if p_any is None:
        p_any = prior.get("prior_feasibility") or prior.get("prior_success_probability")
    if p_any is None:
        reasons.append("missing route prior; using fallback")
        return 0.5, reasons
    reasons.append("route prior from physics")
    return float(p_any), reasons


def score_serine(shared_state: Dict[str, Any]) -> ExpertVote:
    score, reasons = _extract_prior(shared_state, "serine_hydrolase")
    mismatch = ((shared_state.get("mechanism") or {}).get("mismatch") or {}).get("penalty") or 0.0
    adjusted = _clamp01(float(score) * (1.0 - float(mismatch)))
    reasons.append("mechanism mismatch penalty applied")
    return ExpertVote("serine_hydrolase", adjusted, 0.8, reasons, [])


def score_metallo(shared_state: Dict[str, Any]) -> ExpertVote:
    score, reasons = _extract_prior(shared_state, "metallo_esterase")
    return ExpertVote("metallo_esterase", _clamp01(float(score)), 0.75, reasons, [])


def score_other(shared_state: Dict[str, Any]) -> ExpertVote:
    score = 0.4
    reasons = ["fallback other-route scorer"]
    return ExpertVote("other", score, 0.4, reasons, [])


def score_heuristic(shared_state: Dict[str, Any]) -> ExpertVote:
    ledger = ((shared_state.get("scoring") or {}).get("ledger") or {}).get("module0") or {}
    terms = ledger.get("terms") or []
    score = None
    for term in terms:
        if term.get("name") == "route_confidence":
            score = term.get("value")
            break
    if score is None:
        score = 0.5
    return ExpertVote("heuristic", _clamp01(float(score)), 0.5, ["module0 route confidence"], [])


class MoERouter:
    def __init__(self) -> None:
        self.expert_funcs = {
            "serine_hydrolase": score_serine,
            "metallo_esterase": score_metallo,
            "other": score_other,
            "heuristic": score_heuristic,
        }

    def evaluate(
        self,
        shared_state: Dict[str, Any],
        mechanism_spec: MechanismSpec,
        features: Dict[str, float],
        top_k: int = 2,
        fork_hypotheses: bool = False,
    ) -> Dict[str, Any]:
        candidates, criteria = self._select_experts(mechanism_spec, features, fork_hypotheses)
        expert_votes = [self.expert_funcs[name](shared_state) for name in candidates]
        weights = self._gate_weights(shared_state, expert_votes)
        final_score = 0.0
        for vote in expert_votes:
            final_score += float(weights.get(vote.name, 0.0)) * float(vote.score)
        ordered = sorted(expert_votes, key=lambda v: v.score, reverse=True)
        selected = ordered[: max(1, int(top_k))]
        summary = (
            f"Experts considered: {[v.name for v in expert_votes]}, "
            f"weights: {[round(weights.get(v.name, 0.0), 3) for v in expert_votes]}, "
            f"reason: {criteria}"
        )
        return {
            "experts": [vote.to_dict() for vote in expert_votes],
            "weights": {key: round(float(val), 4) for key, val in weights.items()},
            "decision": {
                "score": round(float(final_score), 4),
                "route_id": mechanism_spec.route_id,
                "selected_experts": [vote.name for vote in selected],
                "summary": summary,
            },
        }

    def _select_experts(
        self,
        mechanism_spec: MechanismSpec,
        features: Dict[str, float],
        fork_hypotheses: bool,
    ) -> Tuple[List[str], str]:
        route_id = mechanism_spec.route_id
        candidates: List[str] = []
        criteria: List[str] = [f"mechanism_contract={route_id}"]
        if route_id in {"serine_hydrolase", "metallo_esterase"}:
            candidates.append(route_id)
        else:
            candidates.append("other")
        if fork_hypotheses or mechanism_spec.mismatch_policy_default == "FORK_HYPOTHESES":
            if route_id != "serine_hydrolase":
                candidates.append("serine_hydrolase")
            if route_id != "metallo_esterase":
                candidates.append("metallo_esterase")
            criteria.append("fork_hypotheses")
        candidates.append("heuristic")
        return candidates, ", ".join(criteria)

    def _gate_weights(
        self,
        shared_state: Dict[str, Any],
        expert_votes: List[ExpertVote],
    ) -> Dict[str, float]:
        data_support = (
            (shared_state.get("scoring") or {})
            .get("ledger", {})
            .get("module0", {})
            .get("data_support", 0.0)
        )
        weights: Dict[str, float] = {}
        for vote in expert_votes:
            weight = 1.0
            if vote.required_inputs_missing:
                weight *= 0.7
            if data_support <= 0.0 and vote.name == "heuristic":
                weight *= 0.6
            weight *= float(vote.confidence)
            weights[vote.name] = weight
        total = sum(weights.values())
        if total <= 0.0:
            return {vote.name: 1.0 / len(expert_votes) for vote in expert_votes}
        return {name: value / total for name, value in weights.items()}
