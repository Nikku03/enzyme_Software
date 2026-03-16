from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from enzyme_software.domain import ConditionProfile, ExperimentRecord, ReactionTask
from enzyme_software.physicscore import c_to_k
from enzyme_software.mathcore.uncertainty import (
    ProbabilityCalibrator,
    beta_credible_interval,
)


DEFAULT_STATE_PATH = Path(__file__).resolve().parents[3] / "cache" / "bayes_router_state.json"

PH_BINS = [(0.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0), (10.0, 14.1)]
TEMP_BINS = [
    (0.0, 285.0),
    (285.0, 295.0),
    (295.0, 305.0),
    (305.0, 315.0),
    (315.0, 400.0),
]


@dataclass
class RoutePosterior:
    route: str
    posterior: float
    p_raw: float
    p_cal: float
    support: float
    confidence: str
    uncertainty: float
    bucket: Tuple[str, str, str, str]
    drivers: List[str]
    ci90: Tuple[float, float]
    n_eff: float
    evidence_strength: float


class BayesianDAGRouter:
    def __init__(
        self,
        state_path: Optional[Path] = None,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        calibrator: Optional[ProbabilityCalibrator] = None,
    ) -> None:
        self.state_path = state_path or DEFAULT_STATE_PATH
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.state = self._load_state()
        self.calibrator = calibrator or ProbabilityCalibrator()

    def predict(
        self,
        task: ReactionTask | Dict[str, Any],
        candidates: List[Any],
        conditions: ConditionProfile | Dict[str, Any],
        routes: List[str],
    ) -> Dict[str, Any]:
        if not routes:
            return {
                "route_posteriors": [],
                "chosen_route": None,
                "confidence_label": "Low",
                "data_support": 0.0,
                "uncertainty": 0.25,
                "explanation": [],
                "matched_bins": {},
                "router_empty": True,
            }

        condition_bin = self._condition_bin(conditions)
        substrate_bin = self._substrate_bin(task)
        catalyst_bin = self._catalyst_bin(candidates)

        route_posteriors: List[RoutePosterior] = []
        for route in routes:
            alpha, beta = self._bucket_alpha_beta(
                route, condition_bin, substrate_bin, catalyst_bin
            )
            p_raw = alpha / (alpha + beta)
            support = (alpha + beta) - self._prior_strength()
            evidence_strength = self._evidence_strength(support)
            n_eff = self._n_eff(support, evidence_strength)
            p_cal = self.calibrator.predict(p_raw)
            ci90 = beta_credible_interval(p_cal, n_eff)
            uncertainty = self._variance(alpha, beta)
            confidence_label = self._confidence_label(p_cal, support)
            drivers = self._explain_route(route, condition_bin, substrate_bin, catalyst_bin)
            route_posteriors.append(
                RoutePosterior(
                    route=route,
                    posterior=round(p_cal, 3),
                    p_raw=round(p_raw, 3),
                    p_cal=round(p_cal, 3),
                    support=round(support, 2),
                    confidence=confidence_label,
                    uncertainty=round(uncertainty, 4),
                    bucket=(route, condition_bin, substrate_bin, catalyst_bin),
                    drivers=drivers,
                    ci90=ci90,
                    n_eff=round(n_eff, 2),
                    evidence_strength=round(evidence_strength, 3),
                )
            )

        route_posteriors.sort(key=lambda item: (item.posterior, item.support), reverse=True)
        chosen = self.choose_route(route_posteriors)
        matched_bins = {
            "condition_bin": condition_bin,
            "substrate_bin": substrate_bin,
            "catalyst_family_bin": catalyst_bin,
        }
        return {
            "route_posteriors": [item.__dict__ for item in route_posteriors],
            "chosen_route": chosen.route if chosen else None,
            "confidence_label": chosen.confidence if chosen else "Low",
            "data_support": chosen.support if chosen else 0.0,
            "uncertainty": chosen.uncertainty if chosen else 0.25,
            "explanation": chosen.drivers if chosen else [],
            "matched_bins": matched_bins,
            "route_p_raw": chosen.p_raw if chosen else None,
            "route_p_cal": chosen.p_cal if chosen else None,
            "route_ci90": chosen.ci90 if chosen else None,
            "n_eff": chosen.n_eff if chosen else None,
            "evidence_strength": chosen.evidence_strength if chosen else None,
            "router_empty": self._is_empty(),
        }

    def choose_route(self, posteriors: List[RoutePosterior]) -> Optional[RoutePosterior]:
        if not posteriors:
            return None
        def score(item: RoutePosterior) -> float:
            risk_penalty = 0.5 * float(item.uncertainty)
            support_bonus = 0.002 * float(item.support)
            return float(item.posterior) - risk_penalty + support_bonus

        posteriors = sorted(
            posteriors,
            key=lambda item: (score(item), item.support, item.route),
            reverse=True,
        )
        return posteriors[0]

    @staticmethod
    def effective_sample_size(
        confidence: float,
        min_eff: float = 2.0,
        max_eff: float = 25.0,
    ) -> float:
        try:
            conf = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            conf = 0.5
        return min_eff + (max_eff - min_eff) * conf

    def update_from_records(self, records: Iterable[ExperimentRecord | Dict[str, Any]]) -> None:
        updated = False
        for record in records:
            if isinstance(record, dict):
                record_obj = self._record_from_dict(record)
            else:
                record_obj = record
            if record_obj is None:
                continue
            route = record_obj.route or "unknown"
            substrate_bin = record_obj.substrate_bin or "unknown"
            catalyst_bin = record_obj.catalyst_family or "unknown"
            condition_bin = self._condition_bin(record_obj.condition_profile)
            weight = float(record_obj.weight) if record_obj.weight is not None else 1.0

            alpha, beta = self._bucket_alpha_beta(
                route, condition_bin, substrate_bin, catalyst_bin
            )
            if record_obj.observed_success >= 0.5:
                alpha += weight
            else:
                beta += weight
            self._set_bucket(route, condition_bin, substrate_bin, catalyst_bin, alpha, beta)
            updated = True

        if updated:
            self._save_state()

    def observe_weighted(
        self,
        route: str,
        condition_bin: str,
        substrate_bin: str,
        catalyst_bin: str,
        success_weight: float,
        fail_weight: float,
    ) -> None:
        alpha, beta = self._bucket_alpha_beta(route, condition_bin, substrate_bin, catalyst_bin)
        alpha += max(0.0, float(success_weight))
        beta += max(0.0, float(fail_weight))
        self._set_bucket(route, condition_bin, substrate_bin, catalyst_bin, alpha, beta)
        self._save_state()

    def _record_from_dict(self, payload: Dict[str, Any]) -> Optional[ExperimentRecord]:
        try:
            condition = payload.get("condition_profile")
            if isinstance(condition, dict):
                condition_profile = ConditionProfile(**condition)
            elif isinstance(condition, ConditionProfile):
                condition_profile = condition
            else:
                condition_profile = ConditionProfile()
            return ExperimentRecord(
                reaction_task_fingerprint=payload.get("reaction_task_fingerprint", "unknown"),
                condition_profile=condition_profile,
                candidate_fingerprint=payload.get("candidate_fingerprint", "unknown"),
                observed_success=float(payload.get("observed_success", 0.0)),
                observed_rate_or_yield=payload.get("observed_rate_or_yield"),
                notes=payload.get("notes"),
                source_quality=float(payload.get("source_quality", 0.5)),
                route=payload.get("route"),
                substrate_bin=payload.get("substrate_bin"),
                catalyst_family=payload.get("catalyst_family"),
                metadata=payload.get("metadata"),
                weight=payload.get("weight"),
            )
        except Exception:
            return None

    def _bucket_alpha_beta(
        self,
        route: str,
        condition_bin: str,
        substrate_bin: str,
        catalyst_bin: str,
    ) -> Tuple[float, float]:
        key = self._key(route, condition_bin, substrate_bin, catalyst_bin)
        entry = self.state.get("buckets", {}).get(key)
        if not entry:
            return self.prior_alpha, self.prior_beta
        return float(entry.get("alpha", self.prior_alpha)), float(
            entry.get("beta", self.prior_beta)
        )

    def _set_bucket(
        self,
        route: str,
        condition_bin: str,
        substrate_bin: str,
        catalyst_bin: str,
        alpha: float,
        beta: float,
    ) -> None:
        key = self._key(route, condition_bin, substrate_bin, catalyst_bin)
        self.state.setdefault("buckets", {})[key] = {"alpha": alpha, "beta": beta}

    def _is_empty(self) -> bool:
        return not bool(self.state.get("buckets"))

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {
                "prior_alpha": self.prior_alpha,
                "prior_beta": self.prior_beta,
                "bins": {
                    "ph": PH_BINS,
                    "temperature_K": TEMP_BINS,
                },
                "buckets": {},
            }
        try:
            with self.state_path.open("r", encoding="utf-8") as handle:
                state = json.load(handle)
        except (json.JSONDecodeError, OSError):
            state = {}
        state.setdefault("prior_alpha", self.prior_alpha)
        state.setdefault("prior_beta", self.prior_beta)
        state.setdefault("bins", {"ph": PH_BINS, "temperature_K": TEMP_BINS})
        state.setdefault("buckets", {})
        self.prior_alpha = float(state.get("prior_alpha", self.prior_alpha))
        self.prior_beta = float(state.get("prior_beta", self.prior_beta))
        return state

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "bins": {
                "ph": PH_BINS,
                "temperature_K": TEMP_BINS,
            },
            "buckets": self.state.get("buckets", {}),
        }
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _condition_bin(self, conditions: ConditionProfile | Dict[str, Any]) -> str:
        if isinstance(conditions, dict):
            ph_value = conditions.get("pH")
            temp_k = conditions.get("temperature_K")
            if temp_k is None and conditions.get("temperature_C") is not None:
                temp_k = c_to_k(float(conditions.get("temperature_C")))
        else:
            ph_value = conditions.pH
            temp_k = conditions.temperature_K
            if temp_k is None and conditions.temperature_C is not None:
                temp_k = c_to_k(float(conditions.temperature_C))
        ph_bin = self._bin_value(ph_value, PH_BINS, prefix="ph")
        temp_bin = self._bin_value(temp_k, TEMP_BINS, prefix="temp")
        return f"{ph_bin}|{temp_bin}"

    def _substrate_bin(self, task: ReactionTask | Dict[str, Any]) -> str:
        if isinstance(task, ReactionTask):
            target = task.bond_to_break_or_form or ""
            hint = task.mechanism_hint or ""
        else:
            target = task.get("bond_to_break_or_form", "")
            hint = task.get("mechanism_hint", "")
        text = f"{target} {hint}".lower()
        mapping = {
            "ester": "ester",
            "amide": "amide",
            "phosphate": "phosphate",
            "halide": "aryl_halide",
            "c-h": "c-h",
            "c-n": "c-n",
            "c-o": "c-o",
            "c-s": "c-s",
            "aryl": "aryl",
            "cc": "c-c",
        }
        for key, label in mapping.items():
            if key in text:
                return label
        return "unknown"

    def _catalyst_bin(self, candidates: List[Any]) -> str:
        for candidate in candidates:
            if isinstance(candidate, dict):
                family = candidate.get("predicted_mechanism") or candidate.get("scaffold_id")
            else:
                family = getattr(candidate, "predicted_mechanism", None)
            if family:
                return str(family)
        return "unknown"

    def _bin_value(
        self,
        value: Optional[float],
        bins: List[Tuple[float, float]],
        prefix: str,
    ) -> str:
        if value is None:
            return f"{prefix}_unknown"
        for low, high in bins:
            if low <= value < high:
                return f"{prefix}_{int(low)}_{int(high)}"
        return f"{prefix}_unknown"

    def _prior_strength(self) -> float:
        return float(self.prior_alpha + self.prior_beta)

    def _variance(self, alpha: float, beta: float) -> float:
        total = alpha + beta
        return (alpha * beta) / (total * total * (total + 1.0))

    def _confidence_label(self, posterior: float, support: float) -> str:
        if support >= 25 and posterior >= 0.70:
            return "High"
        if support >= 10 and posterior >= 0.55:
            return "Medium"
        return "Low"

    def _evidence_strength(self, support: float) -> float:
        if support <= 0:
            return 0.0
        return min(1.0, 1.0 - math.exp(-support / 10.0))

    def _n_eff(self, support: float, evidence_strength: float) -> float:
        base = self._prior_strength()
        return max(2.0, base + support * evidence_strength)

    def _key(self, route: str, condition_bin: str, substrate_bin: str, catalyst_bin: str) -> str:
        return f"{route}|{condition_bin}|{substrate_bin}|{catalyst_bin}"

    def _bucket_success_failure(self, alpha: float, beta: float) -> Tuple[float, float]:
        success = max(0.0, alpha - self.prior_alpha)
        failure = max(0.0, beta - self.prior_beta)
        return success, failure

    def _aggregate_bucket(
        self,
        route: str,
        condition_bin: Optional[str],
        substrate_bin: Optional[str],
        catalyst_bin: Optional[str],
    ) -> Tuple[float, float]:
        total_success = 0.0
        total_failure = 0.0
        for key, entry in self.state.get("buckets", {}).items():
            parts = key.split("|")
            if len(parts) != 4:
                continue
            r_key, c_key, s_key, f_key = parts
            if route and r_key != route:
                continue
            if condition_bin and c_key != condition_bin:
                continue
            if substrate_bin and s_key != substrate_bin:
                continue
            if catalyst_bin and f_key != catalyst_bin:
                continue
            success, failure = self._bucket_success_failure(
                float(entry.get("alpha", self.prior_alpha)),
                float(entry.get("beta", self.prior_beta)),
            )
            total_success += success
            total_failure += failure
        alpha = self.prior_alpha + total_success
        beta = self.prior_beta + total_failure
        return alpha, beta

    def _explain_route(
        self,
        route: str,
        condition_bin: str,
        substrate_bin: str,
        catalyst_bin: str,
    ) -> List[str]:
        alpha, beta = self._bucket_alpha_beta(route, condition_bin, substrate_bin, catalyst_bin)
        posterior = alpha / (alpha + beta)
        drivers = []

        cond_alpha, cond_beta = self._aggregate_bucket(
            route, None, substrate_bin, catalyst_bin
        )
        cond_post = cond_alpha / (cond_alpha + cond_beta)
        substrate_alpha, substrate_beta = self._aggregate_bucket(
            route, condition_bin, None, catalyst_bin
        )
        substrate_post = substrate_alpha / (substrate_alpha + substrate_beta)
        catalyst_alpha, catalyst_beta = self._aggregate_bucket(
            route, condition_bin, substrate_bin, None
        )
        catalyst_post = catalyst_alpha / (catalyst_alpha + catalyst_beta)

        deltas = {
            "conditions": posterior - cond_post,
            "substrate": posterior - substrate_post,
            "catalyst_family": posterior - catalyst_post,
        }
        for key, delta in sorted(deltas.items(), key=lambda item: abs(item[1]), reverse=True):
            drivers.append(f"{key} Δ={delta:+.3f}")
        return drivers[:5]
