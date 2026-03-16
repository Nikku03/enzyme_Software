from __future__ import annotations

# Contract Notes (output contract freeze):
# - ctx.data["module1_topogate"] must preserve keys: status, halt_reason, cache_stats, mode, weights,
#   diffusion_cap_s_inv, pareto_front, predictor_model, condition_context, module1_confidence,
#   module1_physics_audit, evidence_record, predicted_under_given_conditions,
#   optimum_conditions_estimate, delta_from_optimum, confidence_calibrated, ranked_scaffolds,
#   rejected_scaffolds, module2_handoff, math_contract, warnings/errors.
# - shared_io updates via _merge_shared_io must keep shared_io["input"/"outputs"] intact.
# - New physics fields should live under module1_physics_audit or math_contract, not replace existing keys.
# - Access/Reach/Retention scoring occurs in _topogate_pass_a/_topogate_pass_b,
#   _reach_gate, and _retention_gate; weights derived in _determine_mode and compute_plan overrides.

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from enzyme_software.context import OperationalConstraints, PipelineContext
from enzyme_software.domain import (
    BondSpec,
    ConditionProfile,
    EvidenceRecord,
    FeatureVector,
    SharedInput,
    SharedIO,
    SharedOutput,
    SubstrateContext,
    TelemetryContext,
)
from enzyme_software.mathcore import (
    ProbabilityEstimate,
    DistributionEstimate,
    QCReport,
    compute_signature,
    record_event,
    score_signature,
    validate_math_contract,
)
from enzyme_software.mathcore.persistent_homology import topology_energy_component
from enzyme_software.mathcore.uncertainty import (
    weighted_mean,
    weighted_quantile,
    weighted_std,
)
from enzyme_software.chemcore import screening_factor
from enzyme_software.config import RETENTION_WEAK_THRESHOLD
from enzyme_software.physicscore import (
    R_J_per_molK,
    access_score_from_tunnel,
    boltzmann_weight,
    c_to_k,
    diffusion_cap_rate,
    format_rate,
    screened_coulomb_energy_kJ,
)
from enzyme_software.scorecard import (
    ScoreCard,
    ScoreCardMetric,
    contributors_from_features,
    metric_status,
)
from enzyme_software.score_ledger import ScoreLedger, ScoreTerm
from enzyme_software.unity_layer import record_interlink
from enzyme_software.modules.base import BaseModule
from enzyme_software.biocore import enzyme_family_prior

try:
    from enzyme_software.calibration.layer2_structure_db import get_family_structures
except Exception:  # pragma: no cover - optional integration
    def get_family_structures(enzyme_family: Optional[str]) -> List[Dict[str, Any]]:
        return []

MODULE1_VERSION = "v1.0"
CACHE_PATH = Path(__file__).resolve().parents[3] / "cache" / "module1_cache.json"
ENSEMBLE_SAMPLE_COUNT = 64
ENSEMBLE_TEMPERATURE = 0.15
ENSEMBLE_ENERGY_NOISE = 0.04
DEFAULT_ENERGY_WEIGHTS = {
    "access": 0.35,
    "reach": 0.45,
    "retention": 0.2,
    "topology": 0.25,
}
ENERGY_MODEL_SCALE_KJ_PER_MOL = 20.0
BOLTZMANN_CUTOFF_KJ_PER_MOL = 20.0
BOLTZMANN_TINY_WEIGHT = 1e-8
DIFFUSION_CAP_S_INV = 1.0e9
DEFAULT_DIFFUSION_COEFF_M2_S = 5.0e-10
PHYSICS_MULTIPLIER_MIN = 0.3
PHYSICS_CAP_FLOOR = 0.4
PHYSICS_ELECTROSTATICS_FLOOR = 0.4
PHYSICS_OCCUPANCY_FLOOR = 0.3
ASSUMED_INTERACTION_DISTANCE_A = 5.0
ASSUMED_COUNTER_CHARGE_E = 0.5
ASSUMED_ENCOUNTER_MOLARITY = 1.0e-3

FAIL_NO_POCKET = "FAIL_NO_POCKET"
FAIL_NO_TUNNEL_COARSE = "FAIL_NO_TUNNEL_COARSE"
FAIL_TUNNEL_TOO_NARROW = "FAIL_TUNNEL_TOO_NARROW"
FAIL_TUNNEL_COLLAPSE = "FAIL_TUNNEL_COLLAPSE"
FAIL_REACH_FAIL = "FAIL_REACH_FAIL"
FAIL_MECH_COMPAT = "FAIL_MECH_COMPAT"
FAIL_RETENTION_IMPOSSIBLE = "FAIL_RETENTION_IMPOSSIBLE"
FAIL_ALL_REJECTED = "FAIL_ALL_REJECTED"
PASS_TOPK_SELECTED = "PASS_TOPK_SELECTED"
WARN_RETENTION_WEAK_BINDING = "WARN_RETENTION_WEAK_BINDING"
WARN_TUNNEL_BORDERLINE = "WARN_TUNNEL_BORDERLINE"
FAIL_PHYSICS_ACCESS = "FAIL_PHYSICS_ACCESS"


@dataclass
class Scaffold:
    scaffold_id: str
    pdb_path: str
    track: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Module1TopoGate(BaseModule):
    name = "Module 1 - TopoGate + ReachGate + RetentionGate"

    def __init__(self, scaffold_library: Optional[List[Dict[str, Any]]] = None) -> None:
        self._scaffold_library = scaffold_library

    @staticmethod
    def _softmax_weights(
        energies: List[float],
        temperature: float = 1.0,
    ) -> List[float]:
        """
        Softmax weighting over heuristic scores.

        This is not a thermodynamic Boltzmann factor. For the physical form,
        use the physicscore helpers.
        """
        if not energies:
            return []
        beta = 1.0 / max(1e-9, float(temperature))
        min_energy = min(float(value) for value in energies)
        scaled = [
            math.exp(-beta * (float(value) - min_energy)) for value in energies
        ]
        total = sum(scaled) or 1.0
        return [value / total for value in scaled]

    @staticmethod
    def _ensemble_metrics(
        base_scores: Dict[str, float],
        n: int = 128,
        rng: Optional[random.Random] = None,
        energy_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        sample_count = max(4, int(n))
        rng = rng or random.Random(17)
        keys = sorted(base_scores.keys())
        if not keys:
            return {"mean": {}, "ci90": {}, "sample_count": sample_count}
        if energy_weights is None or len(energy_weights) != len(keys):
            energy_weights = [1.0 for _ in keys]

        samples: Dict[str, List[float]] = {key: [] for key in keys}
        energies: List[float] = []
        for _ in range(sample_count):
            energy = 0.0
            for idx, key in enumerate(keys):
                base = float(base_scores.get(key, 0.0) or 0.0)
                jitter = rng.uniform(-0.06, 0.06)
                value = max(0.0, min(1.0, base + jitter))
                samples[key].append(value)
                energy += float(energy_weights[idx]) * (1.0 - value)
            energies.append(energy)

        weights = (
            Module1TopoGate._softmax_weights(energies, temperature=0.6)
            if energy_weights is not None
            else [1.0 / sample_count for _ in range(sample_count)]
        )

        def _weighted_mean(values: List[float], weights: List[float]) -> float:
            total_weight = sum(weights)
            if total_weight <= 0.0:
                return sum(values) / len(values)
            return sum(val * weight for val, weight in zip(values, weights)) / total_weight

        def _weighted_quantile(
            values: List[float], weights: List[float], quantile: float
        ) -> float:
            pairs = sorted(zip(values, weights), key=lambda item: item[0])
            total_weight = sum(weight for _, weight in pairs)
            if total_weight <= 0.0:
                index = int(round(quantile * (len(values) - 1)))
                return pairs[index][0]
            cumulative = 0.0
            for value, weight in pairs:
                cumulative += weight
                if cumulative / total_weight >= quantile:
                    return value
            return pairs[-1][0]

        mean: Dict[str, float] = {}
        ci90: Dict[str, Tuple[float, float]] = {}
        for key in keys:
            values = samples[key]
            mean[key] = _weighted_mean(values, weights)
            lo = _weighted_quantile(values, weights, 0.05)
            hi = _weighted_quantile(values, weights, 0.95)
            ci90[key] = (lo, hi)

        return {
            "mean": mean,
            "ci90": ci90,
            "sample_count": sample_count,
            "weights": weights,
        }

    def run(self, ctx: PipelineContext) -> PipelineContext:
        job_card = ctx.data.get("job_card") or {}
        if not job_card:
            ctx.data["module1_topogate"] = {
                "handoff": {},
                "status": "FAIL",
                "halt_reason": "FAIL_MISSING_JOB_CARD",
            }
            return ctx
        shared = ctx.data.get("shared_io") or {}
        if _reaction_hash_mismatch(shared, job_card):
            job_card["pipeline_halt_reason"] = "HASH_MISMATCH"
            warnings = job_card.get("warnings") or []
            warnings.append("W_HASH_MISMATCH: reaction identity mismatch; halting module1.")
            job_card["warnings"] = list(dict.fromkeys(warnings))
            ctx.data["job_card"] = job_card
            result = _fail_handoff("FAIL_HASH_MISMATCH")
            ctx.data["module1_topogate"] = result
            ctx.data["shared_io"] = _merge_shared_io(ctx, result)
            _update_unity_record_parts(ctx, result)
            return ctx
        module0_job_card = (ctx.data.get("module0_strategy_router") or {}).get("job_card")
        if module0_job_card and module0_job_card != job_card:
            warnings = job_card.get("warnings") or []
            warnings.append(
                "W_JOB_CARD_MISMATCH: module0_strategy_router.job_card differs from data.job_card; using data.job_card."
            )
            job_card["warnings"] = list(dict.fromkeys(warnings))
            ctx.data["job_card"] = job_card

        unity_state = ctx.data.get("unity_state")
        result = run_module1(
            smiles=ctx.smiles,
            job_card=job_card,
            constraints=ctx.constraints,
            scaffold_library=self._scaffold_library or ctx.data.get("scaffold_library"),
            unity_state=unity_state if isinstance(unity_state, dict) else None,
        )
        energy_ledger = (ctx.data.get("shared_io") or {}).get("energy_ledger") or {}
        energy_update = _energy_ledger_update(result, energy_ledger)
        result["energy_ledger_update"] = energy_update
        module1_physics = result.get("module1_physics_audit")
        if isinstance(module1_physics, dict):
            module1_physics["energy_ledger_update"] = energy_update
            result["module1_physics_audit"] = module1_physics
        module1_confidence = result.get("module1_confidence") or {}
        total_confidence = module1_confidence.get("total")
        if isinstance(total_confidence, (int, float)):
            confidence = job_card.get("confidence") or {}
            feasibility = confidence.get("feasibility_if_specified")
            if not isinstance(feasibility, (int, float)):
                feasibility = confidence.get("route", 0.0)
            completeness = confidence.get("completeness")
            if not isinstance(completeness, (int, float)):
                completeness = 1.0
            adjusted = min(float(feasibility), 0.6 + 0.4 * float(total_confidence))
            confidence["feasibility_if_specified"] = round(adjusted, 3)
            confidence["route"] = round(adjusted * float(completeness), 3)
            confidence["module1_total"] = round(float(total_confidence), 3)
            job_card["confidence"] = confidence
            ctx.data["job_card"] = job_card

        feedback = _module1_bidirectional_feedback(job_card, result)
        if feedback:
            job_card.setdefault("bidirectional_feedback", {})["module1"] = feedback
            penalty = feedback.get("penalty") or 0.0
            if isinstance(penalty, (int, float)) and penalty > 0.0:
                confidence = job_card.get("confidence") or {}
                for key in ("route", "feasibility_if_specified"):
                    value = confidence.get(key)
                    if isinstance(value, (int, float)):
                        adjusted = max(0.0, min(1.0, float(value) * (1.0 - float(penalty))))
                        confidence[key] = round(adjusted, 3)
                confidence["bidirectional_penalty"] = round(float(penalty), 3)
                job_card["confidence"] = confidence
            if feedback.get("hard_veto"):
                warnings = job_card.get("warnings") or []
                warnings.append(
                    "W_MODULE1_VETO: downstream physics/geometry indicates reroute or review."
                )
                job_card["warnings"] = list(dict.fromkeys(warnings))
            ctx.data["job_card"] = job_card

        ctx.data["module1_topogate"] = result
        ctx.data["shared_io"] = _merge_shared_io(ctx, result)
        _update_unity_record_parts(ctx, result)
        return ctx


def _update_unity_record_parts(ctx: PipelineContext, module_output: Dict[str, Any]) -> None:
    parts = ctx.data.setdefault("unity_record_parts", {})
    parts["module1"] = {"module_output": module_output}


def _module1_bidirectional_feedback(
    job_card: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    reasons: List[str] = []
    penalty = 0.0
    hard_veto = False
    status = result.get("status")
    if status and status != "PASS":
        hard_veto = True
        reasons.append(f"module1_status_{status.lower()}")
        penalty += 0.2
    module1_confidence = result.get("module1_confidence") or {}
    total = module1_confidence.get("total")
    if isinstance(total, (int, float)) and float(total) < 0.25:
        penalty += 0.2
        reasons.append("low_module1_total")
    physics_multiplier = module1_confidence.get("physics_multiplier")
    if physics_multiplier is None:
        physics_multiplier = (result.get("module1_physics_audit") or {}).get(
            "physics_multiplier"
        )
    if isinstance(physics_multiplier, (int, float)) and float(physics_multiplier) < 0.6:
        penalty += 0.15
        reasons.append("low_physics_multiplier")
    retention_mean = (
        (module1_confidence.get("ensemble") or {}).get("retention_mean")
    )
    if isinstance(retention_mean, (int, float)) and float(retention_mean) < 0.35:
        penalty += 0.1
        reasons.append("retention_mean_below_threshold")
    penalty = min(0.5, float(penalty))
    return {
        "hard_veto": hard_veto,
        "penalty": round(float(penalty), 3),
        "reasons": reasons,
        "recommendation": "review_required" if hard_veto else "monitor",
    }


def run_module1(
    smiles: str,
    job_card: Dict[str, Any],
    constraints: OperationalConstraints,
    scaffold_library: Optional[List[Dict[str, Any]]] = None,
    unity_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    compute_plan = job_card.get("compute_plan") or {}
    route = job_card.get("mechanism_route") or {}
    bond_context = job_card.get("bond_context") or {}
    structure_summary = job_card.get("structure_summary") or {}
    resolved = job_card.get("resolved_target") or {}
    difficulty = job_card.get("difficulty_label") or job_card.get("difficulty") or "MEDIUM"
    size_proxies = job_card.get("substrate_size_proxies") or {}
    radius_assumptions: List[str] = []
    radius_A = _estimate_substrate_radius_A(size_proxies, structure_summary, radius_assumptions)

    if not resolved.get("selected_bond") and not resolved.get("atom_indices"):
        return _fail_handoff("FAIL_MISSING_TARGET")

    scaffold_count = int(compute_plan.get("scaffold_count") or 0)
    if scaffold_count <= 0:
        return _fail_handoff("FAIL_NO_SCAFFOLD_TARGET")

    strictness = compute_plan.get("topogate_strictness") or "standard"
    mode, weights = _determine_mode(bond_context, structure_summary)
    mode_override = job_card.get("module1_mode")
    weights_override = job_card.get("module1_weights")
    if mode_override and _valid_weights(weights_override):
        mode = mode_override
        weights = weights_override
    energy_weights = _resolve_energy_weights(
        weights,
        job_card.get("module1_energy_weights"),
    )
    ensemble_temperature = _resolve_ensemble_temperature(
        job_card.get("module1_ensemble_temperature")
    )
    target_role = bond_context.get("primary_role") or bond_context.get("bond_role") or "unknown"
    condition_context = _condition_context(job_card, constraints, unity_state)
    temp_k = _temperature_k_from_context(condition_context)
    if not job_card.get("module1_ensemble_temperature"):
        ensemble_temperature = temp_k
    unity_state = unity_state or {}
    unity_physics = unity_state.get("physics") or {}
    unity_physics_audit = unity_physics.get("audit") or {}
    unity_diffusion_cap = (
        unity_physics_audit.get("k_diff_cap_s_inv")
        or unity_physics_audit.get("diffusion_cap_s_inv")
        or unity_physics_audit.get("k_cap_s_inv")
    )

    scaffolds = _select_scaffolds(
        scaffold_library=scaffold_library,
        scaffold_library_id=job_card.get("scaffold_library_id"),
        route=route,
        scaffold_count=scaffold_count,
        difficulty=difficulty,
    )
    if not scaffolds:
        return _fail_handoff("FAIL_NO_SCAFFOLDS")

    cache = _load_cache()
    cache_hits = 0
    cache_misses = 0
    cache_writes = 0

    pass_a_results: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for scaffold in scaffolds:
        cache_key = _cache_key(
            scaffold_id=scaffold.scaffold_id,
            smiles=smiles,
            target_role=target_role,
            mode=mode,
            strictness=strictness,
        )
        cached = cache.get(cache_key)
        if cached:
            cache_hits += 1
            scaffold_state = dict(cached)
            if not scaffold_state.get("pocket_center_candidates"):
                scaffold_state["pocket_center_candidates"] = _detect_pocket_candidates(
                    scaffold_state
                )
            if not scaffold_state.get("pocket_center_candidates"):
                rejected.append(
                    _reject_scaffold(scaffold, [FAIL_NO_POCKET], scaffold_state)
                )
                continue
            if "access_score" not in scaffold_state or "tunnel_summary" not in scaffold_state:
                coarse = _topogate_coarse(
                    scaffold_state,
                    mode=mode,
                    strictness=strictness,
                    structure_summary=structure_summary,
                )
                if not coarse["pass"]:
                    rejected.append(
                        _reject_scaffold(
                            scaffold,
                            [FAIL_NO_TUNNEL_COARSE],
                            scaffold_state,
                            tunnel_summary=coarse["tunnel_summary"],
                        )
                    )
                    continue
                scaffold_state.update(coarse)
                scaffold_state["candidate_residues"] = _seed_reach_candidates(scaffold_state)
            if "candidate_residues" not in scaffold_state:
                scaffold_state["candidate_residues"] = _seed_reach_candidates(scaffold_state)
                cache[cache_key] = scaffold_state
                cache_writes += 1
        else:
            cache_misses += 1
            scaffold_state = _initialize_scaffold_state(scaffold)
            pocket_candidates = _detect_pocket_candidates(scaffold_state)
            if not pocket_candidates:
                rejected.append(
                    _reject_scaffold(scaffold, [FAIL_NO_POCKET], scaffold_state)
                )
                continue
            scaffold_state["pocket_center_candidates"] = pocket_candidates
            coarse = _topogate_coarse(
                scaffold_state,
                mode=mode,
                strictness=strictness,
                structure_summary=structure_summary,
            )
            if not coarse["pass"]:
                rejected.append(
                    _reject_scaffold(
                        scaffold,
                        [FAIL_NO_TUNNEL_COARSE],
                        scaffold_state,
                        tunnel_summary=coarse["tunnel_summary"],
                    )
                )
                continue
            scaffold_state.update(coarse)
            scaffold_state["candidate_residues"] = _seed_reach_candidates(scaffold_state)
            cache[cache_key] = scaffold_state
            cache_writes += 1

        pass_a_results.append(
            {
                "scaffold": scaffold,
                "state": scaffold_state,
            }
        )

    if not pass_a_results:
        _save_cache(cache)
        return _fail_handoff(
            FAIL_ALL_REJECTED,
            rejected=rejected,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_writes=cache_writes,
            cache_size=len(cache),
        )

    pass_a_results.sort(key=lambda item: item["state"].get("access_score", 0.0), reverse=True)
    pass_b_candidates = pass_a_results[: min(20, len(pass_a_results))]

    final_scaffolds: List[Dict[str, Any]] = []
    for entry in pass_b_candidates:
        scaffold = entry["scaffold"]
        state = entry["state"]
        refined = _topogate_refined(
            state,
            mode=mode,
            strictness=strictness,
            structure_summary=structure_summary,
            size_proxies=size_proxies,
        )
        if refined.get("fail_code"):
            rejected.append(
                _reject_scaffold(
                    scaffold,
                    [refined["fail_code"]],
                    state,
                    tunnel_summary=refined.get("tunnel_summary"),
                )
            )
            continue
        warn_codes: List[str] = []
        if refined.get("warn_code"):
            warn_codes.append(refined["warn_code"])

        flexibility = _flexibility_check(state, mode=mode)
        if flexibility["fail_code"]:
            rejected.append(
                _reject_scaffold(
                    scaffold,
                    [flexibility["fail_code"]],
                    state,
                    tunnel_summary=refined.get("tunnel_summary"),
                )
            )
            continue

        attack_envelope = _build_attack_envelope(
            state,
            route=route,
            strictness=strictness,
        )
        reach_summary = _reach_gate(
            state,
            attack_envelope=attack_envelope,
            route=route,
        )
        if reach_summary["fail_code"] and reach_summary["fail_code"] != FAIL_MECH_COMPAT:
            rejected.append(
                _reject_scaffold(
                    scaffold,
                    [reach_summary["fail_code"]],
                    state,
                    tunnel_summary=refined.get("tunnel_summary"),
                )
            )
            continue

        retention_metrics = _retention_gate(
            state,
            structure_summary=structure_summary,
            route=route,
            mode=mode,
            size_proxies=size_proxies,
        )
        if retention_metrics["fail_code"]:
            rejected.append(
                _reject_scaffold(
                    scaffold,
                    [retention_metrics["fail_code"]],
                    state,
                    tunnel_summary=refined.get("tunnel_summary"),
                )
            )
            continue

        tunnel_summary = refined.get("tunnel_summary") or {}
        diffusion_assumptions: List[str] = []
        diffusion_info = _diffusion_cap_factor(
            radius_A,
            temp_k,
            diffusion_assumptions,
            tunnel_summary.get("path_length"),
            condition_profile={"temperature_K": temp_k},
            substrate_context={"approx_radius": radius_A, "substrate_size_proxies": size_proxies},
            external_cap_s_inv=unity_diffusion_cap,
        )

        access_score = refined["access_score"]
        reach_score = reach_summary["reach_score"]
        reach_cap_factor = diffusion_info.get("cap_factor")
        if isinstance(reach_cap_factor, (int, float)):
            reach_score = min(reach_score, reach_score * float(reach_cap_factor))
        retention_score = retention_metrics["retention_score"]
        physics_access = None
        physics_access_ok = None
        physics_access_score = None
        substrate_radius_A = size_proxies.get("approx_radius")
        if substrate_radius_A is None and size_proxies.get("min_diameter_proxy") is not None:
            substrate_radius_A = float(size_proxies["min_diameter_proxy"]) / 2.0
        bottleneck = (refined.get("tunnel_summary") or {}).get("bottleneck_radius")
        if isinstance(substrate_radius_A, (int, float)) and isinstance(bottleneck, (int, float)):
            temp_k = _temperature_k_from_context(condition_context)
            physics_access = access_score_from_tunnel(
                float(substrate_radius_A),
                [float(bottleneck)],
                temp_k,
            )
            physics_access_score = physics_access.get("score")
            physics_access_ok = physics_access.get("ok")
            if isinstance(physics_access_score, (int, float)):
                access_score = min(access_score, float(physics_access_score))
        adjusted_scores, condition_penalties = _apply_condition_adjustment(
            access_score,
            reach_score,
            retention_score,
            condition_context,
        )
        base_total = (
            weights["access"] * access_score
            + weights["reach"] * reach_score
            + weights["retention"] * retention_score
        )
        total_score = (
            weights["access"] * adjusted_scores["access"]
            + weights["reach"] * adjusted_scores["reach"]
            + weights["retention"] * adjusted_scores["retention"]
        )
        retention_multiplier = retention_metrics.get("score_multiplier", 1.0)
        if isinstance(retention_multiplier, (int, float)):
            total_score *= retention_multiplier
        fail_codes = []
        if reach_summary.get("fail_code") == FAIL_MECH_COMPAT:
            fail_codes.append(FAIL_MECH_COMPAT)
        if retention_metrics.get("warning_codes"):
            warn_codes.extend(retention_metrics["warning_codes"])
        if warn_codes:
            fail_codes.extend(warn_codes)
        if physics_access_ok == 0.0:
            fail_codes.append(FAIL_PHYSICS_ACCESS)

        access_confidence = _access_confidence(
            refined.get("tunnel_summary"), adjusted_scores["access"]
        )
        reach_confidence = (
            reach_summary.get("reach_geom_score", 0.0)
            * reach_summary.get("mechanism_compat", {}).get("score", 0.0)
        )
        retention_confidence = adjusted_scores["retention"]
        module1_total = (
            weights["access"] * access_confidence
            + weights["reach"] * reach_confidence
            + weights["retention"] * retention_confidence
        )
        if isinstance(retention_multiplier, (int, float)):
            module1_total *= retention_multiplier
        topology_signature = _topology_signature(
            refined.get("tunnel_summary"),
            refined.get("pocket_center"),
            mode,
        )
        topology_score = topology_signature.get("topology_score") or 0.0
        topology_robustness = topology_signature.get("robustness") or 0.0
        module1_total *= 0.7 + 0.3 * float(topology_score)
        ensemble_metrics = _ensemble_metrics(
            scaffold.scaffold_id,
            adjusted_scores["access"],
            adjusted_scores["reach"],
            adjusted_scores["retention"],
            topology_score,
            weights,
            energy_weights=energy_weights,
            temperature=ensemble_temperature,
            sample_count=ENSEMBLE_SAMPLE_COUNT,
        )
        predictor_score = _predictor_score(
            adjusted_scores["access"],
            adjusted_scores["reach"],
            adjusted_scores["retention"],
            reach_summary,
            retention_metrics,
        )

        final_scaffolds.append(
            {
                "scaffold_id": scaffold.scaffold_id,
                "pdb_path": scaffold.pdb_path,
                "pdb_id": state.get("pdb_id"),
                "enzyme_family": state.get("enzyme_family"),
                "scaffold_metadata": state.get("scaffold_metadata"),
                "pocket_center": refined["pocket_center"],
                "tunnel_summary": refined["tunnel_summary"],
                "reach_summary": reach_summary,
                "attack_envelope": attack_envelope,
                "retention_metrics": retention_metrics,
                "physics_access": physics_access,
                "topology_signature": topology_signature,
                "topology_feasibility_score": round(topology_score, 3),
                "topology_robustness": round(topology_robustness, 3),
                "feasibility_score": round(module1_total, 3),
                "feasibility_flag": _feasibility_flag(
                    module1_total,
                    fail_codes,
                    topology_score,
                ),
                "required_topology_constraints": _required_topology_constraints(
                    refined.get("tunnel_summary"),
                    flexibility,
                ),
                "condition_adjusted_scores": adjusted_scores,
                "condition_penalties": condition_penalties,
                "scores": {
                    "access_score": round(access_score, 3),
                    "reach_score": round(reach_score, 3),
                    "retention_score": round(retention_score, 3),
                    "raw_total": round(base_total, 3),
                    "condition_adjusted_total": round(total_score, 3),
                    "total": round(total_score, 3),
                },
                "module1_confidence": {
                    "access": round(access_confidence, 3),
                    "reach": round(reach_confidence, 3),
                    "retention": round(retention_confidence, 3),
                    "total": round(module1_total, 3),
                    "topology": round(topology_score, 3),
                    "robustness": round(topology_robustness, 3),
                    "ensemble": ensemble_metrics,
                },
                "module1_physics_audit": {
                    "temperature_K": round(float(temp_k), 2),
                    "deltaE_model_kJ_per_mol": ensemble_metrics.get("deltaE_model_kJ_per_mol"),
                    "boltzmann_weight_summary": ensemble_metrics.get("boltzmann_weight_summary"),
                    "boltzmann_audit": ensemble_metrics.get("boltzmann_audit"),
                    "occupancy": ensemble_metrics.get("occupancy"),
                    "diffusion_cap_s_inv": diffusion_info.get("k_cap_s_inv")
                    or diffusion_info.get("k_diff_s_inv"),
                    "diffusion_cap_display": diffusion_info.get("k_cap_display"),
                    "diffusion_D_m2_s": diffusion_info.get("D_m2_s"),
                    "diffusion_L_A": diffusion_info.get("L_A"),
                    "diffusion_tau_s": diffusion_info.get("tau_s"),
                    "diffusion_k_diff_s_inv": diffusion_info.get("k_diff_s_inv"),
                    "diffusion_assumptions": diffusion_info.get("assumptions", []),
                },
                "predictor_score": round(predictor_score, 3),
                "fail_codes": fail_codes,
            }
        )

    _save_cache(cache)

    if not final_scaffolds:
        return _fail_handoff(
            FAIL_ALL_REJECTED,
            rejected=rejected,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_writes=cache_writes,
            cache_size=len(cache),
        )

    top_k = _top_k_from_difficulty(difficulty, scaffold_count)
    pareto_front_ids = _pareto_front(final_scaffolds)
    for scaffold in final_scaffolds:
        scaffold["pareto_rank"] = 0 if scaffold["scaffold_id"] in pareto_front_ids else 1

    pareto_scaffolds = [
        scaffold for scaffold in final_scaffolds if scaffold["scaffold_id"] in pareto_front_ids
    ]
    non_pareto_scaffolds = [
        scaffold for scaffold in final_scaffolds if scaffold["scaffold_id"] not in pareto_front_ids
    ]
    pareto_scaffolds.sort(key=lambda item: item["scores"]["total"], reverse=True)
    non_pareto_scaffolds.sort(key=lambda item: item["scores"]["total"], reverse=True)
    ranked = (pareto_scaffolds + non_pareto_scaffolds)[:top_k]
    selected_ids = {scaffold["scaffold_id"] for scaffold in ranked}
    for scaffold in final_scaffolds:
        scaffold["selected_for_refinement"] = scaffold["scaffold_id"] in selected_ids
    for scaffold in ranked:
        scaffold["fail_codes"] = list(dict.fromkeys(scaffold["fail_codes"] + [PASS_TOPK_SELECTED]))

    retention_threshold = float(RETENTION_WEAK_THRESHOLD.value)
    retention_attention = False
    retention_mean_value = None
    weights_original = dict(weights)
    weights_adjusted = None
    if ranked:
        initial_best = max(
            ranked,
            key=lambda item: item.get("module1_confidence", {}).get("total", 0.0),
        )
        retention_mean_value = (
            initial_best.get("module1_confidence", {})
            .get("ensemble", {})
            .get("retention_mean")
        )
        if isinstance(retention_mean_value, (int, float)) and retention_mean_value < retention_threshold:
            retention_attention = True
            retention_weight = min(0.35, float(weights["retention"]) + 0.10)
            delta = retention_weight - float(weights["retention"])
            reach_weight = max(0.0, float(weights["reach"]) - delta)
            adjusted = {
                "access": float(weights["access"]),
                "reach": reach_weight,
                "retention": retention_weight,
            }
            total_weight = sum(adjusted.values())
            if total_weight > 0:
                adjusted = {key: val / total_weight for key, val in adjusted.items()}
            weights_adjusted = {key: round(val, 3) for key, val in adjusted.items()}
            for scaffold in ranked:
                conf = scaffold.get("module1_confidence") or {}
                access_conf = conf.get("access")
                reach_conf = conf.get("reach")
                retention_conf = conf.get("retention")
                if not all(isinstance(val, (int, float)) for val in [access_conf, reach_conf, retention_conf]):
                    continue
                total_conf = (
                    adjusted["access"] * float(access_conf)
                    + adjusted["reach"] * float(reach_conf)
                    + adjusted["retention"] * float(retention_conf)
                )
                retention_multiplier = scaffold.get("retention_metrics", {}).get(
                    "score_multiplier", 1.0
                )
                if isinstance(retention_multiplier, (int, float)):
                    total_conf *= float(retention_multiplier)
                topology_score = float(scaffold.get("topology_feasibility_score") or 0.0)
                total_conf *= 0.7 + 0.3 * topology_score
                conf["total"] = round(float(total_conf), 3)
                scaffold["module1_confidence"] = conf

    module2_candidates = [
        scaffold
        for scaffold in ranked
        if scaffold.get("reach_summary", {}).get("fail_code") != FAIL_MECH_COMPAT
        and (scaffold.get("reach_summary", {}).get("mechanism_compat", {}).get("score") or 0.0)
        > 0.0
    ]

    module2_handoff = {
        "top_scaffolds": [
            {
                "scaffold_id": scaffold["scaffold_id"],
                "pdb_path": scaffold["pdb_path"],
                "pdb_id": scaffold.get("pdb_id"),
                "enzyme_family": scaffold.get("enzyme_family"),
                "scaffold_metadata": scaffold.get("scaffold_metadata"),
                "attack_envelope": scaffold["attack_envelope"],
                "candidate_residues_by_role": scaffold["reach_summary"].get(
                    "candidate_residues_by_role", {}
                ),
                "tunnel_metrics": scaffold["tunnel_summary"],
                "scores": {"total": scaffold["scores"]["total"]},
                "topology_signature": scaffold.get("topology_signature"),
                "topology_feasibility_score": scaffold.get("topology_feasibility_score"),
                "topology_robustness": scaffold.get("topology_robustness"),
                "feasibility_flag": scaffold.get("feasibility_flag"),
                "feasibility_score": scaffold.get("feasibility_score"),
                "required_topology_constraints": scaffold.get("required_topology_constraints"),
                "condition_adjusted_scores": scaffold.get("condition_adjusted_scores"),
                "retention_metrics": {
                    "volume_ratio": scaffold["retention_metrics"].get("volume_ratio"),
                    "retention_risk_flag": scaffold["retention_metrics"].get(
                        "retention_risk_flag"
                    ),
                    "warning_codes": scaffold["retention_metrics"].get("warning_codes", []),
                    "score_multiplier": scaffold["retention_metrics"].get("score_multiplier"),
                },
                "reach_summary": {
                    "mechanism_compat_score": scaffold["reach_summary"]
                    .get("mechanism_compat", {})
                    .get("score"),
                    "nucleophile_type": scaffold["reach_summary"]
                    .get("mechanism_compat", {})
                    .get("nucleophile_type"),
                    "mechanism_label": scaffold["reach_summary"]
                    .get("mechanism_compat", {})
                    .get("mechanism_label"),
                    "required_flags": scaffold["reach_summary"]
                    .get("mechanism_compat", {})
                    .get("required_flags", []),
                    "nucleophile_geometry": _nucleophile_geometry(
                        scaffold["reach_summary"].get("mechanism_compat", {})
                    ),
                },
                "bond_center_hint": job_card.get("bond_center_hint") or {},
            }
            for scaffold in module2_candidates
        ]
    }

    module1_confidence = {}
    best_by_conf: Optional[Dict[str, Any]] = None
    if ranked:
        best_by_conf = max(
            ranked,
            key=lambda item: item.get("module1_confidence", {}).get("total", 0.0),
        )
        module1_confidence = dict(best_by_conf.get("module1_confidence", {}))
        if module1_confidence:
            module1_confidence["scaffold_id"] = best_by_conf.get("scaffold_id")
    path_length_A = None
    if best_by_conf:
        path_length_A = (best_by_conf.get("tunnel_summary") or {}).get("path_length")
    physics_block = _physics_multiplier(job_card, condition_context, path_length_A=path_length_A)
    if module1_confidence.get("total") is not None:
        raw_total = float(module1_confidence.get("total") or 0.0)
        module1_confidence["total_raw"] = round(raw_total, 3)
        module1_confidence["total"] = round(raw_total * float(physics_block["multiplier"]), 3)
        module1_confidence["physics_multiplier"] = physics_block["multiplier"]
    predicted_under_given = {
        "module1_total": module1_confidence.get("total"),
        "confidence_calibrated": module1_confidence.get("total"),
        "given_conditions": condition_context.get("given_conditions"),
    }
    module1_physics_audit = {}
    if best_by_conf:
        module1_physics_audit = dict(best_by_conf.get("module1_physics_audit") or {})
        if not module1_physics_audit.get("temperature_K"):
            module1_physics_audit["temperature_K"] = round(float(temp_k), 2)
        module1_physics_audit.setdefault(
            "diffusion_cap_s_inv",
            module1_physics_audit.get("diffusion_k_diff_s_inv")
            or module1_physics_audit.get("diffusion_cap_s_inv")
            or (physics_block.get("diffusion") or {}).get("k_cap_s_inv")
            or (physics_block.get("diffusion") or {}).get("k_diff_s_inv"),
        )
        module1_physics_audit.setdefault(
            "diffusion_cap_display",
            format_rate(module1_physics_audit.get("diffusion_cap_s_inv")),
        )
        module1_physics_audit.setdefault("electrostatics", physics_block.get("electrostatics"))
        module1_physics_audit.setdefault(
            "deltaE_electrostatic_kJ_per_mol",
            (physics_block.get("electrostatics") or {}).get("E_kJ_mol"),
        )
        module1_physics_audit.setdefault("physics_multiplier", physics_block.get("multiplier"))
        module1_physics_audit.setdefault("attention_flag", "Retention weak" if retention_attention else None)
    base_topology_score = 0.0
    if best_by_conf:
        base_topology_score = float(best_by_conf.get("topology_feasibility_score") or 0.0)
    topology_optimum = _optimum_conditions_for_topology(
        base_topology_score,
        condition_context.get("optimum_conditions_hint") or {},
    )
    optimum_estimate = {
        "strategy_hint": condition_context.get("optimum_conditions_hint") or {},
        "topology_optimum": topology_optimum,
    }
    delta_from_optimum = _delta_from_optimum(
        condition_context.get("given_conditions") or {},
        optimum_estimate,
    )
    evidence_features: Dict[str, float] = {}
    if best_by_conf:
        scores = best_by_conf.get("scores") or {}
        evidence_features = {
            "access_score": float(scores.get("access_score") or 0.0),
            "reach_score": float(scores.get("reach_score") or 0.0),
            "retention_score": float(scores.get("retention_score") or 0.0),
            "topology_score": float(best_by_conf.get("topology_feasibility_score") or 0.0),
            "module1_total": float(module1_confidence.get("total") or 0.0),
        }
    evidence_record = EvidenceRecord(
        module_id=1,
        inputs={
            "mode": mode,
            "conditions": condition_context.get("given_conditions"),
        },
        features_used=FeatureVector(values=evidence_features, missing=[], source="module1"),
        score=float(module1_confidence.get("total") or 0.0),
        confidence=float(module1_confidence.get("total") or 0.0),
        uncertainty={"ensemble": module1_confidence.get("ensemble")},
        optimum_conditions=condition_context.get("optimum_conditions_hint"),
        explanations=[],
        diagnostics={"model": "persistent_homology_v1"},
    ).to_dict()

    ensemble = module1_confidence.get("ensemble") or {}
    total_ci = ensemble.get("score_ci90", {}).get("total") or [0.0, 1.0]
    retention_ci = ensemble.get("score_ci90", {}).get("retention") or [0.0, 1.0]
    confidence_estimate = ProbabilityEstimate(
        p_raw=float(module1_confidence.get("total") or 0.0),
        p_cal=float(module1_confidence.get("total") or 0.0),
        ci90=(float(total_ci[0]), float(total_ci[1])),
        n_eff=float(ensemble.get("sample_count") or 2.0),
    ).to_dict()
    prediction_estimates = {
        "module1_total": DistributionEstimate(
            mean=float(ensemble.get("total_mean") or 0.0),
            std=float(ensemble.get("total_stdev") or 0.0),
            ci90=(float(total_ci[0]), float(total_ci[1])),
        ).to_dict(),
        "retention_score": DistributionEstimate(
            mean=float(ensemble.get("retention_mean") or 0.0),
            std=float(ensemble.get("retention_stdev") or 0.0),
            ci90=(float(retention_ci[0]), float(retention_ci[1])),
        ).to_dict(),
    }
    math_contract = {
        "confidence": confidence_estimate,
        "predictions": prediction_estimates,
        "qc": QCReport(status="N/A", reasons=[], metrics={}).to_dict(),
    }
    scorecard = _build_scorecard_module1(module1_confidence, evidence_record)
    score_ledger = _build_score_ledger_module1(
        module1_confidence=module1_confidence,
        weight_adjustment={
            "original": weights_original,
            "adjusted": weights_adjusted or weights_original,
        },
        module1_physics_audit=module1_physics_audit,
        retention_threshold=retention_threshold,
    )

    handoff = {
        "status": "PASS",
        "halt_reason": None,
        "cache_stats": {
            "hits": cache_hits,
            "misses": cache_misses,
            "entries_written": cache_writes,
            "cache_size": len(cache),
        },
        "mode": mode,
        "weights": weights,
        "weight_adjustment": {
            "original": weights_original,
            "adjusted": weights_adjusted or weights_original,
            "retention_mean": round(float(retention_mean_value), 3)
            if isinstance(retention_mean_value, (int, float))
            else None,
            "retention_threshold": retention_threshold,
            "attention_flag": "Retention weak" if retention_attention else None,
        },
        "diffusion_cap_s_inv": (
            physics_block.get("diffusion", {}).get("k_cap_s_inv")
            or physics_block.get("diffusion", {}).get("k_diff_s_inv")
            or DIFFUSION_CAP_S_INV
        ),
        "pareto_front": pareto_front_ids,
        "predictor_model": "heuristic_v1",
        "condition_context": condition_context,
        "module1_confidence": module1_confidence,
        "module1_physics_audit": module1_physics_audit,
        "physics": physics_block,
        "evidence_record": evidence_record,
        "predicted_under_given_conditions": predicted_under_given,
        "optimum_conditions_estimate": optimum_estimate,
        "delta_from_optimum": delta_from_optimum,
        "confidence_calibrated": module1_confidence.get("total"),
        "ranked_scaffolds": ranked,
        "rejected_scaffolds": rejected,
        "module2_handoff": module2_handoff,
        "math_contract": math_contract,
        "scorecard": scorecard,
        "score_ledger": score_ledger,
    }
    contract_violations = validate_math_contract(handoff)
    if contract_violations:
        handoff["warnings"] = list(
            dict.fromkeys((handoff.get("warnings") or []) + contract_violations)
        )
    record_event(
        {
            "module": "module1",
            "status": "PASS",
            "scaffold_count": len(ranked),
            "top_scaffold": module1_confidence.get("scaffold_id"),
            "module1_total": module1_confidence.get("total"),
            "given_conditions": condition_context.get("given_conditions"),
        }
    )
    return handoff


def _determine_mode(
    bond_context: Dict[str, Any],
    structure_summary: Dict[str, Any],
) -> Tuple[str, Dict[str, float]]:
    heavy_atoms = structure_summary.get("heavy_atoms") or 0
    rotatable = structure_summary.get("rotatable_bonds") or 0
    gas_flag = bond_context.get("is_gas_like_small_molecule") is True or heavy_atoms <= 6
    if gas_flag:
        return "small_gas", {"access": 0.15, "reach": 0.50, "retention": 0.35}
    if heavy_atoms >= 35 or rotatable >= 12:
        return "bulky_substrate", {"access": 0.45, "reach": 0.40, "retention": 0.15}
    return "standard", {"access": 0.35, "reach": 0.45, "retention": 0.20}


def _merge_shared_io(ctx: PipelineContext, module1_result: Dict[str, Any]) -> Dict[str, Any]:
    shared = ctx.data.get("shared_io")
    job_card = ctx.data.get("job_card") or {}
    if shared:
        shared_input = shared.get("input", {})
    else:
        bond_spec = BondSpec(
            target_bond=ctx.target_bond,
            selection_mode=(job_card.get("resolved_target") or {}).get("selection_mode"),
            resolved_target=job_card.get("resolved_target"),
            context=job_card.get("bond_context"),
        )
        substrate_context = SubstrateContext(
            smiles=ctx.smiles,
            structure_summary=job_card.get("structure_summary"),
        )
        telemetry = TelemetryContext(run_id="run_unknown", trace=["module1"])
        shared_input = SharedInput(
            bond_spec=bond_spec,
            condition_profile=ConditionProfile(**(job_card.get("condition_profile") or {})),
            substrate_context=substrate_context,
            telemetry=telemetry,
        ).to_dict()
    telemetry = shared_input.get("telemetry") or {}
    trace = telemetry.get("trace") or []
    if "module1" not in trace:
        trace.append("module1")
    telemetry["trace"] = trace
    shared_input["telemetry"] = telemetry

    module1_conf = module1_result.get("module1_confidence") or {}
    ranked = module1_result.get("ranked_scaffolds") or []
    best = ranked[0] if ranked else {}
    given_conditions = (module1_result.get("condition_context") or {}).get("given_conditions") or {}
    retry_suggestion = _retry_loop_suggestion(
        given_conditions,
        module1_result.get("optimum_conditions_estimate") or {},
    )
    output = SharedOutput(
        result={
            "status": module1_result.get("status"),
            "top_scaffold": module1_conf.get("scaffold_id"),
            "selected_scaffolds": [scaffold.get("scaffold_id") for scaffold in ranked],
        },
        given_conditions_effect={
            "module1_total": module1_conf.get("total"),
            "topology_score": best.get("topology_feasibility_score"),
            "robustness": best.get("topology_robustness"),
            "diffusion_cap_s_inv": module1_result.get("diffusion_cap_s_inv"),
        },
        optimum_conditions=module1_result.get("optimum_conditions_estimate") or {},
        confidence={
            "calibrated_probability": module1_conf.get("total"),
            "uncertainty": module1_conf.get("ensemble"),
        },
        retry_loop_suggestion=retry_suggestion,
    )
    outputs = dict(shared.get("outputs", {})) if shared else {}
    outputs["module1"] = output.to_dict()
    payload = dict(shared) if shared else {}
    payload["input"] = shared_input
    payload["outputs"] = outputs
    energy_ledger = payload.get("energy_ledger") or {}
    energy_update = module1_result.get("energy_ledger_update")
    if isinstance(energy_update, dict):
        energy_ledger = dict(energy_ledger)
        energy_ledger.update({key: value for key, value in energy_update.items() if value is not None})
    payload["energy_ledger"] = energy_ledger
    return payload


def _reaction_hash_mismatch(shared_io: Dict[str, Any], job_card: Dict[str, Any]) -> bool:
    shared_hash = (shared_io.get("input") or {}).get("reaction_identity_hash")
    job_hash = (job_card.get("reaction_identity") or {}).get("hash")
    return bool(shared_hash and job_hash and shared_hash != job_hash)


def _energy_ledger_update(
    module1_result: Dict[str, Any],
    shared_energy_ledger: Dict[str, Any],
) -> Dict[str, Any]:
    module1_conf = module1_result.get("module1_confidence") or {}
    retention = module1_conf.get("retention")
    if not isinstance(retention, (int, float)):
        retention = (module1_conf.get("ensemble") or {}).get("retention_mean")
    if not isinstance(retention, (int, float)):
        return dict(shared_energy_ledger or {})
    delta_g_bind_kj = round(-4.0 * float(retention), 3)
    merged = dict(shared_energy_ledger or {})
    merged["deltaG_bind_kJ"] = delta_g_bind_kj
    notes = list(merged.get("notes") or [])
    notes.append("module1 retention -> binding proxy")
    merged["notes"] = list(dict.fromkeys(notes))
    return merged


def _select_scaffolds(
    scaffold_library: Optional[List[Dict[str, Any]]],
    scaffold_library_id: Optional[str],
    route: Dict[str, Any],
    scaffold_count: int,
    difficulty: str,
) -> List[Scaffold]:
    if scaffold_library:
        scaffolds = []
        for entry in scaffold_library:
            if not isinstance(entry, dict):
                continue
            scaffold_id = entry.get("scaffold_id") or entry.get("id")
            pdb_path = entry.get("pdb_path") or entry.get("path")
            if scaffold_id and pdb_path:
                scaffolds.append(
                    Scaffold(
                        scaffold_id=scaffold_id,
                        pdb_path=pdb_path,
                        metadata=dict(entry),
                    )
                )
        return scaffolds[:scaffold_count]

    primary_route = str((route or {}).get("primary") or "")
    family = (enzyme_family_prior(primary_route) or {}).get("family")
    layer2_entries = get_family_structures(family)
    if layer2_entries:
        chosen: List[Scaffold] = []
        for entry in layer2_entries[: max(1, int(scaffold_count))]:
            pdb_id = str(entry.get("pdb_id") or "").strip().upper()
            if not pdb_id:
                continue
            chain = str(entry.get("chain") or "A").strip()
            scaffold_id = f"{family}_{pdb_id}_{chain}"
            pdb_path = f"scaffolds/pdb/{pdb_id}.pdb"
            meta = dict(entry)
            meta["enzyme_family"] = family
            meta["track"] = route.get("primary")
            chosen.append(
                Scaffold(
                    scaffold_id=scaffold_id,
                    pdb_path=pdb_path,
                    track=route.get("primary"),
                    metadata=meta,
                )
            )
        if chosen:
            if len(chosen) < scaffold_count:
                fallback = _generate_scaffolds(
                    scaffold_library_id or f"scaffold_lib_{family or 'generic'}_v1",
                    scaffold_count - len(chosen),
                )
                chosen.extend(fallback)
            return chosen[:scaffold_count]

    library_id = scaffold_library_id or "scaffold_lib_generic_v1"
    tracks = [track.get("track") for track in route.get("expert_tracks") or [] if track.get("track")]
    if difficulty == "HARD" and tracks:
        return _generate_scaffolds_with_tracks(library_id, tracks, scaffold_count)
    return _generate_scaffolds(library_id, scaffold_count)


def _generate_scaffolds(library_id: str, count: int) -> List[Scaffold]:
    scaffolds = []
    for idx in range(count):
        scaffold_id = f"{library_id}_scaffold_{idx + 1:03d}"
        pdb_path = f"scaffolds/{library_id}/{scaffold_id}.pdb"
        scaffolds.append(Scaffold(scaffold_id=scaffold_id, pdb_path=pdb_path))
    return scaffolds


def _generate_scaffolds_with_tracks(
    library_id: str,
    tracks: List[str],
    count: int,
) -> List[Scaffold]:
    scaffolds: List[Scaffold] = []
    track_count = max(1, min(len(tracks), 3))
    per_track = max(1, math.ceil(count / track_count))
    for track in tracks[:track_count]:
        for idx in range(per_track):
            if len(scaffolds) >= count:
                break
            scaffold_id = f"{library_id}_{track}_{idx + 1:03d}"
            pdb_path = f"scaffolds/{library_id}/{track}/{scaffold_id}.pdb"
            scaffolds.append(Scaffold(scaffold_id=scaffold_id, pdb_path=pdb_path, track=track))
    return scaffolds


def _cache_key(
    scaffold_id: str,
    smiles: str,
    target_role: str,
    mode: str,
    strictness: str,
) -> str:
    substrate_hash = hashlib.sha1(smiles.encode("utf-8")).hexdigest()
    raw = f"{scaffold_id}|{substrate_hash}|{target_role}|{mode}|{strictness}|{MODULE1_VERSION}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CACHE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2)
    try:
        tmp_path.replace(CACHE_PATH)
    except FileNotFoundError:
        # macOS path normalization/case aliasing can occasionally miss the temp inode.
        with CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2)


def _initialize_scaffold_state(scaffold: Scaffold) -> Dict[str, Any]:
    state = {
        "scaffold_id": scaffold.scaffold_id,
        "pdb_path": scaffold.pdb_path,
    }
    meta = scaffold.metadata if isinstance(scaffold.metadata, dict) else {}
    if meta:
        state["scaffold_metadata"] = meta
        state["pdb_id"] = meta.get("pdb_id")
        state["enzyme_family"] = meta.get("enzyme_family")
        pocket_metrics = meta.get("pocket_metrics") if isinstance(meta.get("pocket_metrics"), dict) else {}
        if pocket_metrics:
            state["layer2_pocket_metrics"] = dict(pocket_metrics)
        residue_seed: List[str] = []
        active_site = meta.get("active_site") if isinstance(meta.get("active_site"), dict) else {}
        for key in ("first_shell_residues", "access_channel_residues", "tunnel_residues", "halide_stabilizing"):
            items = active_site.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                resid = item.get("resid")
                resname = item.get("resname")
                if isinstance(resid, int) and isinstance(resname, str):
                    residue_seed.append(f"{resname.title()}{resid}")
        catalytic = active_site.get("catalytic_residues")
        if isinstance(catalytic, dict):
            for val in catalytic.values():
                if not isinstance(val, dict):
                    continue
                resid = val.get("resid")
                resname = val.get("resname")
                if isinstance(resid, int) and isinstance(resname, str):
                    residue_seed.append(f"{resname.title()}{resid}")
        if residue_seed:
            state["layer2_residue_seed"] = sorted(set(residue_seed))
    return state


def _detect_pocket_candidates(scaffold_state: Dict[str, Any]) -> List[List[float]]:
    if isinstance(scaffold_state.get("layer2_pocket_metrics"), dict):
        # Real PDB metadata available; use deterministic center proxy.
        return [[0.0, 0.0, 0.0]]
    rng = _rng_for_step(scaffold_state["scaffold_id"], "pocket")
    if rng.random() < 0.08:
        return []
    count = 1 + int(rng.random() * 2)
    centers = []
    for _ in range(count):
        centers.append(
            [
                round(rng.uniform(-5.0, 5.0), 3),
                round(rng.uniform(-5.0, 5.0), 3),
                round(rng.uniform(-5.0, 5.0), 3),
            ]
        )
    return centers


def _topogate_coarse(
    scaffold_state: Dict[str, Any],
    mode: str,
    strictness: str,
    structure_summary: Dict[str, Any],
) -> Dict[str, Any]:
    layer2_metrics = scaffold_state.get("layer2_pocket_metrics")
    if isinstance(layer2_metrics, dict):
        diameter_candidates = [
            layer2_metrics.get("channel_1_diameter_A"),
            layer2_metrics.get("channel_2_diameter_A"),
            layer2_metrics.get("tunnel_bottleneck_radius_A"),
        ]
        diameter_values = [float(v) for v in diameter_candidates if isinstance(v, (int, float))]
        if diameter_values:
            channel_diameter = min(diameter_values)
        else:
            channel_diameter = 3.0
        # convert diameter to radius when needed
        bottleneck = (
            channel_diameter if channel_diameter <= 2.2 else channel_diameter / 2.0
        )
        path_length = float(
            layer2_metrics.get("tunnel_length_A")
            or layer2_metrics.get("fe_to_channel_mouth_A")
            or layer2_metrics.get("asp_to_tunnel_mouth_A")
            or 12.0
        )
        probe_hint = _probe_radius(structure_summary, mode)
        access_score = max(0.0, min(1.0, (2.0 * float(bottleneck)) / max(0.5, float(probe_hint))))
        strict_penalty = {"lenient": 0.0, "standard": 0.05, "strict": 0.1}.get(strictness, 0.05)
        access_score = max(0.0, min(1.0, access_score - strict_penalty))
        tunnel_summary = {
            "bottleneck_radius": round(float(bottleneck), 3),
            "path_length": round(float(path_length), 3),
            "entry_point": [0.0, 0.0, 0.0],
            "curvature_proxy": 0.25,
            "source": "layer2_structure_db",
        }
        return {
            "pass": access_score >= (0.22 if mode == "small_gas" else 0.28),
            "access_score": access_score,
            "tunnel_summary": tunnel_summary,
            "pocket_center": scaffold_state["pocket_center_candidates"][0],
        }

    rng = _rng_for_step(scaffold_state["scaffold_id"], "topogate_coarse")
    base_access = {"small_gas": 0.85, "standard": 0.65, "bulky_substrate": 0.45}[mode]
    strict_penalty = {"lenient": 0.0, "standard": 0.05, "strict": 0.1}.get(strictness, 0.05)
    access_score = max(0.0, min(1.0, base_access - strict_penalty + rng.uniform(-0.1, 0.1)))
    pass_gate = rng.random() < access_score
    bottleneck = max(0.4, 1.0 + rng.uniform(-0.3, 0.5))
    path_length = max(5.0, 15.0 + rng.uniform(-5.0, 12.0))
    entry_point = [round(rng.uniform(-10.0, 10.0), 3) for _ in range(3)]
    tunnel_summary = {
        "bottleneck_radius": round(bottleneck, 3),
        "path_length": round(path_length, 3),
        "entry_point": entry_point,
        "curvature_proxy": round(rng.uniform(0.1, 0.6), 3),
    }
    return {
        "pass": pass_gate,
        "access_score": access_score,
        "tunnel_summary": tunnel_summary,
        "pocket_center": scaffold_state["pocket_center_candidates"][0],
    }


def _topogate_refined(
    scaffold_state: Dict[str, Any],
    mode: str,
    strictness: str,
    structure_summary: Dict[str, Any],
    size_proxies: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    layer2_metrics = scaffold_state.get("layer2_pocket_metrics")
    if isinstance(layer2_metrics, dict):
        probe_radius = _probe_radius(structure_summary, mode, size_proxies=size_proxies)
        clearance_margin = _clearance_margin(strictness)
        required_bottleneck = probe_radius + clearance_margin
        diameter_candidates = [
            layer2_metrics.get("channel_1_diameter_A"),
            layer2_metrics.get("channel_2_diameter_A"),
            layer2_metrics.get("tunnel_bottleneck_radius_A"),
        ]
        diameter_values = [float(v) for v in diameter_candidates if isinstance(v, (int, float))]
        if diameter_values:
            channel_diameter = min(diameter_values)
        else:
            channel_diameter = 3.0
        bottleneck = channel_diameter if channel_diameter <= 2.2 else channel_diameter / 2.0
        path_length = float(
            layer2_metrics.get("tunnel_length_A")
            or layer2_metrics.get("fe_to_channel_mouth_A")
            or layer2_metrics.get("asp_to_tunnel_mouth_A")
            or 12.0
        )
        access_score = min(1.0, float(bottleneck) / max(0.2, float(probe_radius)))
        access_score *= max(0.4, 1.0 - (float(path_length) / 40.0))
        access_score = max(0.0, min(1.0, access_score))
        tunnel_summary = {
            "bottleneck_radius": round(float(bottleneck), 3),
            "required_bottleneck_radius": round(float(required_bottleneck), 3),
            "path_length": round(float(path_length), 3),
            "curvature_proxy": 0.2,
            "entry_point": [0.0, 0.0, 0.0],
            "source": "layer2_structure_db",
        }
        if float(bottleneck) < float(required_bottleneck):
            shortfall = float(required_bottleneck) - float(bottleneck)
            if shortfall <= 0.1:
                tunnel_summary["borderline_clearance"] = True
                return {
                    "access_score": round(access_score * 0.9, 3),
                    "pocket_center": scaffold_state["pocket_center_candidates"][0],
                    "tunnel_summary": tunnel_summary,
                    "warn_code": WARN_TUNNEL_BORDERLINE,
                }
            return {
                "fail_code": FAIL_TUNNEL_TOO_NARROW,
                "access_score": access_score,
                "pocket_center": scaffold_state["pocket_center_candidates"][0],
                "tunnel_summary": tunnel_summary,
            }
        return {
            "access_score": round(access_score, 3),
            "pocket_center": scaffold_state["pocket_center_candidates"][0],
            "tunnel_summary": tunnel_summary,
        }

    rng = _rng_for_step(scaffold_state["scaffold_id"], "topogate_refined")
    probe_radius = _probe_radius(structure_summary, mode, size_proxies=size_proxies)
    clearance_margin = _clearance_margin(strictness)
    required_bottleneck = probe_radius + clearance_margin
    bottleneck = max(0.3, probe_radius * rng.uniform(0.7, 1.6))
    path_length = max(6.0, 12.0 + rng.uniform(-4.0, 10.0))
    curvature = round(rng.uniform(0.1, 0.7), 3)
    entry_point = [round(rng.uniform(-8.0, 8.0), 3) for _ in range(3)]
    access_score = min(1.0, bottleneck / max(0.2, probe_radius))
    access_score *= max(0.4, 1.0 - (path_length / 40.0))
    access_score = max(0.0, min(1.0, access_score))

    tunnel_summary = {
        "bottleneck_radius": round(bottleneck, 3),
        "required_bottleneck_radius": round(required_bottleneck, 3),
        "path_length": round(path_length, 3),
        "curvature_proxy": curvature,
        "entry_point": entry_point,
    }

    if bottleneck < required_bottleneck:
        shortfall = required_bottleneck - bottleneck
        if shortfall <= 0.1:
            tunnel_summary["borderline_clearance"] = True
            return {
                "access_score": round(access_score * 0.9, 3),
                "pocket_center": scaffold_state["pocket_center_candidates"][0],
                "tunnel_summary": tunnel_summary,
                "warn_code": WARN_TUNNEL_BORDERLINE,
            }
        return {
            "fail_code": FAIL_TUNNEL_TOO_NARROW,
            "access_score": access_score,
            "pocket_center": scaffold_state["pocket_center_candidates"][0],
            "tunnel_summary": tunnel_summary,
        }

    return {
        "access_score": round(access_score, 3),
        "pocket_center": scaffold_state["pocket_center_candidates"][0],
        "tunnel_summary": tunnel_summary,
    }


def _flexibility_check(scaffold_state: Dict[str, Any], mode: str) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "flexibility")
    open_fraction = rng.uniform(0.2, 0.9)
    threshold = 0.3 if mode != "small_gas" else 0.2
    if open_fraction < threshold:
        return {"fail_code": FAIL_TUNNEL_COLLAPSE, "open_fraction": round(open_fraction, 3)}
    return {"fail_code": None, "open_fraction": round(open_fraction, 3)}


def _build_attack_envelope(
    scaffold_state: Dict[str, Any],
    route: Dict[str, Any],
    strictness: str,
) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "attack_envelope")
    pocket_center = scaffold_state["pocket_center_candidates"][0]
    axis = _normalize_vector([rng.uniform(-1.0, 1.0) for _ in range(3)])
    apex = [round(pocket_center[i] + axis[i] * 1.5, 3) for i in range(3)]
    primary = route.get("primary") or "unknown"
    if primary in {"serine_hydrolase", "amidase", "hydrolase"}:
        distance_band = [2.5, 3.5]
    else:
        distance_band = [3.0, 5.0]
    cone_angle = 15 if strictness == "strict" else 30
    return {
        "apex_point": apex,
        "axis_vector": [round(val, 3) for val in axis],
        "distance_band": distance_band,
        "cone_angle": cone_angle,
        "mechanism_family": primary,
    }


def _reach_gate(
    scaffold_state: Dict[str, Any],
    attack_envelope: Dict[str, Any],
    route: Dict[str, Any],
) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "reach_gate")
    candidates = scaffold_state.get("candidate_residues") or []
    if not candidates:
        candidates = _seed_reach_candidates(scaffold_state)

    reach_candidates = []
    for residue in candidates:
        reach_score = max(0.0, min(1.0, rng.uniform(0.3, 0.95)))
        reach_candidates.append(
            {
                "residue": residue,
                "reach_score": round(reach_score, 3),
            }
        )

    if not reach_candidates:
        return {"fail_code": FAIL_REACH_FAIL, "reach_score": 0.0, "reachable_residues": []}

    reach_candidates.sort(key=lambda item: item["reach_score"], reverse=True)
    best = reach_candidates[:3]
    reach_score = sum(item["reach_score"] for item in best) / len(best)
    residue_names = [item["residue"] for item in reach_candidates]
    residues_by_role = _categorize_residues(residue_names)
    mechanism_compat = _mechanism_compatibility(route, residues_by_role)
    reach_geom_score = reach_score
    reach_score *= mechanism_compat["score"]

    fail_code = None
    primary = route.get("primary")
    if primary == "serine_hydrolase":
        if not (
            mechanism_compat["has_nucleophile"]
            and mechanism_compat["has_base"]
            and mechanism_compat["has_acid"]
        ):
            fail_code = FAIL_MECH_COMPAT
    return {
        "fail_code": fail_code,
        "reach_score": round(reach_score, 3),
        "reach_geom_score": round(reach_geom_score, 3),
        "reachable_residues": reach_candidates,
        "best_candidates": best,
        "candidate_residues_by_role": residues_by_role,
        "mechanism_compat": mechanism_compat,
    }


def _retention_gate(
    scaffold_state: Dict[str, Any],
    structure_summary: Dict[str, Any],
    route: Dict[str, Any],
    mode: str,
    size_proxies: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_state["scaffold_id"], "retention")
    heavy_atoms = structure_summary.get("heavy_atoms") or 1
    hetero_atoms = structure_summary.get("hetero_atoms") or 0
    ring_count = structure_summary.get("ring_count") or 0
    substrate_volume = max(50.0, heavy_atoms * 18.0)
    if size_proxies:
        proxy_volume = size_proxies.get("approx_volume")
        if isinstance(proxy_volume, (int, float)) and proxy_volume > 0:
            substrate_volume = float(proxy_volume)
        proxy_radius = size_proxies.get("approx_radius")
        if (
            proxy_volume is None
            and isinstance(proxy_radius, (int, float))
            and proxy_radius > 0
        ):
            substrate_volume = (4.0 / 3.0) * math.pi * (float(proxy_radius) ** 3)
    pocket_volume = max(120.0, 150.0 + rng.uniform(-40.0, 250.0))
    volume_ratio = pocket_volume / substrate_volume
    anchor_score = min(1.0, 0.1 + hetero_atoms * 0.05 + ring_count * 0.04)
    anchor_score = max(0.0, min(1.0, anchor_score))
    volume_penalty = 0.0
    if volume_ratio > 0:
        volume_penalty = _clamp01(
            1.0 - (math.log10(volume_ratio) / math.log10(100.0))
        )
    retention_score = (0.6 * anchor_score) + (0.4 * volume_penalty)

    risk_flag = "LOW"
    if volume_ratio >= 15:
        risk_flag = "MEDIUM"
    if volume_ratio >= 30 and anchor_score < 0.5:
        risk_flag = "HIGH"

    fail_code = None
    requires_alignment = route.get("primary") in {"serine_hydrolase", "amidase", "hydrolase"}
    if mode == "small_gas" and volume_ratio > 20 and anchor_score < 0.15 and requires_alignment:
        fail_code = FAIL_RETENTION_IMPOSSIBLE
    warning_codes = []
    score_multiplier = 1.0
    if volume_ratio >= 40 and anchor_score < 0.45:
        warning_codes.append(WARN_RETENTION_WEAK_BINDING)
        score_multiplier = 0.85

    return {
        "fail_code": fail_code,
        "retention_score": round(max(0.0, min(1.0, retention_score)), 3),
        "retention_risk_flag": risk_flag,
        "pocket_volume_proxy": round(pocket_volume, 2),
        "substrate_volume_proxy": round(substrate_volume, 2),
        "volume_ratio": round(volume_ratio, 2),
        "volume_penalty": round(volume_penalty, 3),
        "anchor_score": round(anchor_score, 3),
        "warning_codes": warning_codes,
        "score_multiplier": score_multiplier,
    }


def _seed_reach_candidates(scaffold_state: Dict[str, Any]) -> List[str]:
    if isinstance(scaffold_state.get("layer2_residue_seed"), list) and scaffold_state.get(
        "layer2_residue_seed"
    ):
        return list(scaffold_state.get("layer2_residue_seed"))
    rng = _rng_for_step(scaffold_state["scaffold_id"], "reach_seed")
    residues = ["Ser", "His", "Asp", "Glu", "Cys", "Lys", "Tyr", "Met"]
    count = max(1, int(rng.uniform(2, 6)))
    candidates = []
    for _ in range(count):
        residue = rng.choice(residues)
        idx = rng.randint(20, 220)
        candidates.append(f"{residue}{idx}")
    return candidates


def _categorize_residues(residues: List[str]) -> Dict[str, List[str]]:
    roles = {"nucleophile": [], "base": [], "acid": [], "other": []}
    nucleophiles = {"SER", "THR", "CYS"}
    bases = {"HIS"}
    acids = {"ASP", "GLU"}
    for residue in residues:
        match = re.match(r"[A-Za-z]+", residue or "")
        token = match.group(0).upper() if match else ""
        if token in nucleophiles:
            roles["nucleophile"].append(residue)
        elif token in bases:
            roles["base"].append(residue)
        elif token in acids:
            roles["acid"].append(residue)
        else:
            roles["other"].append(residue)
    return roles


def _extract_residue_token(residue: str) -> str:
    match = re.match(r"[A-Za-z]+", residue or "")
    return match.group(0).upper() if match else ""


def _mechanism_compatibility(
    route: Dict[str, Any],
    residues_by_role: Dict[str, List[str]],
) -> Dict[str, Any]:
    primary = route.get("primary") or "unknown"
    mechanism_label = primary
    required_flags: List[str] = []
    nucleophile_res = residues_by_role.get("nucleophile", [])
    base_res = residues_by_role.get("base", [])
    acid_res = residues_by_role.get("acid", [])
    nucleophile_tokens = {_extract_residue_token(res) for res in nucleophile_res}
    has_nucleophile = bool(nucleophile_res)
    has_base = bool(base_res)
    has_acid = bool(acid_res)
    nucleophile_type = "None"
    if nucleophile_tokens.intersection({"SER", "THR"}):
        nucleophile_type = "Ser/Thr"
    elif "CYS" in nucleophile_tokens:
        nucleophile_type = "Cys"
    elif has_nucleophile:
        nucleophile_type = "Other"

    notes: List[str] = []
    score = 0.0
    if primary == "serine_hydrolase":
        nucleophile_score = 1.0 if nucleophile_type == "Ser/Thr" else 0.7 if nucleophile_type == "Cys" else 0.0
        base_score = 1.0 if has_base else 0.0
        acid_score = 1.0 if has_acid else 0.0
        score = (nucleophile_score + base_score + acid_score) / 3.0
        if nucleophile_type == "Cys":
            notes.append("Cys nucleophile; serine-hydrolase track downgraded")
            score = min(score, 0.8)
            mechanism_label = "thiol_hydrolase_like"
            required_flags.append("requires_thiol_nucleophile")
        if not has_nucleophile:
            notes.append("Missing nucleophile residue")
        if not has_base:
            notes.append("Missing His base")
        if not has_acid:
            notes.append("Missing Asp/Glu acid")
    elif primary == "metallo_esterase":
        base_score = 1.0 if has_base else 0.0
        acid_score = 1.0 if has_acid else 0.0
        nucleophile_score = 0.2 if has_nucleophile else 0.1
        score = (0.4 * base_score) + (0.4 * acid_score) + (0.2 * nucleophile_score)
        metal_hint = len(acid_res) >= 2 and len(base_res) >= 1
        if metal_hint:
            score = min(1.0, score + 0.1)
            notes.append("Metal-binding hint present")
        if not has_base:
            notes.append("Missing His base")
        if not has_acid:
            notes.append("Missing acidic residue")
    else:
        score = (
            float(has_nucleophile) + float(has_base) + float(has_acid)
        ) / 3.0
        if not has_nucleophile:
            notes.append("Missing nucleophile residue")
        if not has_base:
            notes.append("Missing base residue")
        if not has_acid:
            notes.append("Missing acid residue")

    return {
        "track": primary,
        "mechanism_label": mechanism_label,
        "required_flags": required_flags,
        "has_nucleophile": has_nucleophile,
        "nucleophile_type": nucleophile_type,
        "has_base": has_base,
        "has_acid": has_acid,
        "score": round(score, 3),
        "notes": notes,
    }


def _probe_radius(
    structure_summary: Dict[str, Any],
    mode: str,
    size_proxies: Optional[Dict[str, Any]] = None,
) -> float:
    if size_proxies:
        min_diameter = size_proxies.get("min_diameter_proxy")
        if isinstance(min_diameter, (int, float)) and min_diameter > 0:
            base = max(1.4, float(min_diameter) / 2.0)
            return max(0.6, min(2.8, base))
    heavy_atoms = structure_summary.get("heavy_atoms") or 1
    base = 0.7 + heavy_atoms * 0.02
    if mode == "bulky_substrate":
        base += 0.4
    return max(0.6, min(2.8, base))


def _valid_weights(weights: Any) -> bool:
    if not isinstance(weights, dict):
        return False
    required = {"access", "reach", "retention"}
    if not required.issubset(weights.keys()):
        return False
    return all(isinstance(weights[key], (int, float)) for key in required)


def _valid_energy_weights(weights: Any) -> bool:
    if not isinstance(weights, dict):
        return False
    required = {"access", "reach", "retention", "topology"}
    if not required.issubset(weights.keys()):
        return False
    return all(isinstance(weights[key], (int, float)) for key in required)


def _normalize_energy_weights(weights: Dict[str, float]) -> Dict[str, float]:
    required = ("access", "reach", "retention", "topology")
    total = sum(float(weights.get(key, 0.0)) for key in required)
    if total <= 0.0:
        total = 1.0
    return {key: float(weights.get(key, 0.0)) / total for key in required}


def _resolve_energy_weights(
    route_weights: Dict[str, float],
    override: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if override and _valid_energy_weights(override):
        return _normalize_energy_weights(override)
    base = {
        "access": float(route_weights.get("access", DEFAULT_ENERGY_WEIGHTS["access"])),
        "reach": float(route_weights.get("reach", DEFAULT_ENERGY_WEIGHTS["reach"])),
        "retention": float(route_weights.get("retention", DEFAULT_ENERGY_WEIGHTS["retention"])),
        "topology": float(DEFAULT_ENERGY_WEIGHTS["topology"]),
    }
    return _normalize_energy_weights(base)


def _resolve_ensemble_temperature(value: Any) -> float:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return float(ENSEMBLE_TEMPERATURE)


def _normalize_vector(vector: List[float]) -> List[float]:
    norm = math.sqrt(sum(val * val for val in vector)) or 1.0
    return [val / norm for val in vector]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _clearance_margin(strictness: str) -> float:
    return {"lenient": 0.2, "standard": 0.4, "strict": 0.6}.get(strictness, 0.4)


def _nucleophile_geometry(mechanism_compat: Dict[str, Any]) -> str:
    nucleophile_type = (mechanism_compat or {}).get("nucleophile_type")
    if nucleophile_type == "Ser/Thr":
        return "serine_oxyanion"
    if nucleophile_type == "Cys":
        return "cysteine_thiol"
    if nucleophile_type in {"None", None}:
        return "none"
    return "generic_nucleophile"


def _access_confidence(tunnel_summary: Optional[Dict[str, Any]], access_score: float) -> float:
    if not tunnel_summary:
        return round(_clamp01(access_score), 3)
    bottleneck = tunnel_summary.get("bottleneck_radius")
    required = tunnel_summary.get("required_bottleneck_radius")
    path_length = tunnel_summary.get("path_length")
    curvature = tunnel_summary.get("curvature_proxy")
    if not isinstance(bottleneck, (int, float)) or not isinstance(required, (int, float)):
        return round(_clamp01(access_score), 3)
    clearance = bottleneck - required
    clearance_score = _clamp01(0.5 + (clearance / max(0.5, required)))
    path_score = _clamp01(1.0 - (float(path_length) / 40.0)) if isinstance(path_length, (int, float)) else 0.5
    curvature_score = _clamp01(1.0 - float(curvature)) if isinstance(curvature, (int, float)) else 0.5
    access_conf = (0.5 * clearance_score) + (0.3 * path_score) + (0.2 * curvature_score)
    return round(_clamp01(access_conf), 3)


def _temperature_k_from_context(condition_context: Dict[str, Any]) -> float:
    given = condition_context.get("given_conditions") or {}
    temp_c = given.get("temperature_c")
    if isinstance(temp_c, (int, float)):
        return c_to_k(float(temp_c))
    return 298.15


def _condition_context(
    job_card: Dict[str, Any],
    constraints: OperationalConstraints,
    unity_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rcf = job_card.get("reaction_condition_field") or {}
    given = rcf.get("given_conditions") or {}
    optimum = rcf.get("optimum_conditions_hint") or {}
    ph_value = given.get("pH")
    if ph_value is None:
        ph_min = constraints.ph_min
        ph_max = constraints.ph_max
        if ph_min is not None and ph_max is not None:
            ph_value = (ph_min + ph_max) / 2.0
        else:
            ph_value = ph_min if ph_min is not None else ph_max
    temp_value = given.get("temperature_c")
    if temp_value is None:
        temp_value = constraints.temperature_c
    if unity_state:
        unity_chem = unity_state.get("chem") or {}
        unity_phys = unity_state.get("physics") or {}
        unity_ph = (unity_chem.get("context") or {}).get("pH")
        unity_temp = (unity_phys.get("audit") or {}).get("temperature_K")
        if ph_value is None and isinstance(unity_ph, (int, float)):
            ph_value = float(unity_ph)
        if temp_value is None and isinstance(unity_temp, (int, float)):
            temp_value = k_to_c(float(unity_temp))

    ph_range = optimum.get("pH_range") or [6.5, 8.0]
    temp_range = optimum.get("temperature_c") or [25.0, 45.0]

    multipliers = {"access": 1.0, "reach": 1.0, "retention": 1.0}
    penalties: List[str] = []

    if ph_value is not None:
        if ph_value < ph_range[0] - 0.5 or ph_value > ph_range[1] + 0.5:
            multipliers["reach"] *= 0.9
            multipliers["retention"] *= 0.9
            penalties.append("pH outside preferred window reduces catalytic readiness.")
        if ph_value < ph_range[0] - 1.0 or ph_value > ph_range[1] + 1.0:
            multipliers["reach"] *= 0.85
            multipliers["retention"] *= 0.85
            penalties.append("Large pH deviation limits residue protonation states.")

    if temp_value is not None:
        if temp_value < temp_range[0] - 5:
            multipliers["reach"] *= 0.9
            multipliers["retention"] *= 1.03
            penalties.append("Low temperature slows kinetics; retention may improve slightly.")
        if temp_value > temp_range[1] + 5:
            multipliers["retention"] *= 0.9
            multipliers["access"] *= 0.95
            penalties.append("High temperature weakens binding stability.")

    return {
        "given_conditions": {"pH": ph_value, "temperature_c": temp_value},
        "optimum_conditions_hint": {"pH_range": ph_range, "temperature_c": temp_range},
        "multipliers": multipliers,
        "penalties": penalties,
    }


def _optimum_conditions_for_topology(
    base_score: float,
    optimum_hint: Dict[str, Any],
) -> Dict[str, Any]:
    ph_range = optimum_hint.get("pH_range") or [6.5, 8.0]
    temp_range = optimum_hint.get("temperature_c") or [25.0, 45.0]
    ph_candidates = [ph_range[0], sum(ph_range) / 2.0, ph_range[1]]
    temp_candidates = [temp_range[0], sum(temp_range) / 2.0, temp_range[1]]
    best = {"score": -1.0, "pH": ph_candidates[1], "temperature_c": temp_candidates[1]}
    for ph in ph_candidates:
        for temp in temp_candidates:
            score = _stability_score(base_score, ph, temp, optimum_hint)
            if score > best["score"]:
                best = {"score": score, "pH": round(ph, 2), "temperature_c": round(temp, 1)}
    return best


def _stability_score(
    base_score: float,
    ph_value: float,
    temp_value: float,
    optimum_hint: Dict[str, Any],
) -> float:
    ph_range = optimum_hint.get("pH_range") or [6.5, 8.0]
    temp_range = optimum_hint.get("temperature_c") or [25.0, 45.0]
    multiplier = 1.0
    if ph_value < ph_range[0] - 0.5 or ph_value > ph_range[1] + 0.5:
        multiplier *= 0.9
    if ph_value < ph_range[0] - 1.0 or ph_value > ph_range[1] + 1.0:
        multiplier *= 0.85
    if temp_value < temp_range[0] - 5:
        multiplier *= 0.9
    if temp_value > temp_range[1] + 5:
        multiplier *= 0.9
    return _clamp01(base_score * multiplier)


def _delta_from_optimum(
    given: Dict[str, Any],
    optimum_hint: Dict[str, Any],
) -> Dict[str, Any]:
    delta_ph = None
    delta_t = None
    if given.get("pH") is not None and optimum_hint.get("pH_range"):
        delta_ph = round(
            float(given["pH"]) - float(sum(optimum_hint["pH_range"]) / 2.0), 2
        )
    if given.get("temperature_c") is not None and optimum_hint.get("temperature_c"):
        delta_t = round(
            float(given["temperature_c"]) - float(sum(optimum_hint["temperature_c"]) / 2.0), 2
        )
    return {"delta_pH": delta_ph, "delta_T_C": delta_t}


def _retry_loop_suggestion(
    given: Dict[str, Any],
    optimum: Dict[str, Any],
    threshold_ph: float = 0.6,
    threshold_temp: float = 8.0,
) -> Dict[str, Any]:
    ph = given.get("pH")
    temp = given.get("temperature_c")
    if ph is None or temp is None:
        return {"status": "insufficient_data"}
    if "strategy_hint" in optimum:
        optimum = optimum.get("strategy_hint") or {}
    opt_ph = None
    opt_temp = None
    if optimum.get("pH_range"):
        opt_ph = sum(optimum["pH_range"]) / 2.0
    if optimum.get("temperature_c"):
        opt_temp = sum(optimum["temperature_c"]) / 2.0
    if opt_ph is None or opt_temp is None:
        return {"status": "unknown_optimum"}
    delta_ph = opt_ph - ph
    delta_temp = opt_temp - temp
    if abs(delta_ph) <= threshold_ph and abs(delta_temp) <= threshold_temp:
        return {"status": "close_enough"}
    return {
        "status": "suggest_retry",
        "proposed_conditions": {
            "pH": round(ph + (delta_ph * 0.5), 2),
            "temperature_c": round(temp + (delta_temp * 0.5), 1),
        },
        "expected_improvement": "moderate",
    }


def _apply_condition_adjustment(
    access_score: float,
    reach_score: float,
    retention_score: float,
    condition_context: Dict[str, Any],
) -> tuple[Dict[str, float], List[str]]:
    multipliers = condition_context.get("multipliers") or {}
    penalties = list(condition_context.get("penalties") or [])
    adjusted = {
        "access": _clamp01(access_score * multipliers.get("access", 1.0)),
        "reach": _clamp01(reach_score * multipliers.get("reach", 1.0)),
        "retention": _clamp01(retention_score * multipliers.get("retention", 1.0)),
    }
    return adjusted, penalties


def _energy_for_scores(
    access_score: float,
    reach_score: float,
    retention_score: float,
    topology_score: float,
    energy_weights: Dict[str, float],
) -> float:
    return (
        energy_weights["access"] * (1.0 - _clamp01(access_score))
        + energy_weights["reach"] * (1.0 - _clamp01(reach_score))
        + energy_weights["retention"] * (1.0 - _clamp01(retention_score))
        + topology_energy_component(topology_score, energy_weights["topology"])
    )


def compute_boltzmann_weight(deltaE_kJ_per_mol: float, T_K: float) -> float:
    """Return Boltzmann weight exp(-ΔE/(R*T)) using ΔE in kJ/mol and T in K."""
    deltaE_kj = float(deltaE_kJ_per_mol)
    if deltaE_kj > BOLTZMANN_CUTOFF_KJ_PER_MOL:
        return BOLTZMANN_TINY_WEIGHT
    return boltzmann_weight(deltaE_kj, T_K)


def _boltzmann_weights_physics_with_audit(
    deltaEs_kJ_per_mol: List[float],
    temperature_K: float,
) -> Tuple[List[float], Dict[str, Any]]:
    if not deltaEs_kJ_per_mol:
        return [], {
            "R_J_per_molK": R_J_per_molK,
            "T_K": round(float(temperature_K), 3),
            "beta": 0.0,
            "E_min_kJ_per_mol": 0.0,
            "sum_w_before_norm": 0.0,
            "sum_w_after_norm": 0.0,
            "sum_w": 0.0,
            "entropy_S": 0.0,
            "top_pose_p": 0.0,
            "expected_deltaE_kJ_per_mol": 0.0,
        }
    temp = max(1e-6, float(temperature_K))
    min_energy = min(float(value) for value in deltaEs_kJ_per_mol)
    beta = 1.0 / (R_J_per_molK * temp)
    raw = [
        boltzmann_weight(float(value) - min_energy, temp)
        for value in deltaEs_kJ_per_mol
    ]
    sum_raw = sum(raw)
    if sum_raw <= 0.0:
        weights = [1.0 / len(raw) for _ in raw]
        sum_raw = 0.0
    else:
        weights = [val / sum_raw for val in raw]
    sum_w = sum(weights)
    entropy = 0.0
    for weight in weights:
        if weight > 0.0:
            entropy -= weight * math.log(weight)
    expected_deltaE = sum(
        weight * float(value) for weight, value in zip(weights, deltaEs_kJ_per_mol)
    )
    top_pose_p = max(weights) if weights else 0.0
    audit = {
        "R_J_per_molK": R_J_per_molK,
        "T_K": round(float(temp), 3),
        "beta": round(float(beta), 8),
        "E_min_kJ_per_mol": round(float(min_energy), 6),
        "sum_w_before_norm": round(float(sum_raw), 6),
        "sum_w_after_norm": round(float(sum(weights)), 6),
        "sum_w": round(float(sum_w), 6),
        "entropy_S": round(float(entropy), 6),
        "top_pose_p": round(float(top_pose_p), 6),
        "expected_deltaE_kJ_per_mol": round(float(expected_deltaE), 6),
    }
    return weights, audit


def _boltzmann_weights(energies: List[float], temperature: float) -> List[float]:
    if not energies:
        return []
    k_b = 0.0019872041
    temp = max(1e-6, float(temperature))
    min_energy = min(float(value) for value in energies)
    raw = [
        math.exp(-(float(value) - min_energy) / (k_b * temp)) for value in energies
    ]
    total = sum(raw) or 1.0
    return [value / total for value in raw]


def _ci_band(
    values: List[float],
    weights: List[float],
    sample_count: int,
    quantile: float = 0.9,
) -> Tuple[float, float]:
    tail = (1.0 - quantile) / 2.0
    mean_val = weighted_mean(values, weights)
    low = weighted_quantile(values, weights, tail)
    high = weighted_quantile(values, weights, 1.0 - tail)
    if sample_count > 0:
        shrink = min(1.0, math.sqrt(32.0 / float(sample_count)))
        low = mean_val + (low - mean_val) * shrink
        high = mean_val + (high - mean_val) * shrink
    return _clamp01(low), _clamp01(high)


def _ensemble_metrics(
    scaffold_id: str,
    access_score: float,
    reach_score: float,
    retention_score: float,
    topology_score: float,
    weights: Dict[str, float],
    sample_count: int,
    energy_weights: Optional[Dict[str, float]] = None,
    temperature: float = ENSEMBLE_TEMPERATURE,
) -> Dict[str, Any]:
    rng = _rng_for_step(scaffold_id, "ensemble")
    count = max(3, sample_count)
    energy_weights = _normalize_energy_weights(
        energy_weights or DEFAULT_ENERGY_WEIGHTS
    )
    base_energy = _energy_for_scores(
        access_score,
        reach_score,
        retention_score,
        topology_score,
        energy_weights,
    )
    deltaE_base_kJ = max(0.0, float(base_energy)) * ENERGY_MODEL_SCALE_KJ_PER_MOL
    base_energy_score = max(1e-6, 1.0 - base_energy)
    samples: List[Tuple[float, float, float, float, float]] = []
    energies: List[float] = []
    deltaE_samples: List[float] = []
    for _ in range(count):
        energy = _clamp01(base_energy + rng.gauss(0.0, ENSEMBLE_ENERGY_NOISE))
        energy_score = 1.0 - energy
        scale = energy_score / base_energy_score
        access = _clamp01(access_score * scale)
        reach = _clamp01(reach_score * scale)
        retention = _clamp01(retention_score * scale)
        topology = _clamp01(topology_score * scale)
        total = (
            weights["access"] * access
            + weights["reach"] * reach
            + weights["retention"] * retention
        )
        samples.append((access, reach, retention, topology, total))
        energies.append(energy)
        deltaE_samples.append(max(0.0, float(energy)) * ENERGY_MODEL_SCALE_KJ_PER_MOL)

    temp_k = max(1e-6, float(temperature))
    weights_prob, boltz_audit = _boltzmann_weights_physics_with_audit(deltaE_samples, temp_k)
    access_vals = [val[0] for val in samples]
    reach_vals = [val[1] for val in samples]
    retention_vals = [val[2] for val in samples]
    topology_vals = [val[3] for val in samples]
    total_vals = [val[4] for val in samples]

    access_mean = weighted_mean(access_vals, weights_prob)
    reach_mean = weighted_mean(reach_vals, weights_prob)
    retention_mean = weighted_mean(retention_vals, weights_prob)
    topology_mean = weighted_mean(topology_vals, weights_prob)

    access_std = weighted_std(access_vals, weights_prob, access_mean)
    reach_std = weighted_std(reach_vals, weights_prob, reach_mean)
    retention_std = weighted_std(retention_vals, weights_prob, retention_mean)
    topology_std = weighted_std(topology_vals, weights_prob, topology_mean)

    access_prob = sum(
        weight for val, weight in zip(access_vals, weights_prob) if val >= 0.6
    )
    reach_prob = sum(weight for val, weight in zip(reach_vals, weights_prob) if val >= 0.6)
    retention_prob = sum(
        weight for val, weight in zip(retention_vals, weights_prob) if val >= 0.5
    )

    energy_mean = sum(energies) / len(energies)
    energy_std = math.sqrt(
        sum((energy - energy_mean) ** 2 for energy in energies) / len(energies)
    )
    deltaE_mean = sum(deltaE_samples) / len(deltaE_samples)
    weight_summary = {
        "min": round(min(weights_prob) if weights_prob else 0.0, 6),
        "max": round(max(weights_prob) if weights_prob else 0.0, 6),
        "mean": round(sum(weights_prob) / len(weights_prob) if weights_prob else 0.0, 6),
    }
    expected_deltaE = boltz_audit.get("expected_deltaE_kJ_per_mol")
    if expected_deltaE is None:
        expected_deltaE = float(deltaE_mean)
    temp_k = max(1e-6, float(temperature))
    occupancy_factor = math.exp(-(float(expected_deltaE) * 1000.0) / (R_J_per_molK * temp_k))
    occupancy_factor = max(PHYSICS_OCCUPANCY_FLOOR, min(1.0, float(occupancy_factor)))

    reach_vals_adjusted = [_clamp01(val * occupancy_factor) for val in reach_vals]
    retention_vals_adjusted = [_clamp01(val * occupancy_factor) for val in retention_vals]
    total_vals_adjusted = [
        _clamp01(
            weights["access"] * access
            + weights["reach"] * reach
            + weights["retention"] * retention
        )
        for access, reach, retention in zip(
            access_vals, reach_vals_adjusted, retention_vals_adjusted
        )
    ]

    reach_mean = weighted_mean(reach_vals_adjusted, weights_prob)
    retention_mean = weighted_mean(retention_vals_adjusted, weights_prob)
    total_mean = weighted_mean(total_vals_adjusted, weights_prob)
    reach_std = weighted_std(reach_vals_adjusted, weights_prob, reach_mean)
    retention_std = weighted_std(retention_vals_adjusted, weights_prob, retention_mean)
    total_std = weighted_std(total_vals_adjusted, weights_prob, total_mean)
    total_ci = _ci_band(total_vals_adjusted, weights_prob, count)
    retention_ci = _ci_band(retention_vals_adjusted, weights_prob, count)

    return {
        "sample_count": len(samples),
        "ensemble_temperature": round(float(temperature), 3),
        "energy_stats": {
            "mean": round(energy_mean, 3),
            "stdev": round(energy_std, 3),
        },
        "deltaE_model_kJ_per_mol": round(float(deltaE_mean), 3),
        "boltzmann_weight_summary": weight_summary,
        "boltzmann_audit": boltz_audit,
        "access_mean": round(access_mean, 3),
        "reach_mean": round(reach_mean, 3),
        "retention_mean": round(retention_mean, 3),
        "topology_mean": round(topology_mean, 3),
        "total_mean": round(total_mean, 3),
        "access_stdev": round(access_std, 3),
        "reach_stdev": round(reach_std, 3),
        "retention_stdev": round(retention_std, 3),
        "topology_stdev": round(topology_std, 3),
        "total_stdev": round(total_std, 3),
        "access_prob": round(access_prob, 3),
        "reach_prob": round(reach_prob, 3),
        "retention_prob": round(retention_prob, 3),
        "score_ci90": {
            "total": [round(total_ci[0], 3), round(total_ci[1], 3)],
            "retention": [round(retention_ci[0], 3), round(retention_ci[1], 3)],
        },
        "occupancy": {
            "expected_deltaE_kJ_per_mol": round(float(expected_deltaE), 3),
            "top_pose_p": boltz_audit.get("top_pose_p"),
            "entropy_S": boltz_audit.get("entropy_S"),
            "sum_w": boltz_audit.get("sum_w"),
            "factor": round(float(occupancy_factor), 3),
            "reach_mean_raw": round(weighted_mean(reach_vals, weights_prob), 3),
            "retention_mean_raw": round(weighted_mean(retention_vals, weights_prob), 3),
        },
    }


def _estimate_substrate_radius_A(
    size_proxies: Dict[str, Any],
    structure_summary: Dict[str, Any],
    assumptions: List[str],
) -> float:
    radius = size_proxies.get("approx_radius")
    if isinstance(radius, (int, float)) and radius > 0:
        return float(radius)
    heavy_atoms = structure_summary.get("heavy_atoms") or 0
    if heavy_atoms:
        assumptions.append("radius inferred from heavy atom count")
        estimate = 0.8 * (float(heavy_atoms) ** (1.0 / 3.0))
        return max(0.6, min(3.5, estimate))
    assumptions.append("radius defaulted to 1.5A")
    return 1.5


def _diffusion_cap_factor(
    radius_A: float,
    temp_k: float,
    assumptions: List[str],
    path_length_A: Optional[float] = None,
    condition_profile: Optional[Dict[str, Any]] = None,
    substrate_context: Optional[Dict[str, Any]] = None,
    external_cap_s_inv: Optional[float] = None,
) -> Dict[str, Any]:
    D = DEFAULT_DIFFUSION_COEFF_M2_S
    tau_s = None
    length_A = None
    radius_m = max(1e-12, float(radius_A) * 1e-10)
    k_cap = diffusion_cap_rate(substrate_context, condition_profile)
    if not isinstance(k_cap, (int, float)) or not math.isfinite(float(k_cap)):
        k_cap = DIFFUSION_CAP_S_INV
    if isinstance(external_cap_s_inv, (int, float)) and math.isfinite(float(external_cap_s_inv)):
        k_cap = min(float(k_cap), float(external_cap_s_inv))
        assumptions.append("unity diffusion cap applied")
    if isinstance(path_length_A, (int, float)) and float(path_length_A) > 0.0:
        length_A = float(path_length_A)
        length_m = length_A * 1e-10
        tau_s = (length_m ** 2) / (2.0 * D)
        k_s_inv = 1.0 / max(float(tau_s), 1e-12)
        k_s_inv = min(float(k_s_inv), float(k_cap))
        assumptions.append("diffusion cap derived from tunnel length")
    else:
        k_s_inv = float(k_cap)
        assumptions.append("diffusion cap derived from Smoluchowski (no tunnel length)")
    cap_ratio = math.log10(1.0 + k_s_inv) / math.log10(1.0 + max(float(k_cap), 1.0))
    cap_factor = 0.6 + (0.4 * max(0.0, min(1.0, cap_ratio)))
    cap_factor = max(PHYSICS_CAP_FLOOR, min(1.0, cap_factor))
    return {
        "D_m2_s": float(D),
        "L_A": round(float(length_A), 3) if length_A is not None else None,
        "tau_s": float(tau_s) if tau_s is not None else None,
        "k_diff_s_inv": float(k_s_inv),
        "k_cap_s_inv": float(k_cap),
        "k_cap_display": format_rate(k_cap),
        "kdiff": float(k_s_inv),
        "pseudo_first_order_s_inv": float(k_s_inv),
        "cap_factor": round(float(cap_factor), 3),
        "assumptions": assumptions,
    }


def _electrostatics_factor(
    bond_context: Dict[str, Any],
    temp_k: float,
    assumptions: List[str],
    ionic_strength: Optional[float] = None,
    residues: Optional[List[str]] = None,
) -> Dict[str, Any]:
    charge = bond_context.get("gasteiger_charge_a")
    if charge is None:
        charge = bond_context.get("gasteiger_charge_C")
    if not isinstance(charge, (int, float)):
        assumptions.append("charges unavailable; electrostatics neutral")
        return {"E_kJ_mol": 0.0, "boltz_factor": 1.0}
    counter_charge = 0.0
    residue_charge = 0.0
    if residues:
        for residue in residues:
            name = str(residue or "").upper()
            if name.startswith("ARG") or name.startswith("LYS"):
                residue_charge += 1.0
            elif name.startswith("ASP") or name.startswith("GLU"):
                residue_charge -= 1.0
            elif name.startswith("HIS"):
                residue_charge += 0.1
    if residue_charge != 0.0:
        counter_charge = residue_charge
        assumptions.append("counter-charge inferred from residue list")
    elif abs(float(charge)) >= 0.05:
        counter_charge = -ASSUMED_COUNTER_CHARGE_E if float(charge) > 0 else ASSUMED_COUNTER_CHARGE_E
        assumptions.append("assumed counter-charge for pocket residue")
    else:
        assumptions.append("near-neutral charge; electrostatics skipped")
        return {"E_kJ_mol": 0.0, "boltz_factor": 1.0}
    distance_A = ASSUMED_INTERACTION_DISTANCE_A
    assumptions.append(f"assumed interaction distance {distance_A}A")
    energy_kj_mol = screened_coulomb_energy_kJ(
        float(charge),
        float(counter_charge),
        distance_A,
        dielectric=20.0,
    )
    screening = screening_factor(ionic_strength)
    screening_value = screening.get("value")
    if screening_value is None:
        assumptions.append("ionic strength unknown; electrostatics unscreened")
        screening_value = 1.0
    else:
        assumptions.append("ionic screening applied")
    energy_kj_mol *= float(screening_value)
    if energy_kj_mol <= 0:
        boltz_factor = 1.0
    else:
        boltz_factor = boltzmann_weight(float(energy_kj_mol), temp_k)
    boltz_factor = max(PHYSICS_ELECTROSTATICS_FLOOR, min(1.0, float(boltz_factor)))
    return {
        "E_kJ_mol": round(float(energy_kj_mol), 3),
        "boltz_factor": round(float(boltz_factor), 3),
        "ionic_screening": screening,
    }


def _physics_multiplier(
    job_card: Dict[str, Any],
    condition_context: Dict[str, Any],
    path_length_A: Optional[float] = None,
) -> Dict[str, Any]:
    assumptions: List[str] = []
    size_proxies = job_card.get("substrate_size_proxies") or {}
    structure_summary = job_card.get("structure_summary") or {}
    radius_A = _estimate_substrate_radius_A(size_proxies, structure_summary, assumptions)
    temp_k = _temperature_k_from_context(condition_context)
    diffusion = _diffusion_cap_factor(
        radius_A,
        temp_k,
        assumptions,
        path_length_A,
        condition_profile={"temperature_K": temp_k},
        substrate_context={"approx_radius": radius_A, "substrate_size_proxies": size_proxies},
    )
    condition_profile = job_card.get("condition_profile") or {}
    ionic_strength = condition_profile.get("ionic_strength")
    electrostatics = _electrostatics_factor(
        job_card.get("bond_context") or {},
        temp_k,
        assumptions,
        ionic_strength=ionic_strength,
    )
    multiplier = float(diffusion["cap_factor"]) * float(electrostatics["boltz_factor"])
    multiplier = max(PHYSICS_MULTIPLIER_MIN, min(1.0, multiplier))
    return {
        "diffusion": {
            **diffusion,
            "assumptions": assumptions,
        },
        "electrostatics": {
            **electrostatics,
            "assumptions": assumptions,
        },
        "multiplier": round(float(multiplier), 3),
    }


def _predictor_score(
    access_score: float,
    reach_score: float,
    retention_score: float,
    reach_summary: Dict[str, Any],
    retention_metrics: Dict[str, Any],
) -> float:
    base = 0.45 * access_score + 0.35 * reach_score + 0.2 * retention_score
    if retention_metrics.get("retention_risk_flag") == "HIGH":
        base -= 0.1
    if "WARN_RETENTION_WEAK_BINDING" in (retention_metrics.get("warning_codes") or []):
        base -= 0.05
    if reach_summary.get("fail_code") == FAIL_MECH_COMPAT:
        base -= 0.2
    return _clamp01(base)


def _pareto_front(scaffolds: List[Dict[str, Any]]) -> List[str]:
    front_ids: List[str] = []
    for scaffold in scaffolds:
        scores = scaffold.get("scores") or {}
        candidate = (
            scores.get("access_score", 0.0),
            scores.get("reach_score", 0.0),
            scores.get("retention_score", 0.0),
        )
        dominated = False
        for other in scaffolds:
            if other is scaffold:
                continue
            other_scores = other.get("scores") or {}
            other_vec = (
                other_scores.get("access_score", 0.0),
                other_scores.get("reach_score", 0.0),
                other_scores.get("retention_score", 0.0),
            )
            if (
                other_vec[0] >= candidate[0]
                and other_vec[1] >= candidate[1]
                and other_vec[2] >= candidate[2]
                and (
                    other_vec[0] > candidate[0]
                    or other_vec[1] > candidate[1]
                    or other_vec[2] > candidate[2]
                )
            ):
                dominated = True
                break
        if not dominated:
            front_ids.append(scaffold.get("scaffold_id"))
    return front_ids


def _topology_signature(
    tunnel_summary: Optional[Dict[str, Any]],
    pocket_center: Optional[List[float]],
    mode: str,
) -> Dict[str, Any]:
    point_cloud: List[List[float]] = []
    if pocket_center:
        point_cloud.append(list(pocket_center))
    if tunnel_summary and tunnel_summary.get("entry_point"):
        entry = tunnel_summary.get("entry_point")
        if isinstance(entry, list) and len(entry) == 3:
            point_cloud.append(list(entry))
    signature = compute_signature(point_cloud or None, tunnel_summary, mode)
    topology_score, robustness = score_signature(signature)
    signature["pocket_center"] = pocket_center
    signature["tunnel"] = tunnel_summary or {}
    signature["topology_score"] = topology_score
    signature["robustness"] = robustness
    return signature


def _feasibility_flag(
    module1_total: float,
    fail_codes: List[str],
    topology_score: Optional[float] = None,
) -> bool:
    if fail_codes and FAIL_MECH_COMPAT in fail_codes:
        return False
    if topology_score is not None and topology_score < 0.3:
        return False
    return module1_total >= 0.5


def _required_topology_constraints(
    tunnel_summary: Optional[Dict[str, Any]],
    flexibility: Dict[str, Any],
) -> List[str]:
    constraints: List[str] = []
    if tunnel_summary:
        bottleneck = tunnel_summary.get("bottleneck_radius")
        required = tunnel_summary.get("required_bottleneck_radius")
        if bottleneck is not None and required is not None:
            constraints.append(
                f"bottleneck_radius >= {required}"
            )
    if flexibility.get("open_fraction") is not None:
        constraints.append("tunnel_open_fraction >= 0.3")
    return constraints


def _ci90_from_mean_std(mean: Optional[float], std: Optional[float]) -> Optional[Tuple[float, float]]:
    if mean is None or std is None:
        return None
    try:
        m_val = float(mean)
        s_val = float(std)
    except (TypeError, ValueError):
        return None
    margin = 1.64 * s_val
    return (max(0.0, m_val - margin), min(1.0, m_val + margin))


def _build_scorecard_module1(
    module1_confidence: Dict[str, Any],
    evidence_record: Dict[str, Any],
) -> Dict[str, Any]:
    ensemble = module1_confidence.get("ensemble") or {}
    total_mean = ensemble.get("total_mean")
    total_std = ensemble.get("total_stdev")
    total_value = module1_confidence.get("total")
    retention_mean = ensemble.get("retention_mean")
    retention_std = ensemble.get("retention_stdev")
    retention_value = module1_confidence.get("retention")
    topology_value = module1_confidence.get("topology")

    ci_total = _ci90_from_mean_std(
        total_mean if total_mean is not None else total_value,
        total_std,
    )
    ci_retention = _ci90_from_mean_std(
        retention_mean if retention_mean is not None else retention_value,
        retention_std,
    )
    features = ((evidence_record.get("features_used") or {}).get("values") or {})
    contributors = contributors_from_features(features, limit=5)

    metrics = [
        ScoreCardMetric(
            name="module1_total",
            raw=float(total_value) if isinstance(total_value, (int, float)) else None,
            calibrated=float(total_value) if isinstance(total_value, (int, float)) else None,
            ci90=ci_total,
            n_eff=None,
            status=metric_status(total_value, None),
            definition="Composite access/reach/retention score after physics weighting.",
            contributors=contributors,
        ),
        ScoreCardMetric(
            name="retention",
            raw=float(retention_value) if isinstance(retention_value, (int, float)) else None,
            calibrated=float(retention_value)
            if isinstance(retention_value, (int, float))
            else None,
            ci90=ci_retention,
            n_eff=None,
            status=metric_status(retention_value, None),
            definition="Retention feasibility for substrate orientation in pocket.",
            contributors=contributors,
        ),
        ScoreCardMetric(
            name="topology",
            raw=float(topology_value) if isinstance(topology_value, (int, float)) else None,
            calibrated=float(topology_value) if isinstance(topology_value, (int, float)) else None,
            ci90=None,
            n_eff=None,
            status=metric_status(topology_value, None),
            definition="Topological feasibility from persistent homology signature.",
            contributors=contributors,
        ),
    ]
    return ScoreCard(module_id=1, metrics=metrics, calibration_status="heuristic").to_dict()


def _build_score_ledger_module1(
    module1_confidence: Dict[str, Any],
    weight_adjustment: Dict[str, Any],
    module1_physics_audit: Dict[str, Any],
    retention_threshold: float,
) -> Dict[str, Any]:
    def _as_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    adjusted = weight_adjustment.get("adjusted") or {}
    original = weight_adjustment.get("original") or {}
    terms = [
        ScoreTerm(
            name="access_score",
            value=_as_float(module1_confidence.get("access")),
            unit="probability",
            formula="module1_confidence.access",
            inputs={},
            notes="Access feasibility after physics weighting.",
        ),
        ScoreTerm(
            name="reach_score",
            value=_as_float(module1_confidence.get("reach")),
            unit="probability",
            formula="module1_confidence.reach",
            inputs={},
            notes="Reach feasibility after physics weighting.",
        ),
        ScoreTerm(
            name="retention_score",
            value=_as_float(module1_confidence.get("retention")),
            unit="probability",
            formula="module1_confidence.retention",
            inputs={},
            notes="Retention feasibility after physics weighting.",
        ),
        ScoreTerm(
            name="module1_total",
            value=_as_float(module1_confidence.get("total")),
            unit="probability",
            formula="weighted_sum(access, reach, retention)",
            inputs={"weights": adjusted or original},
            notes="Composite score using access/reach/retention weights.",
        ),
        ScoreTerm(
            name="weight_access",
            value=_as_float(adjusted.get("access") or original.get("access")),
            unit="weight",
            formula="config or retention-adjusted weight",
            inputs={"original": original.get("access"), "adjusted": adjusted.get("access")},
            notes="Access weighting in total score.",
        ),
        ScoreTerm(
            name="weight_reach",
            value=_as_float(adjusted.get("reach") or original.get("reach")),
            unit="weight",
            formula="config or retention-adjusted weight",
            inputs={"original": original.get("reach"), "adjusted": adjusted.get("reach")},
            notes="Reach weighting in total score.",
        ),
        ScoreTerm(
            name="weight_retention",
            value=_as_float(adjusted.get("retention") or original.get("retention")),
            unit="weight",
            formula="config or retention-adjusted weight",
            inputs={"original": original.get("retention"), "adjusted": adjusted.get("retention")},
            notes="Retention weighting in total score.",
        ),
        ScoreTerm(
            name="retention_weak_threshold",
            value=float(retention_threshold),
            unit="probability",
            formula="config.RETENTION_WEAK_THRESHOLD",
            inputs={},
            notes=RETENTION_WEAK_THRESHOLD.rationale,
        ),
        ScoreTerm(
            name="physics_multiplier",
            value=_as_float(module1_physics_audit.get("physics_multiplier")),
            unit="multiplier",
            formula="physics_block.multiplier",
            inputs={},
            notes="Physics multiplier applied to access/reach/retention.",
        ),
        ScoreTerm(
            name="diffusion_cap_s_inv",
            value=_as_float(module1_physics_audit.get("diffusion_cap_s_inv")),
            unit="s^-1",
            formula="physicscore.diffusion_cap_rate",
            inputs={},
            notes="Diffusion-limited cap on effective rate.",
        ),
    ]
    return ScoreLedger(module_id=1, terms=terms).to_dict()


def _rng_for_step(scaffold_id: str, step: str) -> random.Random:
    seed = f"{scaffold_id}:{step}"
    value = int(hashlib.sha1(seed.encode("utf-8")).hexdigest(), 16)
    return random.Random(value)


def _top_k_from_difficulty(difficulty: str, scaffold_count: int) -> int:
    if difficulty == "EASY":
        return min(5, scaffold_count)
    if difficulty == "MEDIUM":
        return min(10, scaffold_count)
    if scaffold_count >= 200:
        return min(25, scaffold_count)
    return min(15, scaffold_count)


def _reject_scaffold(
    scaffold: Scaffold,
    fail_codes: List[str],
    state: Dict[str, Any],
    tunnel_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "scaffold_id": scaffold.scaffold_id,
        "pdb_path": scaffold.pdb_path,
        "pocket_center": (state.get("pocket_center_candidates") or [None])[0],
        "tunnel_summary": tunnel_summary,
        "reach_summary": {},
        "attack_envelope": {},
        "retention_metrics": {},
        "scores": {},
        "fail_codes": fail_codes,
    }


def _fail_handoff(
    halt_reason: str,
    rejected: Optional[List[Dict[str, Any]]] = None,
    cache_hits: int = 0,
    cache_misses: int = 0,
    cache_writes: int = 0,
    cache_size: int = 0,
) -> Dict[str, Any]:
    return {
        "status": "FAIL",
        "halt_reason": halt_reason,
        "cache_stats": {
            "hits": cache_hits,
            "misses": cache_misses,
            "entries_written": cache_writes,
            "cache_size": cache_size,
        },
        "ranked_scaffolds": [],
        "rejected_scaffolds": rejected or [],
        "module2_handoff": {"top_scaffolds": []},
    }
