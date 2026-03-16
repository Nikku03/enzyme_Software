from __future__ import annotations

# Contract Notes (output contract freeze):
# - ctx.data["module3_experiment_designer"] must preserve keys: status, halt_reason, protocol_card,
#   information_gain, qc_guardrails, learning_update, warnings/errors, qc_result, qc_status/qc_reasons,
#   predicted_under_given_conditions, module3_physics_audit, and evidence_record.
# - shared_io updates via _merge_shared_io must keep shared_io["input"/"outputs"] intact.
# - New physics fields should be added under module3_physics_audit or math_contract, not rename existing keys.
# - Plan score computed in _compute_information_gain and _plan_score_physics; QC guardrails in
#   _qc_rules/_qc_check and wetlab ingestion in _ingest_wetlab_results.

from dataclasses import dataclass
import os
import math
import statistics
import uuid
from typing import Any, Dict, List, Optional, Tuple

from enzyme_software.context import PipelineContext
from enzyme_software.domain import (
    ConditionProfile,
    EvidenceRecord,
    ExperimentRecord,
    FeatureVector,
    SharedOutput,
)
from enzyme_software.evidence_store import add_datapoints, save_run
from enzyme_software.mathcore import BayesianDAGRouter
from enzyme_software.mathcore.uncertainty import (
    DistributionEstimate,
    ProbabilityEstimate,
    QCReport,
    beta_entropy,
    bernoulli_entropy,
    sigmoid,
    validate_math_contract,
)
from enzyme_software.modules.base import BaseModule
from enzyme_software.physicscore import compute_route_prior
from enzyme_software.unity_schema import (
    BondContext,
    ConditionProfile as UnityConditionProfile,
    Module0Out,
    Module1Out,
    Module2Out,
    Module3Out,
    PhysicsAudit,
    UnityRecord,
)
from enzyme_software.scorecard import (
    ScoreCard,
    ScoreCardMetric,
    contributors_from_features,
    metric_status,
)
from enzyme_software.score_ledger import ScoreLedger, ScoreTerm

PHYSICS_PLAN_HORIZON_S = 3600.0
NOISE_FLOOR_CONVERSION = 0.02
PLAN_SCORE_LOG_NORM = 1.5
PLAN_SCORE_SNR_REF = 30.0
DETECTABILITY_TIME_S = 3600.0
DETECTABILITY_NOISE_FLOOR = 0.02
OVERALL_SIGNAL_SNR_SCALE = 1.5

HYPOTHESIS_ARM_RELEVANCE: Dict[str, Dict[str, float]] = {
    "conditions_limited": {
        "baseline": 0.3,
        "improved_conditions": 1.0,
        "negative_control": 0.1,
        "variant_disambiguation": 0.2,
        "stress_boundary": 0.8,
    },
    "access_limited": {
        "baseline": 0.4,
        "improved_conditions": 0.2,
        "negative_control": 0.1,
        "variant_disambiguation": 0.7,
        "stress_boundary": 0.3,
    },
    "retention_limited": {
        "baseline": 0.3,
        "improved_conditions": 0.4,
        "negative_control": 0.1,
        "variant_disambiguation": 0.9,
        "stress_boundary": 0.5,
    },
    "mechanism_mismatch": {
        "baseline": 0.5,
        "improved_conditions": 0.3,
        "negative_control": 0.8,
        "variant_disambiguation": 0.4,
        "stress_boundary": 0.3,
    },
}


@dataclass
class Module3ExperimentDesigner(BaseModule):
    module_id: int = 3
    name: str = "Module 3 - Experiment Designer"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        result = run_module3(ctx)
        ctx.data["module3_experiment_designer"] = result
        return ctx

    @staticmethod
    def _expected_information_gain(
        prior: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> List[float]:
        """Estimate entropy reduction from candidate deltas against a Bernoulli prior."""
        def _entropy(probability: float) -> float:
            prob = max(1e-6, min(1.0 - 1e-6, float(probability)))
            return -(prob * math.log(prob) + (1.0 - prob) * math.log(1.0 - prob))

        p_success = prior.get("p_success")
        if not isinstance(p_success, (int, float)):
            p_success = 0.5
        base_entropy = _entropy(float(p_success))

        scores: List[float] = []
        for candidate in candidates:
            delta = candidate.get("delta", 0.0)
            try:
                delta_val = float(delta)
            except (TypeError, ValueError):
                delta_val = 0.0
            shifted = max(1e-6, min(1.0 - 1e-6, float(p_success) + delta_val))
            scores.append(base_entropy - _entropy(shifted))
        return scores


def run_module3(ctx: PipelineContext) -> Dict[str, Any]:
    shared = ctx.data.get("shared_io") or {}
    shared_input = shared.get("input") or {}
    shared_outputs = shared.get("outputs") or {}

    job_card = ctx.data.get("job_card") or {}
    if _reaction_hash_mismatch(shared_input, job_card):
        warnings = ["W_HASH_MISMATCH: reaction identity mismatch; halting module3."]
        output = {
            "status": "FAIL",
            "halt_reason": "FAIL_HASH_MISMATCH",
            "protocol_card": {},
            "information_gain": {},
            "qc_guardrails": {},
            "learning_update": {"status": "REJECTED_HASH_MISMATCH", "records_to_write": 0},
            "warnings": warnings,
            "errors": [],
        }
        ctx.data["shared_io"] = _merge_shared_io(ctx, output)
        _update_unity_record_parts(ctx, output)
        _emit_unity_record(ctx, output)
        return output

    wetlab_results = shared_input.get("wetlab_results")
    existing_protocol = None
    if wetlab_results:
        existing_protocol = (ctx.data.get("module3_experiment_designer") or {}).get(
            "protocol_card"
        )

    module2_output = shared_outputs.get("module2") or {}
    module2_raw = ctx.data.get("module2_active_site_refinement") or {}
    module2_handoff = module2_raw.get("module3_handoff") or {}
    variants = module2_raw.get("variant_set") or []
    mechanism_spec = (
        ((ctx.data.get("unity_state") or {}).get("bio") or {}).get("mechanism_spec")
        or module2_raw.get("mechanism_spec")
        or {}
    )

    best_variant = module2_handoff.get("best_variant") or module2_raw.get("best_variant")
    if not best_variant:
        best_variant = _find_variant(variants, "V0") or {"variant_id": "V0", "label": "Baseline"}

    baseline_conditions = _baseline_conditions(ctx, shared_input)
    if existing_protocol:
        protocol_card = existing_protocol
        arms = protocol_card.get("arms", [])
    else:
        improved_conditions = _improved_conditions(
            baseline_conditions,
            module2_output.get("retry_loop_suggestion") or {},
            module2_output.get("optimum_conditions") or {},
        )
        mechanism_disambiguation = None
        require_mechanism_disambiguation = (
            mechanism_spec.get("policy_action") == "REQUEST_DISAMBIGUATION"
        )
        if require_mechanism_disambiguation:
            mechanism_disambiguation = _mechanism_disambiguation(best_variant, variants)
        disambiguation_candidate = _variant_disambiguation(best_variant, variants)
        stress_conditions = _stress_conditions(ctx, baseline_conditions)

        candidates = _candidate_arms(
            baseline_conditions,
            improved_conditions,
            stress_conditions,
            best_variant,
            disambiguation_candidate,
            mechanism_disambiguation,
            variants,
        )
        require_disambiguation = _has_strong_variant(variants)
        if require_mechanism_disambiguation:
            require_disambiguation = False
        arms = _select_arms_by_eig(
            ctx,
            candidates,
            baseline_conditions,
            require_disambiguation,
            require_mechanism_disambiguation,
        )
        _assign_arm_ids(arms)
        negative_control_id = _find_arm_id(arms, "negative_control")
        protocol_card = {
            "batch_id": str(uuid.uuid4()),
            "arms": arms,
            "controls": {"negative_control_arm_id": negative_control_id},
        }

    information_gain = _compute_information_gain(ctx, protocol_card, baseline_conditions)
    selected_physics_gate = (
        (module2_raw.get("selected_scaffold") or {}).get("physics_gate") or {}
    )
    if selected_physics_gate.get("ok") == 0.0:
        notes = information_gain.get("notes") or []
        notes.append("PHYSICS_BARRIER_HIGH: thermal barrier exceeds limit.")
        information_gain["notes"] = list(dict.fromkeys(notes))
    predicted_under_given_conditions = _predicted_under_given_conditions(protocol_card)
    evidence_record = None

    qc_guardrails = {
        "veto_rules": [
            "Abort if negative control shows >5% conversion.",
            "Abort if baseline fails to reach detectable conversion.",
        ],
        "variance_rules": [
            "Repeat any arm with >20% replicate variance.",
            "Flag arms with inconsistent pH drift (>0.3).",
        ],
        "qc_rules": _qc_rules(status="n/a"),
    }

    learning_update = {"status": "PLANNED", "records_to_write": 5, "router_updates": []}
    wetlab_ingest_summary = None
    qc_result = None
    if wetlab_results:
        wetlab_ingest_summary, qc_result, learning_update = _ingest_wetlab_results(
            ctx,
            protocol_card,
            wetlab_results,
            baseline_conditions,
        )
        qc_guardrails["qc_rules"] = _qc_rules_from_result(qc_result)

    plan_score_raw = float(information_gain.get("plan_score") or 0.0)
    plan_score_physics, physics_audit = _plan_score_physics(ctx, protocol_card)
    detectability = _detectability_model(ctx, protocol_card)
    detectability_factor = float(detectability.get("detectability_factor") or 1.0)
    final_plan_score = max(
        0.0, min(1.0, (0.5 * plan_score_raw + 0.5 * plan_score_physics) * detectability_factor)
    )
    information_gain["plan_score_eig"] = round(plan_score_raw, 3)
    information_gain["plan_score_physics"] = round(plan_score_physics, 3)
    if isinstance(physics_audit.get("plan_phys"), (int, float)):
        information_gain["plan_phys"] = round(float(physics_audit.get("plan_phys")), 3)
    information_gain["plan_score"] = round(final_plan_score, 3)
    plan_score = float(information_gain.get("plan_score") or 0.0)
    evidence_record = _build_evidence_record(protocol_card, information_gain)
    confidence_estimate = ProbabilityEstimate(
        p_raw=plan_score,
        p_cal=plan_score,
        ci90=(plan_score, plan_score),
        n_eff=BayesianDAGRouter.effective_sample_size(plan_score),
    ).to_dict()
    prediction_estimates = {
        "plan_score": DistributionEstimate(
            mean=plan_score,
            std=0.0,
            ci90=(plan_score, plan_score),
        ).to_dict()
    }
    qc_payload = qc_result or {"status": "N/A", "reasons": [], "warnings": [], "outcome_entropy": None}
    qc_report = QCReport(
        status=qc_payload.get("status", "N/A"),
        reasons=qc_payload.get("reasons") or [],
        metrics={
            "outcome_entropy": qc_payload.get("outcome_entropy"),
            "warnings": qc_payload.get("warnings", []),
        },
    ).to_dict()
    math_contract = {
        "confidence": confidence_estimate,
        "predictions": prediction_estimates,
        "qc": qc_report,
    }
    scorecard = _build_scorecard_module3(information_gain, physics_audit, math_contract)
    score_ledger = _build_score_ledger_module3(information_gain, physics_audit, detectability)

    output = {
        "status": "PASS",
        "halt_reason": None,
        "protocol_card": protocol_card,
        "information_gain": information_gain,
        "qc_guardrails": qc_guardrails,
        "qc_status": qc_payload.get("status", "N/A"),
        "qc_reasons": qc_payload.get("reasons") or [],
        "learning_update": learning_update,
        "physics_gate": selected_physics_gate or None,
        "module3_physics_audit": physics_audit,
        "physics": detectability,
        "predicted_under_given_conditions": predicted_under_given_conditions,
        "evidence_record": evidence_record,
        "math_contract": math_contract,
        "scorecard": scorecard,
        "score_ledger": score_ledger,
        "warnings": [],
        "errors": [],
    }
    if output.get("module3_physics_audit", {}).get("plan_phys") is None:
        output["warnings"].append(
            "W_MISSING_PHYSICS: plan_phys unavailable; missing physics inputs."
        )
    energy_ledger = (ctx.data.get("shared_io") or {}).get("energy_ledger") or {}
    output["energy_ledger_update"] = _energy_ledger_update(output, energy_ledger)
    if wetlab_results:
        output["wetlab_ingest_summary"] = wetlab_ingest_summary
        output["qc_result"] = qc_result
    contract_violations = validate_math_contract(output)
    if contract_violations:
        output["warnings"] = list(
            dict.fromkeys((output.get("warnings") or []) + contract_violations)
        )
    ctx.data["shared_io"] = _merge_shared_io(ctx, output)
    _update_unity_record_parts(ctx, output)
    _emit_unity_record(ctx, output)
    return output


def _baseline_conditions(ctx: PipelineContext, shared_input: Dict[str, Any]) -> Dict[str, Any]:
    condition_profile = shared_input.get("condition_profile") or {}
    pH = condition_profile.get("pH")
    temp_c = condition_profile.get("temperature_C")
    if pH is None or temp_c is None:
        job_card = ctx.data.get("job_card") or {}
        given = (job_card.get("reaction_condition_field") or {}).get("given_conditions") or {}
        pH = pH if pH is not None else given.get("pH")
        temp_c = temp_c if temp_c is not None else given.get("temperature_c")
    if pH is None:
        pH = 7.0
    if temp_c is None:
        temp_c = 30.0
    return {"pH": round(float(pH), 2), "temperature_c": round(float(temp_c), 1)}


def _plan_score_physics(ctx: PipelineContext, protocol_card: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    module2_raw = ctx.data.get("module2_active_site_refinement") or {}
    module0_job_card = ctx.data.get("job_card") or {}
    module2_physics = module2_raw.get("module2_physics_audit") or {}
    module0_physics = module0_job_card.get("physics_audit") or {}
    shared_io = ctx.data.get("shared_io") or {}
    physics_block = shared_io.get("physics") or {}
    energy_ledger = shared_io.get("energy_ledger") or {}
    unity_state = ctx.data.get("unity_state") or {}
    unity_physics = unity_state.get("physics") or {}
    unity_mechanism = unity_state.get("mechanism") or {}
    mechanism_evidence = unity_mechanism.get("evidence") or {}
    unity_ledger = (unity_physics.get("derived") or {}).get("energy_ledger") or {}
    if not energy_ledger and unity_ledger:
        energy_ledger = dict(unity_ledger)
    k_effective = physics_block.get("k_eff_s_inv")
    horizon_s = physics_block.get("horizon_s")
    p_convert = physics_block.get("p_convert_horizon")
    if isinstance(energy_ledger.get("k_eff_s_inv"), (int, float)):
        k_effective = energy_ledger.get("k_eff_s_inv")
    if isinstance(energy_ledger.get("horizon_s"), (int, float)):
        horizon_s = energy_ledger.get("horizon_s")
    if isinstance(energy_ledger.get("p_success_horizon"), (int, float)):
        p_convert = energy_ledger.get("p_success_horizon")
    route_priors = unity_physics.get("priors") or {}
    chosen_route = module0_job_card.get("chosen_route")
    if k_effective is None and chosen_route and isinstance(route_priors.get(chosen_route), dict):
        route_payload = route_priors.get(chosen_route) or {}
        k_effective = route_payload.get("k_effective_s_inv") or route_payload.get("k_eff_s_inv")
    if not isinstance(k_effective, (int, float)) or float(k_effective) <= 0.0:
        k_effective = (
            module2_physics.get("k_eff_s_inv")
            or module2_physics.get("k_variant_s_inv")
            or module2_physics.get("eyring_k_variant_s_inv")
            or module0_physics.get("k_eff_s_inv")
            or module0_physics.get("eyring_k_s_inv")
        )
    if not isinstance(k_effective, (int, float)):
        k_effective = 0.0
    if not isinstance(horizon_s, (int, float)) or horizon_s <= 0.0:
        horizon_s = module0_physics.get("horizon_s")
    if not isinstance(horizon_s, (int, float)) or horizon_s <= 0.0:
        horizon_s = PHYSICS_PLAN_HORIZON_S
    if p_convert is None:
        if isinstance(k_effective, (int, float)) and float(k_effective) > 0.0:
            p_convert = 1.0 - math.exp(-float(k_effective) * float(horizon_s))
        else:
            shared_input = (ctx.data.get("shared_io") or {}).get("input") or {}
            condition_profile = shared_input.get("condition_profile") or {}
            bond_context = module0_job_card.get("bond_context") or {}
            if not chosen_route and not bond_context and not module0_physics:
                physics_prior = None
            else:
                bond_class = bond_context.get("bond_class") or bond_context.get("bond_type") or "unknown"
                temp_k = condition_profile.get("temperature_K")
                temp_c = condition_profile.get("temperature_C")
                if not isinstance(temp_k, (int, float)):
                    if isinstance(temp_c, (int, float)):
                        temp_k = float(temp_c) + 273.15
                    else:
                        temp_k = 298.15
                pH = condition_profile.get("pH")
                route_name = chosen_route or "unknown"
                physics_prior = compute_route_prior(
                    route_name=route_name,
                    bond_class=str(bond_class),
                    temperature_K=float(temp_k),
                    horizon_s=float(horizon_s),
                    pH=pH if isinstance(pH, (int, float)) else None,
                    ionic_strength=condition_profile.get("ionic_strength"),
                )
            fallback_k = physics_prior.get("k_effective_s_inv") if isinstance(physics_prior, dict) else None
            if isinstance(fallback_k, (int, float)):
                k_effective = float(fallback_k)
    if p_convert is None:
        p_convert = (
            module0_physics.get("prior_success_probability_final")
            or module0_physics.get("route_prior_target_specific")
            or module0_physics.get("route_prior_any_activity")
        )
    if p_convert is None and isinstance(k_effective, (int, float)):
        p_convert = 1.0 - math.exp(-float(k_effective) * float(horizon_s))
    overall_signal = None
    confidence = module0_job_card.get("confidence") or {}
    route_conf = confidence.get("route")
    target_resolution = confidence.get("target_resolution")
    wetlab_prior = confidence.get("wetlab_prior") or confidence.get("wetlab_prior_target_spec")
    if all(isinstance(val, (int, float)) for val in (route_conf, target_resolution, wetlab_prior)):
        overall_signal = float(route_conf) * float(target_resolution) * float(wetlab_prior)
    expected_signal = None
    if isinstance(overall_signal, (int, float)):
        expected_signal = min(1.0, float(overall_signal) * OVERALL_SIGNAL_SNR_SCALE)
        if overall_signal < 0.4:
            expected_signal = min(expected_signal, 0.95)
    if expected_signal is None and isinstance(p_convert, (int, float)):
        expected_signal = max(0.0, min(1.0, float(p_convert)))
    noise_floor = NOISE_FLOOR_CONVERSION
    noise_floor = max(0.01, min(0.10, float(noise_floor)))
    if expected_signal is None:
        return 0.0, {
            "horizon_s": round(float(horizon_s), 1),
            "k_effective": round(float(k_effective), 6),
            "expected_signal": None,
            "noise_floor": round(float(noise_floor), 4),
            "snr": None,
            "plan_phys": None,
            "plan_phys_note": "Phys n/a (missing physics inputs)",
            "energy_ledger_used": bool(energy_ledger),
            "plan_score_physics": None,
        }
    snr = float(expected_signal) / max(float(noise_floor), 1e-12)
    mechanism_factor = 1.0
    mechanism_status = mechanism_evidence.get("status")
    if mechanism_status == "UNVERIFIED":
        mechanism_factor = 0.8
    elif mechanism_status == "MISMATCH":
        mechanism_factor = 0.6
    route_factor = float(route_conf) if isinstance(route_conf, (int, float)) else 1.0
    plan_phys = None
    if isinstance(p_convert, (int, float)):
        plan_phys = max(0.0, min(1.0, float(p_convert) * route_factor * mechanism_factor))
    plan_score_physics = plan_phys if isinstance(plan_phys, (int, float)) else 0.0
    audit = {
        "horizon_s": round(float(horizon_s), 1),
        "k_effective": round(float(k_effective), 6),
        "expected_signal": round(float(expected_signal), 4) if expected_signal is not None else None,
        "overall_signal": round(float(overall_signal), 4) if overall_signal is not None else None,
        "snr_scale": OVERALL_SIGNAL_SNR_SCALE,
        "noise_floor": round(float(noise_floor), 4),
        "snr": round(float(snr), 3),
        "plan_phys": round(float(plan_phys), 3) if plan_phys is not None else None,
        "plan_score_physics": round(float(plan_score_physics), 3),
        "route_confidence": round(float(route_conf), 3) if isinstance(route_conf, (int, float)) else None,
        "mechanism_status": mechanism_status,
        "mechanism_factor": round(float(mechanism_factor), 3),
        "energy_ledger_used": bool(energy_ledger),
    }
    return plan_score_physics, audit


def _detectability_model(
    ctx: PipelineContext,
    protocol_card: Dict[str, Any],
) -> Dict[str, Any]:
    module2_raw = ctx.data.get("module2_active_site_refinement") or {}
    module0_job_card = ctx.data.get("job_card") or {}
    module2_physics = module2_raw.get("module2_physics_audit") or {}
    module0_physics = module0_job_card.get("physics_audit") or {}
    k_baseline = (
        module2_physics.get("k_variant_s_inv")
        or module2_physics.get("eyring_k_variant_s_inv")
        or module0_physics.get("eyring_k_s_inv")
    )
    if not isinstance(k_baseline, (int, float)):
        k_baseline = 0.0
    time_s = DETECTABILITY_TIME_S
    noise_floor = DETECTABILITY_NOISE_FLOOR
    predicted_conversion_by_arm: Dict[str, float] = {}
    snr_by_arm: Dict[str, float] = {}
    notes: List[str] = []
    for arm in protocol_card.get("arms") or []:
        arm_id = arm.get("arm_id") or "unknown"
        arm_type = arm.get("type") or "unknown"
        k_eff = float(k_baseline)
        if arm_type == "improved_conditions":
            k_eff *= 1.4
            notes.append("improved_conditions uses k_eff * 1.4")
        if arm_type == "stress_boundary":
            k_eff *= 0.7
            notes.append("stress_boundary uses k_eff * 0.7")
        if arm_type == "variant_disambiguation":
            k_eff *= 1.2
            notes.append("variant_disambiguation uses k_eff * 1.2")
        if arm_type == "negative_control":
            k_eff = 0.0
        conversion = 1.0 - math.exp(-k_eff * time_s)
        predicted_conversion_by_arm[arm_id] = round(float(conversion), 4)
        snr_by_arm[arm_id] = round(float(conversion / noise_floor), 3) if noise_floor else 0.0
    baseline_arm = next(
        (arm for arm in protocol_card.get("arms") or [] if arm.get("type") == "baseline"), None
    )
    baseline_id = baseline_arm.get("arm_id") if baseline_arm else None
    baseline_conversion = predicted_conversion_by_arm.get(baseline_id, 0.0)
    detectability_factor = 1.0
    if baseline_conversion < noise_floor:
        detectability_factor = max(0.4, baseline_conversion / noise_floor)
        notes.append("baseline conversion below noise floor; detectability penalty applied")
    return {
        "assay_time_s": round(float(time_s), 1),
        "noise_floor": round(float(noise_floor), 4),
        "predicted_conversion_by_arm": predicted_conversion_by_arm,
        "snr_by_arm": snr_by_arm,
        "detectability_factor": round(float(detectability_factor), 3),
        "notes": list(dict.fromkeys(notes)),
    }


def _improved_conditions(
    baseline: Dict[str, Any],
    retry_suggestion: Dict[str, Any],
    optimum: Dict[str, Any],
) -> Dict[str, Any]:
    proposed = retry_suggestion.get("proposed_conditions")
    if proposed:
        return {
            "pH": proposed.get("pH", baseline["pH"]),
            "temperature_c": proposed.get("temperature_c", baseline["temperature_c"]),
        }
    pH = baseline["pH"]
    temp_c = baseline["temperature_c"]
    if optimum.get("pH_opt") is not None:
        pH = pH + (float(optimum["pH_opt"]) - pH) * 0.5
    temp_c = min(37.0, temp_c + 5.0)
    return {"pH": round(float(pH), 2), "temperature_c": round(float(temp_c), 1)}


def _variant_disambiguation(
    best_variant: Dict[str, Any],
    variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    second = None
    if variants:
        sorted_variants = sorted(
            variants, key=lambda item: item.get("rank", 999)
        )
        for variant in sorted_variants:
            if variant.get("variant_id") != best_variant.get("variant_id"):
                second = variant
                break
    if not second:
        second = _find_variant(variants, "V0") or {"variant_id": "V0", "label": "Baseline"}
    return {"primary": best_variant, "compare_to": second}


def _mechanism_disambiguation(
    best_variant: Dict[str, Any],
    variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    serine_variant = None
    cysteine_variant = None
    for variant in variants:
        if variant.get("category") == "mechanism_alignment":
            serine_variant = variant
            break
    if serine_variant is None:
        serine_variant = best_variant
    for variant in variants:
        if variant.get("category") != "mechanism_alignment":
            cysteine_variant = variant
            break
    if cysteine_variant is None:
        cysteine_variant = _find_variant(variants, "V0") or best_variant
    return {
        "primary": serine_variant,
        "compare_to": cysteine_variant,
        "purpose": "mechanism_disambiguation",
    }


def _stress_conditions(ctx: PipelineContext, baseline: Dict[str, Any]) -> Dict[str, Any]:
    constraints = (ctx.data.get("job_card") or {}).get("constraints") or {}
    pH_min = constraints.get("ph_min")
    pH_max = constraints.get("ph_max")
    if pH_min is not None or pH_max is not None:
        target = pH_min if pH_min is not None else pH_max
        if pH_min is not None and pH_max is not None:
            base = baseline.get("pH", 7.0)
            if abs(base - pH_min) > abs(base - pH_max):
                target = pH_min
            else:
                target = pH_max
        return {"pH": round(float(target), 2), "temperature_c": baseline["temperature_c"]}
    return {
        "pH": baseline["pH"],
        "temperature_c": max(10.0, baseline["temperature_c"] - 8.0),
    }


def _candidate_arms(
    baseline_conditions: Dict[str, Any],
    improved_conditions: Dict[str, Any],
    stress_conditions: Dict[str, Any],
    best_variant: Dict[str, Any],
    disambiguation_candidate: Dict[str, Any],
    mechanism_disambiguation: Optional[Dict[str, Any]],
    variants: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = [
        {
            "type": "baseline",
            "candidate": best_variant,
            "conditions": baseline_conditions,
            "expected_outcome": "Baseline activity under user conditions.",
            "why": "Establish baseline conversion with selected variant.",
        },
        {
            "type": "improved_conditions",
            "candidate": best_variant,
            "conditions": improved_conditions,
            "expected_outcome": "Improved activity vs baseline.",
            "why": "Test condition shift predicted to increase performance.",
        },
        {
            "type": "variant_disambiguation",
            "candidate": disambiguation_candidate,
            "conditions": baseline_conditions,
            "expected_outcome": "Resolve variant performance ranking.",
            "why": "Directly compare top variants under identical conditions.",
        },
        {
            "type": "negative_control",
            "candidate": {"control": "no_enzyme"},
            "conditions": baseline_conditions,
            "expected_outcome": "No conversion beyond background.",
            "why": "Confirm activity is enzyme-dependent.",
        },
        {
            "type": "stress_boundary",
            "candidate": best_variant,
            "conditions": stress_conditions,
            "expected_outcome": "Lower conversion or instability signal.",
            "why": "Stress conditions to map boundary of activity.",
        },
    ]
    if mechanism_disambiguation:
        candidates.append(
            {
                "type": "mechanism_disambiguation",
                "candidate": mechanism_disambiguation,
                "conditions": baseline_conditions,
                "expected_outcome": "Disambiguate serine vs cysteine mechanism.",
                "why": "Compare mechanism-aligned variants under identical conditions.",
            }
        )
    alt_variant = _find_second_variant(variants, best_variant)
    if alt_variant:
        candidates.append(
            {
                "type": "variant_alt_baseline",
                "candidate": alt_variant,
                "conditions": baseline_conditions,
                "expected_outcome": "Alternate variant baseline activity.",
                "why": "Evaluate alternate variant under baseline conditions.",
            }
        )
    return candidates


def _find_second_variant(
    variants: List[Dict[str, Any]], best_variant: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not variants:
        return None
    sorted_variants = sorted(variants, key=lambda item: item.get("rank", 999))
    for variant in sorted_variants:
        if variant.get("variant_id") != best_variant.get("variant_id"):
            return variant
    return None


def _has_strong_variant(variants: List[Dict[str, Any]]) -> bool:
    if len(variants) < 2:
        return False
    sorted_variants = sorted(variants, key=lambda item: item.get("rank", 999))
    top = sorted_variants[0]
    second = sorted_variants[1]
    try:
        diff = abs(float(top.get("score", 0.0)) - float(second.get("score", 0.0)))
    except (TypeError, ValueError):
        diff = 0.0
    return diff <= 0.05


def _select_arms_by_eig(
    ctx: PipelineContext,
    candidates: List[Dict[str, Any]],
    baseline_conditions: Dict[str, Any],
    require_disambiguation: bool,
    require_mechanism_disambiguation: bool,
) -> List[Dict[str, Any]]:
    hypotheses = [
        "conditions_limited",
        "access_limited",
        "retention_limited",
        "mechanism_mismatch",
    ]
    module1 = ctx.data.get("module1_topogate") or {}
    module1_conf = module1.get("module1_confidence") or {}
    module2 = ctx.data.get("module2_active_site_refinement") or {}
    module0 = ctx.data.get("module0_strategy_router") or {}
    hypothesis_priors = _estimate_hypothesis_priors(
        module1_confidence=module1_conf,
        mechanism_mismatch=module2.get("mechanism_check") or {},
        conditions_effect=module0.get("given_conditions_effect") or {},
    )
    variant_predictions = {
        str(variant.get("variant_id")): (
            variant.get("physics_prediction")
            if isinstance(variant.get("physics_prediction"), dict)
            else {}
        )
        for variant in (module2.get("variant_set") or [])
    }
    base_eig_by_arm_id: Dict[int, float] = {}
    for candidate in candidates:
        eig, success_prob, n_eff, entropy_prior = _arm_eig(
            ctx, candidate, baseline_conditions
        )
        candidate["eig"] = eig
        candidate["success_prob"] = success_prob
        candidate["n_eff"] = n_eff
        candidate["prior_entropy"] = entropy_prior
        base_eig_by_arm_id[int(id(candidate))] = float(eig)

    weighted = _compute_hypothesis_weighted_eig(
        arms=candidates,
        hypotheses=hypotheses,
        hypothesis_priors=hypothesis_priors,
        base_eig_by_arm_id=base_eig_by_arm_id,
        variant_predictions=variant_predictions,
    )
    weighted_by_arm = {int(id(candidate)): result for candidate, result in zip(candidates, weighted)}
    for candidate in candidates:
        payload = weighted_by_arm.get(int(id(candidate))) or {}
        if isinstance(payload.get("eig"), (int, float)):
            candidate["eig"] = float(payload["eig"])
        candidate["eig_by_hypothesis"] = payload.get("eig_by_hypothesis", {})
        candidate["most_informative_for"] = payload.get("most_informative_for")

    required_types = {"baseline", "negative_control", "stress_boundary"}
    if require_disambiguation:
        required_types.add("variant_disambiguation")
    if require_mechanism_disambiguation:
        required_types.add("mechanism_disambiguation")

    selected: List[Dict[str, Any]] = []
    for required_type in required_types:
        typed = [item for item in candidates if item.get("type") == required_type]
        if typed:
            selected.append(max(typed, key=lambda item: item.get("eig", 0.0)))

    remaining = [item for item in candidates if item not in selected]
    remaining.sort(key=lambda item: item.get("eig", 0.0), reverse=True)
    while len(selected) < 5 and remaining:
        selected.append(remaining.pop(0))
    return selected[:5]


def _assign_arm_ids(arms: List[Dict[str, Any]]) -> None:
    preferred = {
        "baseline": "A1",
        "improved_conditions": "A2",
        "variant_disambiguation": "A3",
        "mechanism_disambiguation": "A3",
        "negative_control": "A4",
        "stress_boundary": "A5",
    }
    used: Dict[str, Dict[str, Any]] = {}
    available = ["A1", "A2", "A3", "A4", "A5"]
    for arm in arms:
        arm_type = arm.get("type")
        arm_id = preferred.get(arm_type)
        if arm_id and arm_id in available:
            arm["arm_id"] = arm_id
            available.remove(arm_id)
            used[arm_id] = arm
    for arm in arms:
        if arm.get("arm_id"):
            continue
        if available:
            arm["arm_id"] = available.pop(0)


def _arm_eig(
    ctx: PipelineContext,
    arm: Dict[str, Any],
    baseline_conditions: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    success_prob = _predict_success_prob(ctx, arm, baseline_conditions)
    n_eff = _n_eff_from_context(ctx)
    eig, prior_entropy = _expected_information_gain(success_prob, n_eff)
    return eig, success_prob, n_eff, prior_entropy


def _expected_information_gain(success_prob: float, n_eff: float) -> Tuple[float, float]:
    alpha = max(1e-6, success_prob * n_eff)
    beta = max(1e-6, (1.0 - success_prob) * n_eff)
    prior_entropy = beta_entropy(alpha, beta)
    post_success = beta_entropy(alpha + 1.0, beta)
    post_fail = beta_entropy(alpha, beta + 1.0)
    expected_post = success_prob * post_success + (1.0 - success_prob) * post_fail
    eig = max(0.0, prior_entropy - expected_post)
    return eig, prior_entropy


def _predict_success_prob(
    ctx: PipelineContext,
    arm: Dict[str, Any],
    baseline_conditions: Dict[str, Any],
) -> float:
    candidate = arm.get("candidate") or {}
    if candidate.get("control"):
        return 0.02

    module1 = ctx.data.get("module1_topogate") or {}
    module1_conf = module1.get("module1_confidence") or {}
    retention = module1_conf.get("retention")
    if not isinstance(retention, (int, float)):
        retention = 0.5

    module2 = ctx.data.get("module2_active_site_refinement") or {}
    selected_scaffold = module2.get("selected_scaffold") or {}
    model_risk = selected_scaffold.get("model_risk")
    if not isinstance(model_risk, (int, float)):
        model_risk = 0.2

    k_pred_mean = _arm_k_pred_mean(ctx, arm, baseline_conditions)
    log_k = math.log(max(1e-8, k_pred_mean))

    score = 0.35 * log_k + 1.2 * float(retention) - 0.9 * float(model_risk) - 0.3
    probability = sigmoid(score)
    return max(0.02, min(0.98, float(probability)))


def _build_scorecard_module3(
    information_gain: Dict[str, Any],
    physics_audit: Dict[str, Any],
    math_contract: Dict[str, Any],
) -> Dict[str, Any]:
    plan_score = information_gain.get("plan_score")
    plan_phys = physics_audit.get("plan_phys")
    confidence = (math_contract.get("confidence") or {})
    n_eff = confidence.get("n_eff")
    features = {
        "plan_score": plan_score,
        "plan_phys": plan_phys,
        "snr": physics_audit.get("snr"),
    }
    contributors = contributors_from_features(features, limit=5)
    metrics = [
        ScoreCardMetric(
            name="plan_score",
            raw=float(plan_score) if isinstance(plan_score, (int, float)) else None,
            calibrated=float(plan_score) if isinstance(plan_score, (int, float)) else None,
            ci90=confidence.get("ci90") if isinstance(confidence.get("ci90"), (list, tuple)) else None,
            n_eff=float(n_eff) if isinstance(n_eff, (int, float)) else None,
            status=metric_status(plan_score, n_eff),
            definition="Experiment plan score combining EIG and physics detectability.",
            contributors=contributors,
        ),
        ScoreCardMetric(
            name="plan_phys",
            raw=float(plan_phys) if isinstance(plan_phys, (int, float)) else None,
            calibrated=float(plan_phys) if isinstance(plan_phys, (int, float)) else None,
            ci90=None,
            n_eff=None,
            status=metric_status(plan_phys, None),
            definition="Physics-derived detectability score for the planned arms.",
            contributors=contributors,
        ),
    ]
    return ScoreCard(module_id=3, metrics=metrics, calibration_status="heuristic").to_dict()


def _build_score_ledger_module3(
    information_gain: Dict[str, Any],
    physics_audit: Dict[str, Any],
    detectability: Dict[str, Any],
) -> Dict[str, Any]:
    def _as_float(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        return None

    plan_score = information_gain.get("plan_score")
    plan_phys = physics_audit.get("plan_phys")
    terms = [
        ScoreTerm(
            name="plan_score",
            value=_as_float(plan_score),
            unit="probability",
            formula="0.5*EIG + 0.5*plan_phys (detectability-adjusted)",
            inputs={
                "plan_score_eig": information_gain.get("plan_score_eig"),
                "plan_score_physics": information_gain.get("plan_score_physics"),
            },
            notes="Plan score combining information gain and physics detectability.",
        ),
        ScoreTerm(
            name="plan_phys",
            value=_as_float(plan_phys),
            unit="probability",
            formula="p_success_horizon * route_confidence * mechanism_factor",
            inputs={
                "expected_signal": physics_audit.get("expected_signal"),
                "noise_floor": physics_audit.get("noise_floor"),
                "route_confidence": physics_audit.get("route_confidence"),
                "mechanism_status": physics_audit.get("mechanism_status"),
            },
            notes="Physics-derived detectability score for planned arms.",
        ),
        ScoreTerm(
            name="expected_signal",
            value=_as_float(physics_audit.get("expected_signal")),
            unit="fraction",
            formula="1 - exp(-k_eff * horizon_s)",
            inputs={
                "k_eff_s_inv": physics_audit.get("k_effective"),
                "horizon_s": physics_audit.get("horizon_s"),
            },
            notes="Expected conversion proxy based on effective rate.",
        ),
        ScoreTerm(
            name="snr",
            value=_as_float(physics_audit.get("snr")),
            unit="ratio",
            formula="expected_signal / noise_floor",
            inputs={"noise_floor": physics_audit.get("noise_floor")},
            notes="Signal-to-noise estimate for detectability.",
        ),
    ]
    return ScoreLedger(module_id=3, terms=terms).to_dict()


def _arm_k_pred_mean(
    ctx: PipelineContext,
    arm: Dict[str, Any],
    baseline_conditions: Dict[str, Any],
) -> float:
    module2 = ctx.data.get("module2_active_site_refinement") or {}
    selected_scaffold = module2.get("selected_scaffold") or {}
    base_k = selected_scaffold.get("k_pred_mean") or selected_scaffold.get("k_pred") or 1.0
    try:
        base_k = float(base_k)
    except (TypeError, ValueError):
        base_k = 1.0
    optimum = module2.get("optimum_conditions_estimate") or {}
    conditions = arm.get("conditions") or baseline_conditions
    multiplier = _condition_multiplier(conditions, baseline_conditions, optimum)
    return max(1e-8, base_k * multiplier)


def _condition_multiplier(
    conditions: Dict[str, Any],
    baseline: Dict[str, Any],
    optimum: Dict[str, Any],
) -> float:
    base_score = _condition_score(baseline, optimum)
    arm_score = _condition_score(conditions, optimum)
    if base_score <= 0:
        return 1.0
    ratio = arm_score / base_score
    return max(0.7, min(1.3, ratio))


def _condition_score(conditions: Dict[str, Any], optimum: Dict[str, Any]) -> float:
    pH = conditions.get("pH")
    temp_c = conditions.get("temperature_c")
    opt_pH = optimum.get("pH_opt")
    opt_temp = optimum.get("T_opt_C")
    if opt_pH is None and optimum.get("pH_range"):
        opt_pH = sum(optimum["pH_range"]) / 2.0
    if opt_temp is None and optimum.get("temperature_c"):
        opt_temp = sum(optimum["temperature_c"]) / 2.0
    score = 1.0
    if pH is not None and opt_pH is not None:
        score *= math.exp(-0.4 * abs(float(pH) - float(opt_pH)))
    if temp_c is not None and opt_temp is not None:
        score *= math.exp(-0.03 * abs(float(temp_c) - float(opt_temp)))
    return max(0.05, min(1.0, score))


def _n_eff_from_context(ctx: PipelineContext) -> float:
    job_card = ctx.data.get("job_card") or {}
    conf_route = (job_card.get("confidence") or {}).get("route")
    module2 = ctx.data.get("module2_active_site_refinement") or {}
    conf_module2 = module2.get("confidence_calibrated")
    scores = [
        float(value)
        for value in [conf_route, conf_module2]
        if isinstance(value, (int, float))
    ]
    confidence = sum(scores) / len(scores) if scores else 0.5
    return BayesianDAGRouter.effective_sample_size(confidence)


def _qc_rules(status: str) -> List[Dict[str, Any]]:
    return [
        {
            "id": "NEG_CONTROL",
            "description": "Negative control conversion <= 0.05",
            "status": status,
        },
        {
            "id": "BASELINE_VARIANCE",
            "description": "Baseline stdev within noise floor",
            "status": status,
        },
        {
            "id": "SANITY_MARGIN",
            "description": "Baseline >= negative control + 0.1",
            "status": status,
        },
        {
            "id": "LOW_INFO",
            "description": "Outcome entropy sufficient for learning",
            "status": status,
        },
    ]


def _qc_rules_from_result(qc_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules = _qc_rules(status="PASS")
    if not qc_result:
        return _qc_rules(status="n/a")
    reasons = qc_result.get("reasons") or []
    warnings = qc_result.get("warnings") or []
    for rule in rules:
        rule_id = rule["id"]
        if rule_id == "NEG_CONTROL" and any(
            "negative control" in reason for reason in reasons
        ):
            rule["status"] = "FAIL"
        if rule_id == "BASELINE_VARIANCE" and any(
            "baseline variance" in reason or "baseline stdev" in reason for reason in reasons
        ):
            rule["status"] = "FAIL"
        if rule_id == "SANITY_MARGIN" and any(
            "baseline below negative control" in reason for reason in reasons
        ):
            rule["status"] = "FAIL"
        if rule_id == "LOW_INFO" and any("LOW_INFO" in warn for warn in warnings):
            rule["status"] = "WARN"
    return rules


def _find_variant(variants: List[Dict[str, Any]], variant_id: str) -> Optional[Dict[str, Any]]:
    for variant in variants:
        if variant.get("variant_id") == variant_id:
            return variant
    return None


def _candidate_variant_id(candidate: Dict[str, Any]) -> Optional[str]:
    if candidate.get("variant_id"):
        return candidate.get("variant_id")
    primary = candidate.get("primary") or {}
    return primary.get("variant_id")


def _estimate_hypothesis_priors(
    module1_confidence: Dict[str, Any],
    mechanism_mismatch: Dict[str, Any],
    conditions_effect: Dict[str, Any],
) -> Dict[str, float]:
    priors: Dict[str, float] = {}
    condition_score = conditions_effect.get("condition_score")
    if not isinstance(condition_score, (int, float)):
        condition_score = 0.5
    priors["conditions_limited"] = max(0.1, min(0.9, 1.0 - float(condition_score)))

    access = module1_confidence.get("access")
    if not isinstance(access, (int, float)):
        access = module1_confidence.get("calibrated_probability")
    if not isinstance(access, (int, float)):
        access = 0.5
    priors["access_limited"] = max(0.1, min(0.9, 1.0 - float(access)))

    retention = module1_confidence.get("retention")
    if not isinstance(retention, (int, float)):
        retention = 0.5
    priors["retention_limited"] = max(0.1, min(0.9, 1.0 - float(retention)))

    status = str(mechanism_mismatch.get("status") or "UNVERIFIED").upper()
    if status == "VERIFIED":
        priors["mechanism_mismatch"] = 0.1
    elif status == "REJECTED":
        priors["mechanism_mismatch"] = 0.8
    else:
        priors["mechanism_mismatch"] = 0.4

    total = sum(priors.values())
    if total > 1.0:
        priors = {key: float(value) / total for key, value in priors.items()}
    return priors


def _compute_hypothesis_weighted_eig(
    arms: List[Dict[str, Any]],
    hypotheses: List[str],
    hypothesis_priors: Dict[str, float],
    base_eig_by_arm_id: Dict[int, float],
    variant_predictions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    uncertainty_by_h: Dict[str, float] = {}
    for hypothesis in hypotheses:
        p = max(1e-6, min(1.0 - 1e-6, float(hypothesis_priors.get(hypothesis, 0.25))))
        uncertainty_by_h[hypothesis] = -(
            p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)
        )

    weighted: List[Dict[str, Any]] = []
    for arm in arms:
        arm_type = str(arm.get("type") or "unknown")
        arm_key = int(id(arm))
        base_eig = float(base_eig_by_arm_id.get(arm_key, 0.0))
        per_h: Dict[str, float] = {}
        total = 0.0
        for hypothesis in hypotheses:
            relevance = float(
                (HYPOTHESIS_ARM_RELEVANCE.get(hypothesis) or {}).get(arm_type, 0.2)
            )
            boost = 1.0
            if arm_type == "variant_disambiguation":
                variant_id = _candidate_variant_id(arm.get("candidate") or {})
                pred = (variant_predictions or {}).get(str(variant_id))
                if isinstance(pred, dict):
                    if (
                        hypothesis == "retention_limited"
                        and float(pred.get("retention_boost") or 0.0) > 0.05
                    ):
                        boost = 1.5
                    if (
                        hypothesis == "access_limited"
                        and float(pred.get("ddg_kj_mol") or 0.0) < -2.0
                    ):
                        boost = max(boost, 1.3)
            h_eig = base_eig * relevance * uncertainty_by_h.get(hypothesis, 0.0) * boost
            per_h[hypothesis] = round(float(h_eig), 6)
            total += h_eig

        eig = total / max(1, len(hypotheses))
        weighted.append(
            {
                "arm_id": arm.get("arm_id"),
                "type": arm_type,
                "eig": round(float(eig), 6),
                "eig_by_hypothesis": per_h,
                "most_informative_for": max(per_h, key=per_h.get) if per_h else None,
            }
        )
    return weighted


def _compute_information_gain(
    ctx: PipelineContext,
    protocol_card: Dict[str, Any],
    baseline_conditions: Dict[str, Any],
) -> Dict[str, Any]:
    hypotheses = [
        "conditions_limited",
        "access_limited",
        "retention_limited",
        "mechanism_mismatch",
    ]
    arms = protocol_card.get("arms", [])
    notes: List[str] = []
    eig_per_arm: List[Dict[str, Any]] = []
    total_eig = 0.0
    base_eig_by_arm_id: Dict[int, float] = {}
    arm_metrics: Dict[int, Dict[str, Any]] = {}
    for arm in arms:
        eig, success_prob, n_eff, entropy_prior = _arm_eig(
            ctx, arm, baseline_conditions
        )
        key = int(id(arm))
        base_eig_by_arm_id[key] = float(eig)
        arm_metrics[key] = {
            "success_prob": round(success_prob, 3),
            "n_eff": round(n_eff, 2),
            "prior_entropy": round(entropy_prior, 4),
        }

    module1 = ctx.data.get("module1_topogate") or {}
    module1_conf = module1.get("module1_confidence") or {}
    module2 = ctx.data.get("module2_active_site_refinement") or {}
    module0 = ctx.data.get("module0_strategy_router") or {}
    hypothesis_priors = _estimate_hypothesis_priors(
        module1_confidence=module1_conf,
        mechanism_mismatch=module2.get("mechanism_check") or {},
        conditions_effect=module0.get("given_conditions_effect") or {},
    )
    variant_predictions = {
        str(variant.get("variant_id")): (
            variant.get("physics_prediction")
            if isinstance(variant.get("physics_prediction"), dict)
            else {}
        )
        for variant in (module2.get("variant_set") or [])
    }
    weighted_eig = _compute_hypothesis_weighted_eig(
        arms=arms,
        hypotheses=hypotheses,
        hypothesis_priors=hypothesis_priors,
        base_eig_by_arm_id=base_eig_by_arm_id,
        variant_predictions=variant_predictions,
    )
    weighted_by_key = {
        int(id(arm)): payload for arm, payload in zip(arms, weighted_eig)
    }

    for arm in arms:
        key = int(id(arm))
        weighted = weighted_by_key.get(key) or {}
        metrics = arm_metrics.get(key) or {}
        eig_value = float(weighted.get("eig") or 0.0)
        eig_per_arm.append(
            {
                "arm_id": arm.get("arm_id"),
                "type": arm.get("type"),
                "eig": round(eig_value, 6),
                "success_prob": metrics.get("success_prob"),
                "n_eff": metrics.get("n_eff"),
                "prior_entropy": metrics.get("prior_entropy"),
                "eig_by_hypothesis": weighted.get("eig_by_hypothesis", {}),
                "most_informative_for": weighted.get("most_informative_for"),
            }
        )
        total_eig += eig_value
    plan_score = 1.0 - math.exp(-total_eig * 1.5)
    plan_score = max(0.0, min(1.0, plan_score))
    notes.append(f"total_eig={round(total_eig, 6)}")
    notes.append(f"hypothesis_priors={hypothesis_priors}")
    return {
        "hypotheses": hypotheses,
        "hypothesis_priors": hypothesis_priors,
        "plan_score": round(plan_score, 3),
        "notes": notes,
        "eig_per_arm": eig_per_arm,
    }


def _predicted_under_given_conditions(protocol_card: Dict[str, Any]) -> Dict[str, Any]:
    baseline_arm = _find_arm(protocol_card.get("arms") or [], "baseline")
    if not baseline_arm:
        return {}
    return {
        "expected_outcome": baseline_arm.get("expected_outcome"),
        "conditions": baseline_arm.get("conditions"),
    }


def _build_evidence_record(
    protocol_card: Dict[str, Any], information_gain: Dict[str, Any]
) -> Dict[str, Any]:
    plan_score = float(information_gain.get("plan_score") or 0.0)
    features = FeatureVector(
        values={
            "plan_score": plan_score,
            "eig_total": sum(
                arm.get("eig", 0.0) for arm in information_gain.get("eig_per_arm", [])
            ),
        },
        missing=[],
        source="module3",
    )
    evidence = EvidenceRecord(
        module_id=3,
        inputs={
            "batch_id": protocol_card.get("batch_id"),
            "arm_types": [arm.get("type") for arm in protocol_card.get("arms") or []],
        },
        features_used=features,
        score=plan_score,
        confidence=plan_score,
        uncertainty={},
        diagnostics={
            "eig_per_arm": information_gain.get("eig_per_arm", []),
        },
    )
    return evidence.to_dict()


def _ingest_wetlab_results(
    ctx: PipelineContext,
    protocol_card: Dict[str, Any],
    wetlab_results: Dict[str, Any],
    baseline_conditions: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    protocol_batch = protocol_card.get("batch_id")
    results_batch = wetlab_results.get("batch_id")
    reasons: List[str] = []
    warnings: List[str] = []

    arm_results = {arm.get("arm_id"): arm for arm in wetlab_results.get("arms") or []}
    summaries: List[Dict[str, Any]] = []
    summary_by_id: Dict[str, Dict[str, Any]] = {}
    protocol_arms = protocol_card.get("arms", [])
    protocol_arm_by_id = {arm.get("arm_id"): arm for arm in protocol_arms}
    for arm in protocol_arms:
        arm_id = arm.get("arm_id")
        result_arm = arm_results.get(arm_id) or {}
        summary = _summarize_arm(result_arm.get("observations") or [])
        summary["arm_id"] = arm_id
        summaries.append(summary)
        summary_by_id[arm_id] = summary

    if protocol_batch != results_batch:
        reasons.append("batch_id mismatch")

    baseline_id = _find_arm_id(protocol_card.get("arms") or [], "baseline") or "A1"
    negative_id = (protocol_card.get("controls") or {}).get("negative_control_arm_id")
    if not negative_id:
        negative_id = _find_arm_id(protocol_card.get("arms") or [], "negative_control")

    qc_records: List[Dict[str, Any]] = []
    for arm in protocol_arms:
        arm_id = arm.get("arm_id")
        summary = summary_by_id.get(arm_id) or {}
        qc_records.append(
            {
                "arm_id": arm_id,
                "arm_type": arm.get("type"),
                "mean_conversion": summary.get("mean_conversion"),
                "stdev_conversion": summary.get("stdev_conversion"),
                "n": summary.get("n", 0),
            }
        )

    qc_eval = _qc_check(qc_records)
    if not qc_eval.get("ok", True):
        reasons.extend(qc_eval.get("reasons") or [])

    baseline = summary_by_id.get(baseline_id) or {"n": 0, "mean_conversion": None}
    negative = summary_by_id.get(negative_id) or {"n": 0, "mean_conversion": None}

    noise_floor = 0.1
    if (baseline.get("stdev_conversion") is not None) and (
        baseline.get("stdev_conversion") > noise_floor
    ):
        warnings.append("QC_WARN_NOISE_FLOOR")

    baseline_mean = baseline.get("mean_conversion")
    negative_mean = negative.get("mean_conversion")
    if baseline_mean is None or negative_mean is None:
        reasons.append("insufficient data for baseline vs negative control margin")
    elif baseline_mean < (negative_mean + 0.1):
        reasons.append("baseline below negative control margin 0.1")

    outcomes = []
    for arm_id, summary in summary_by_id.items():
        arm_def = protocol_arm_by_id.get(arm_id) or {}
        if arm_def.get("type") == "negative_control":
            continue
        mean_conv = summary.get("mean_conversion")
        if mean_conv is None:
            continue
        outcomes.append(1.0 if mean_conv >= 0.2 else 0.0)
    outcome_entropy = bernoulli_entropy(
        sum(outcomes) / len(outcomes)
    ) if len(outcomes) >= 2 else None
    if outcome_entropy is not None and outcome_entropy < 0.25:
        warnings.append("QC_WARN_LOW_INFO")

    qc_status = "FAIL" if reasons else "PASS"
    router_updates: List[Dict[str, Any]] = []
    router = getattr(ctx, "bayes_router", None) or getattr(ctx, "router", None)

    job_card = ctx.data.get("job_card") or {}
    route = (
        job_card.get("chosen_route")
        or (job_card.get("mechanism_route") or {}).get("primary")
        or "unknown"
    )
    matched_bins = job_card.get("matched_bins") or {}
    substrate_bin = matched_bins.get("substrate_bin") or "unknown"
    catalyst_bin = matched_bins.get("catalyst_family_bin") or "unknown"

    global_controls = wetlab_results.get("global_controls") or {}
    measured_pH = global_controls.get("ph_measured")
    measured_temp_c = global_controls.get("temp_measured_c")
    condition_profile = ConditionProfile(
        pH=measured_pH if measured_pH is not None else baseline_conditions.get("pH"),
        temperature_C=measured_temp_c
        if measured_temp_c is not None
        else baseline_conditions.get("temperature_c"),
    )

    if measured_pH is None and measured_temp_c is None and matched_bins.get("condition_bin"):
        condition_bin = matched_bins.get("condition_bin")
    else:
        bin_router = router or BayesianDAGRouter()
        condition_bin = bin_router._condition_bin(condition_profile)

    total_samples = sum(summary.get("n", 0) for summary in summaries)
    weight = 0.5 if total_samples <= 5 else 1.0
    if (
        baseline.get("stdev_conversion") is not None
        and baseline.get("stdev_conversion") > noise_floor
        and baseline.get("stdev_conversion") <= 0.15
    ):
        weight = min(weight, 0.5)

    scaffold_id = (ctx.data.get("module2_active_site_refinement") or {}).get(
        "selected_scaffold", {}
    ).get("scaffold_id")
    experiment_records: List[ExperimentRecord] = []
    for arm_id, result_arm in arm_results.items():
        summary = summary_by_id.get(arm_id) or {}
        arm_def = protocol_arm_by_id.get(arm_id) or {}
        arm_type = arm_def.get("type")
        candidate = arm_def.get("candidate") or {}
        variant_id = _candidate_variant_id(candidate)

        observed_success = 0.0
        if arm_type != "negative_control":
            mean_conv = summary.get("mean_conversion")
            if mean_conv is not None and mean_conv >= 0.2:
                observed_success = 1.0

        metadata = {
            "arm_id": arm_id,
            "arm_type": arm_type,
            "scaffold_id": scaffold_id,
            "variant_id": variant_id,
            "condition_bin": condition_bin,
            "measured_conditions": {
                "pH": measured_pH,
                "temperature_c": measured_temp_c,
            }
            if measured_pH is not None or measured_temp_c is not None
            else None,
        }
        record = ExperimentRecord(
            reaction_task_fingerprint=f"{ctx.smiles}|{ctx.target_bond}",
            condition_profile=condition_profile,
            candidate_fingerprint=f"{scaffold_id or 'unknown'}:{variant_id or 'unknown'}",
            observed_success=observed_success,
            observed_rate_or_yield=summary.get("mean_conversion"),
            notes=f"arm_id={arm_id}",
            source_quality=1.0,
            route=route,
            substrate_bin=substrate_bin,
            catalyst_family=catalyst_bin,
            metadata=metadata,
            weight=weight,
        )
        experiment_records.append(record)

    if qc_status == "PASS":
        learning_status, router_updates = apply_learning_update(ctx, experiment_records)
        records_to_write = len(experiment_records)
    else:
        learning_status = _reject_status(reasons)
        records_to_write = 0

    qc_result = {
        "status": qc_status,
        "reasons": reasons,
        "warnings": warnings,
        "outcome_entropy": outcome_entropy,
    }
    wetlab_ingest_summary = {
        "batch_id": results_batch,
        "arm_summaries": summaries,
        "global_controls": global_controls,
    }
    learning_update = {
        "status": learning_status,
        "records_to_write": records_to_write,
        "router_updates": router_updates,
    }
    return wetlab_ingest_summary, qc_result, learning_update


def _qc_check(experiment_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate negative control and baseline noise before learning updates."""
    reasons: List[str] = []
    negative = [rec for rec in experiment_records if rec.get("arm_type") == "negative_control"]
    if not negative:
        reasons.append("negative control missing")
    else:
        for rec in negative:
            mean_conv = rec.get("mean_conversion")
            if isinstance(mean_conv, (int, float)) and mean_conv > 0.05:
                reasons.append("negative control conversion > 0.05")
                break

    baseline = [rec for rec in experiment_records if rec.get("arm_type") == "baseline"]
    if not baseline:
        reasons.append("baseline missing")
    else:
        for rec in baseline:
            n = rec.get("n", 0)
            if n < 2:
                reasons.append("baseline n < 2")
                break
        for rec in baseline:
            stdev = rec.get("stdev_conversion")
            if isinstance(stdev, (int, float)) and stdev > 0.15:
                reasons.append("baseline stdev > 0.15")
                break

    return {"ok": not reasons, "reasons": reasons}


def apply_learning_update(
    ctx: PipelineContext, experiment_records: List[ExperimentRecord]
) -> Tuple[str, List[Dict[str, Any]]]:
    router = getattr(ctx, "bayes_router", None) or getattr(ctx, "router", None)
    updates: List[Dict[str, Any]] = []
    for record in experiment_records:
        route = getattr(record, "route", None) or "unknown"
        substrate_bin = getattr(record, "substrate_bin", None) or "unknown"
        catalyst_bin = getattr(record, "catalyst_family", None) or "unknown"
        metadata = getattr(record, "metadata", None) or {}
        condition_bin = metadata.get("condition_bin") or "ph_unknown|temp_unknown"
        weight = float(getattr(record, "weight", 1.0) or 1.0)
        success_delta = weight if getattr(record, "observed_success", 0.0) >= 0.5 else 0.0
        fail_delta = weight if getattr(record, "observed_success", 0.0) < 0.5 else 0.0
        if router:
            router.observe_weighted(
                route,
                condition_bin,
                substrate_bin,
                catalyst_bin,
                success_delta,
                fail_delta,
            )
        updates.append(
            {
                "bucket": [route, condition_bin, substrate_bin, catalyst_bin],
                "delta_success": success_delta,
                "delta_fail": fail_delta,
                "weight": weight,
            }
        )
    status = "APPLIED" if router else "SKIPPED_NO_ROUTER"
    return status, updates


def _summarize_arm(observations: List[Dict[str, Any]]) -> Dict[str, Any]:
    values: List[float] = []
    for obs in observations:
        metric = obs.get("metric")
        if isinstance(metric, str) and metric.lower() == "conversion":
            value = obs.get("value")
            if isinstance(value, (int, float)):
                values.append(float(value))
    n = len(values)
    if n:
        mean = statistics.mean(values)
    else:
        mean = None
    stdev = statistics.stdev(values) if n >= 2 else None
    return {"mean_conversion": mean, "stdev_conversion": stdev, "n": n}


def _find_arm(arms: List[Dict[str, Any]], arm_type: str) -> Optional[Dict[str, Any]]:
    for arm in arms:
        if arm.get("type") == arm_type:
            return arm
    return None


def _find_arm_id(arms: List[Dict[str, Any]], arm_type: str) -> Optional[str]:
    for arm in arms:
        if arm.get("type") == arm_type:
            return arm.get("arm_id")
    return None


def _merge_shared_io(ctx: PipelineContext, module3_result: Dict[str, Any]) -> Dict[str, Any]:
    shared = ctx.data.get("shared_io") or {}
    shared_input = shared.get("input") or {}
    telemetry = shared_input.get("telemetry") or {}
    trace = telemetry.get("trace") or []
    if "module3" not in trace:
        trace.append("module3")
    telemetry["trace"] = trace
    shared_input["telemetry"] = telemetry

    predicted = module3_result.get("predicted_under_given_conditions") or {}
    output = SharedOutput(
        result={
            "status": module3_result.get("status"),
            "batch_id": (module3_result.get("protocol_card") or {}).get("batch_id"),
            "qc_status": (module3_result.get("qc_result") or {}).get("status"),
        },
        given_conditions_effect=predicted,
        optimum_conditions={},
        confidence={
            "calibrated_probability": (module3_result.get("information_gain") or {}).get(
                "plan_score"
            ),
        },
        retry_loop_suggestion={"status": "not_applicable"},
    )
    outputs = dict(shared.get("outputs", {}))
    outputs["module3"] = output.to_dict()
    payload = dict(shared) if shared else {}
    payload["input"] = shared_input
    payload["outputs"] = outputs
    energy_ledger = payload.get("energy_ledger") or {}
    energy_update = module3_result.get("energy_ledger_update")
    if isinstance(energy_update, dict):
        energy_ledger = dict(energy_ledger)
        energy_ledger.update({key: value for key, value in energy_update.items() if value is not None})
    payload["energy_ledger"] = energy_ledger
    return payload


def _update_unity_record_parts(ctx: PipelineContext, module_output: Dict[str, Any]) -> None:
    parts = ctx.data.setdefault("unity_record_parts", {})
    parts["module3"] = {"module_output": module_output}


def _add_datapoint(
    datapoints: List[Dict[str, Any]],
    module_id: int,
    item_type: str,
    data: Dict[str, Any],
    reasons: Optional[List[Any]] = None,
    scaffold_id: Optional[str] = None,
    variant_id: Optional[str] = None,
) -> None:
    entry: Dict[str, Any] = {
        "module_id": module_id,
        "item_type": item_type,
        "data": data,
    }
    if scaffold_id:
        entry["scaffold_id"] = scaffold_id
    if variant_id:
        entry["variant_id"] = variant_id
    if reasons:
        entry["reasons"] = reasons
    datapoints.append(entry)


def _reaction_hash_mismatch(shared_input: Dict[str, Any], job_card: Dict[str, Any]) -> bool:
    shared_hash = shared_input.get("reaction_identity_hash")
    job_hash = (job_card.get("reaction_identity") or {}).get("hash")
    return bool(shared_hash and job_hash and shared_hash != job_hash)


def _energy_ledger_update(
    module3_result: Dict[str, Any],
    shared_energy_ledger: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(shared_energy_ledger or {})
    notes = list(merged.get("notes") or [])
    notes.append("module3 detectability used energy ledger (no overrides)")
    merged["notes"] = list(dict.fromkeys(notes))
    return merged


def _collect_datapoints(ctx: PipelineContext, module3_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    datapoints: List[Dict[str, Any]] = []
    job_card = ctx.data.get("job_card") or {}
    module0 = ctx.data.get("module0_strategy_router") or {}
    module1 = ctx.data.get("module1_topogate") or {}
    module2 = ctx.data.get("module2_active_site_refinement") or {}
    pipeline_summary = ctx.data.get("pipeline_summary") or {}

    _add_datapoint(
        datapoints,
        module_id=3,
        item_type="run_summary",
        data={
            "pipeline_summary": pipeline_summary,
            "unity_arbitration": ctx.data.get("unity_arbitration"),
        },
    )

    _add_datapoint(
        datapoints,
        module_id=0,
        item_type="route_decision",
        data={
            "decision": job_card.get("decision"),
            "chosen_route": job_card.get("chosen_route")
            or (job_card.get("mechanism_route") or {}).get("primary"),
            "route_posteriors": job_card.get("route_posteriors")
            or module0.get("route_posteriors"),
            "confidence": job_card.get("confidence"),
            "physics_audit": job_card.get("physics_audit"),
            "bond_context": job_card.get("bond_context"),
            "requested_output_check": job_card.get("requested_output_check"),
            "job_card_snapshot": job_card,
        },
        reasons=(job_card.get("route_explanation") or job_card.get("reasons") or []),
    )

    _add_datapoint(
        datapoints,
        module_id=1,
        item_type="module1_summary",
        data={
            "status": module1.get("status"),
            "module1_confidence": module1.get("module1_confidence"),
            "module1_physics_audit": module1.get("module1_physics_audit"),
            "weights": module1.get("weights"),
            "mode": module1.get("mode"),
        },
        reasons=module1.get("warnings") or [],
    )

    ranked_scaffolds = module1.get("ranked_scaffolds") or []
    for scaffold in ranked_scaffolds:
        reasons: List[Any] = []
        fail_codes = scaffold.get("fail_codes") or []
        if fail_codes:
            reasons.extend(fail_codes)
        retention_risk = (scaffold.get("retention_metrics") or {}).get(
            "retention_risk_flag"
        )
        if retention_risk:
            reasons.append(f"retention_risk:{retention_risk}")
        mech_notes = (
            (scaffold.get("reach_summary") or {})
            .get("mechanism_compat", {})
            .get("notes")
            or []
        )
        reasons.extend(mech_notes)
        _add_datapoint(
            datapoints,
            module_id=1,
            item_type="scaffold_ranked",
            data=scaffold,
            reasons=reasons,
            scaffold_id=scaffold.get("scaffold_id"),
        )

    _add_datapoint(
        datapoints,
        module_id=2,
        item_type="module2_summary",
        data={
            "status": module2.get("status"),
            "selected_scaffold_id": (module2.get("selected_scaffold") or {}).get("scaffold_id"),
            "best_variant": module2.get("best_variant"),
            "module2_physics_audit": module2.get("module2_physics_audit"),
            "confidence_calibration": module2.get("confidence_calibration"),
        },
        reasons=module2.get("warnings") or [],
    )

    rejected_scaffolds = module1.get("rejected_scaffolds") or []
    for scaffold in rejected_scaffolds:
        fail_codes = scaffold.get("fail_codes") or []
        _add_datapoint(
            datapoints,
            module_id=1,
            item_type="scaffold_rejected",
            data=scaffold,
            reasons=fail_codes,
            scaffold_id=scaffold.get("scaffold_id"),
        )

    selected_scaffold = module2.get("selected_scaffold") or {}
    if selected_scaffold:
        reasons = []
        selection_explain = module2.get("selection_explain")
        if selection_explain:
            reasons.append(selection_explain)
        retention_reasons = selected_scaffold.get("retention_penalty_reasons") or []
        reasons.extend(retention_reasons)
        _add_datapoint(
            datapoints,
            module_id=2,
            item_type="scaffold_selected",
            data=selected_scaffold,
            reasons=reasons,
            scaffold_id=selected_scaffold.get("scaffold_id"),
        )

    scaffold_rankings = module2.get("scaffold_rankings") or []
    for scaffold in scaffold_rankings:
        reasons = scaffold.get("retention_penalty_reasons") or []
        _add_datapoint(
            datapoints,
            module_id=2,
            item_type="scaffold_scored",
            data=scaffold,
            reasons=reasons,
            scaffold_id=scaffold.get("scaffold_id"),
        )

    variants = module2.get("variant_set") or []
    for variant in variants:
        reasons = []
        if variant.get("rationale"):
            reasons.append(variant["rationale"])
        if variant.get("requires_structural_localization"):
            reasons.append("requires_structural_localization")
        _add_datapoint(
            datapoints,
            module_id=2,
            item_type="variant_candidate",
            data=variant,
            reasons=reasons,
            variant_id=variant.get("variant_id"),
            scaffold_id=(selected_scaffold.get("scaffold_id") if selected_scaffold else None),
        )

    protocol_card = module3_output.get("protocol_card") or {}
    for arm in protocol_card.get("arms") or []:
        reasons = []
        if arm.get("why"):
            reasons.append(arm["why"])
        if arm.get("type"):
            reasons.append(f"arm_type:{arm['type']}")
        _add_datapoint(
            datapoints,
            module_id=3,
            item_type="experiment_arm",
            data=arm,
            reasons=reasons,
            variant_id=(arm.get("candidate") or {}).get("variant_id"),
            scaffold_id=(arm.get("candidate") or {}).get("scaffold_id"),
        )

    qc_result = module3_output.get("qc_result")
    if qc_result:
        _add_datapoint(
            datapoints,
            module_id=3,
            item_type="qc_result",
            data=qc_result,
            reasons=qc_result.get("reasons") or [],
        )

    learning_update = module3_output.get("learning_update") or {}
    _add_datapoint(
        datapoints,
        module_id=3,
        item_type="learning_update",
        data=learning_update,
        reasons=learning_update.get("router_updates") or [],
    )
    return datapoints


def _emit_unity_record(ctx: PipelineContext, module3_output: Dict[str, Any]) -> None:
    db_path = os.environ.get("EVIDENCE_DB_PATH")
    if not db_path:
        return
    try:
        record = _build_unity_record(ctx, module3_output)
        save_run(db_path, record)
        datapoints = _collect_datapoints(ctx, module3_output)
        written = add_datapoints(db_path, record.run_id, datapoints)
        learning_update = module3_output.get("learning_update") or {}
        learning_update["datapoints_written"] = written
        module3_output["learning_update"] = learning_update
    except Exception as exc:
        warnings = module3_output.get("warnings") or []
        warnings.append(f"W_EVIDENCE_STORE: {exc}")
        module3_output["warnings"] = list(dict.fromkeys(warnings))


def _build_unity_record(ctx: PipelineContext, module3_output: Dict[str, Any]) -> UnityRecord:
    shared = ctx.data.get("shared_io") or {}
    shared_input = shared.get("input") or {}
    telemetry = shared_input.get("telemetry") or {}
    protocol = module3_output.get("protocol_card") or {}
    run_id = (
        telemetry.get("run_id")
        or protocol.get("batch_id")
        or str(uuid.uuid4())
    )

    condition_payload = shared_input.get("condition_profile") or {}
    if not condition_payload:
        condition_payload = (ctx.data.get("job_card") or {}).get("condition_profile") or {}
    unity_condition = UnityConditionProfile(
        pH=condition_payload.get("pH"),
        temperature_K=condition_payload.get("temperature_K"),
        temperature_C=condition_payload.get("temperature_C"),
        ionic_strength=condition_payload.get("ionic_strength"),
        solvent=condition_payload.get("solvent"),
        cofactors=list(condition_payload.get("cofactors") or []),
        salts_buffer=condition_payload.get("salts_buffer"),
        constraints=condition_payload.get("constraints"),
    )

    job_card = ctx.data.get("job_card") or {}
    bond_context_data = job_card.get("bond_context") or {}
    structure_summary = job_card.get("structure_summary") or {}
    unity_bond = BondContext(
        bond_role=bond_context_data.get("primary_role") or bond_context_data.get("bond_role"),
        bond_role_confidence=bond_context_data.get("primary_role_confidence")
        or bond_context_data.get("role_confidence"),
        bond_class=bond_context_data.get("bond_class"),
        polarity=bond_context_data.get("polarity"),
        atom_count=structure_summary.get("atom_count"),
        hetero_atoms=structure_summary.get("hetero_atoms"),
        ring_count=structure_summary.get("ring_count"),
    )

    physics_audit_data = job_card.get("physics_audit") or {}
    unity_physics = (
        PhysicsAudit(
            deltaG_dagger_kJ_per_mol=physics_audit_data.get("deltaG_dagger_kJ_per_mol"),
            eyring_k_s_inv=physics_audit_data.get("eyring_k_s_inv"),
            k_eff_s_inv=physics_audit_data.get("k_eff_s_inv"),
            temperature_K=physics_audit_data.get("temperature_K")
            or unity_condition.temperature_K,
            horizon_s=physics_audit_data.get("horizon_s"),
            notes=list(physics_audit_data.get("notes") or []),
        )
        if physics_audit_data
        else None
    )

    module0_payload = ctx.data.get("module0_strategy_router") or {}
    module0_route = (job_card.get("mechanism_route") or {}).get("primary") or job_card.get(
        "chosen_route"
    )
    module0_confidence = job_card.get("confidence") or {}
    unity_module0 = Module0Out(
        decision=job_card.get("decision") or module0_payload.get("status"),
        route_family=module0_route,
        route_confidence=module0_confidence.get("route"),
        data_support=module0_payload.get("data_support") or job_card.get("data_support"),
    )

    module1_output = ctx.data.get("module1_topogate") or {}
    module1_confidence = module1_output.get("module1_confidence") or {}
    module1_access = module1_confidence.get("access") or module1_confidence.get("access_mean")
    module1_reach = module1_confidence.get("reach") or module1_confidence.get("reach_mean")
    module1_retention = module1_confidence.get("retention") or module1_confidence.get(
        "retention_mean"
    )
    module1_ranked = module1_output.get("ranked_scaffolds") or []
    unity_module1 = Module1Out(
        status=module1_output.get("status"),
        access_score=module1_access,
        reach_score=module1_reach,
        retention_score=module1_retention,
        top_scaffold=module1_ranked[0].get("scaffold_id") if module1_ranked else None,
    )

    module2_output = ctx.data.get("module2_active_site_refinement") or {}
    module2_selected = module2_output.get("selected_scaffold") or {}
    module2_best = module2_output.get("best_variant") or {}
    module2_physics = module2_output.get("module2_physics_audit") or {}
    unity_module2 = Module2Out(
        status=module2_output.get("status"),
        selected_scaffold=module2_selected.get("scaffold_id"),
        best_variant=module2_best.get("variant_id"),
        deltaG_dagger_kJ_per_mol=module2_physics.get("deltaG_dagger_variant_kJ_per_mol"),
        k_pred_s_inv=module2_physics.get("k_variant_s_inv")
        or module2_physics.get("k_eff_s_inv"),
        route_family=module0_route,
    )

    unity_module3 = Module3Out(
        status=module3_output.get("status"),
        plan_score=(module3_output.get("information_gain") or {}).get("plan_score"),
        qc_status=(module3_output.get("qc_result") or {}).get("status")
        or module3_output.get("qc_status"),
        batch_id=protocol.get("batch_id"),
    )

    return UnityRecord(
        run_id=run_id,
        smiles=ctx.smiles,
        target_bond=ctx.target_bond,
        requested_output=ctx.requested_output,
        condition_profile=unity_condition,
        bond_context=unity_bond,
        physics_audit=unity_physics,
        module0=unity_module0,
        module1=unity_module1,
        module2=unity_module2,
        module3=unity_module3,
    )


def _reject_status(reasons: List[str]) -> str:
    if any("batch_id mismatch" in reason for reason in reasons):
        return "REJECTED_BATCH_ID_MISMATCH"
    if any("negative control conversion" in reason for reason in reasons):
        return "REJECTED_CONTROL_VIOLATION"
    if any("baseline stdev" in reason for reason in reasons):
        return "REJECTED_HIGH_VARIANCE"
    if any("baseline n < 2" in reason for reason in reasons):
        return "REJECTED_LOW_REPLICATES"
    if any("baseline below negative control margin" in reason for reason in reasons):
        return "REJECTED_SANITY_RANKING"
    return "REJECTED_QC_FAIL"
