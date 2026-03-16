from __future__ import annotations

from typing import Any, Dict, List

from .benchmark_schema import PipelineRunResult


def _result(ok: bool, warnings: List[str], errors: List[str], summary: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": bool(ok), "warnings": warnings, "errors": errors, "summary": summary}


def validate_module_minus1(payload: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    resolved = payload.get("resolved_target") or {}
    if not resolved:
        errors.append("missing_resolved_target")
    if resolved and resolved.get("match_count", 0) > 1 and not (resolved.get("selection_mode") or resolved.get("resolution_note")):
        warnings.append("ambiguous_target_without_resolution_note")
    if payload.get("reactivity") is None and payload.get("confidence_prior") is None:
        warnings.append("missing_reactivity_prior")
    if payload.get("mechanism_eligibility") is None:
        errors.append("missing_mechanism_eligibility")
    summary = {
        "status": payload.get("status"),
        "match_count": resolved.get("match_count"),
        "bond_type": resolved.get("bond_type"),
    }
    return _result(not errors, warnings, errors, summary)


def validate_module0(job_card: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    route_posteriors = job_card.get("route_posteriors") or []
    if not route_posteriors:
        errors.append("missing_route_posteriors")
    if not (job_card.get("chosen_route") or (job_card.get("mechanism_route") or {}).get("primary")):
        errors.append("missing_selected_route")
    if not job_card:
        errors.append("missing_job_card")
    if not (job_card.get("physics_audit") or job_card.get("physics_layer")):
        warnings.append("missing_module0_physics")
    summary = {
        "status": payload.get("status"),
        "decision": job_card.get("decision"),
        "route_count": len(route_posteriors),
        "chosen_route": job_card.get("chosen_route") or (job_card.get("mechanism_route") or {}).get("primary"),
    }
    return _result(not errors, warnings, errors, summary)


def validate_module1(payload: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    ranked = payload.get("ranked_scaffolds") or []
    if not ranked:
        errors.append("missing_ranked_scaffolds")
    if not (payload.get("module1_confidence") or {}).get("scaffold_id") and ranked:
        warnings.append("missing_explicit_top_scaffold_id")
    confidence = payload.get("module1_confidence") or {}
    for field in ("access", "reach", "retention", "total"):
        if field not in confidence:
            warnings.append(f"missing_confidence_{field}")
    summary = {
        "status": payload.get("status"),
        "ranked_scaffold_count": len(ranked),
        "top_scaffold": confidence.get("scaffold_id"),
    }
    return _result(not errors, warnings, errors, summary)


def validate_module2(payload: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    variants = payload.get("variant_set") or []
    best_variant = payload.get("best_variant") or {}
    if not variants:
        warnings.append("missing_variant_set")
    if not best_variant:
        errors.append("missing_best_variant")
    if not (payload.get("final_report") or payload.get("candidate_reports")):
        warnings.append("missing_refinement_rationale")
    summary = {
        "status": payload.get("status"),
        "variant_count": len(variants),
        "best_variant": best_variant.get("variant_id") or best_variant.get("label"),
    }
    return _result(not errors, warnings, errors, summary)


def validate_module3(payload: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    protocol = payload.get("protocol_card") or {}
    arms = protocol.get("arms") or []
    if not arms:
        errors.append("missing_protocol_arms")
    if payload.get("information_gain") is None:
        warnings.append("missing_information_gain")
    if payload.get("qc_status") is None:
        warnings.append("missing_qc_status")
    summary = {
        "status": payload.get("status"),
        "arm_count": len(arms),
        "qc_status": payload.get("qc_status"),
    }
    return _result(not errors, warnings, errors, summary)


def validate_shared_and_arbitration(shared_io: Dict[str, Any], arbitration: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    errors: List[str] = []
    if not shared_io:
        errors.append("missing_shared_io")
    if not arbitration:
        warnings.append("missing_unity_arbitration")
    if arbitration.get("hard_veto") and not arbitration.get("recommendation"):
        warnings.append("hard_veto_without_recommendation")
    summary = {
        "has_shared_io": bool(shared_io),
        "has_arbitration": bool(arbitration),
        "hard_veto": arbitration.get("hard_veto"),
    }
    return _result(not errors, warnings, errors, summary)


def validate_pipeline_result(run: PipelineRunResult) -> Dict[str, Dict[str, Any]]:
    if not run.ok:
        fail = _result(False, [], [run.exception or "pipeline_exception"], {"status": "EXCEPTION"})
        return {name: fail for name in ["module_minus1", "module0", "module1", "module2", "module3", "shared"]}

    raw = run.raw_outputs
    return {
        "module_minus1": validate_module_minus1(raw.get("module_minus1") or {}),
        "module0": validate_module0(raw.get("job_card") or {}, raw.get("module0_strategy_router") or {}),
        "module1": validate_module1(raw.get("module1_topogate") or {}),
        "module2": validate_module2(raw.get("module2_active_site_refinement") or {}),
        "module3": validate_module3(raw.get("module3_experiment_designer") or {}),
        "shared": validate_shared_and_arbitration(raw.get("shared_io") or {}, raw.get("unity_arbitration") or {}),
    }
