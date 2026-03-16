from __future__ import annotations

from typing import Any, Dict, List

from enzyme_software.pipeline import run_pipeline

from .benchmark_schema import BenchmarkCase, PipelineRunResult


def _module_summary(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload or {}
    summary = {
        "status": payload.get("status"),
        "halt_reason": payload.get("halt_reason"),
        "warning_count": len(payload.get("warnings") or []),
        "error_count": len(payload.get("errors") or []),
    }
    if name == "module_minus1":
        resolved = payload.get("resolved_target") or {}
        summary.update(
            {
                "match_count": resolved.get("match_count"),
                "canonical_token": resolved.get("canonical_token"),
                "bond_type": resolved.get("bond_type"),
            }
        )
    elif name == "module0":
        shared_io = payload.get("shared_io") or {}
        route_audit = payload.get("route_audit") or {}
        summary["job_card_present"] = bool(shared_io)
        summary["chosen_route"] = ((shared_io.get("outputs") or {}).get("module0") or {}).get("result", {}).get("chosen_route")
        summary["route_gap"] = route_audit.get("route_gap")
        summary["ambiguity_flag"] = route_audit.get("ambiguity_flag")
        summary["fallback_used"] = route_audit.get("fallback_used")
        summary["calibration_mode"] = (route_audit.get("calibration") or {}).get("mode")
    elif name == "module1":
        summary["ranked_scaffold_count"] = len(payload.get("ranked_scaffolds") or [])
        summary["top_scaffold"] = ((payload.get("module1_confidence") or {}).get("scaffold_id"))
    elif name == "module2":
        best_variant = payload.get("best_variant") or {}
        summary["variant_count"] = len(payload.get("variant_set") or [])
        summary["best_variant"] = best_variant.get("variant_id") or best_variant.get("label")
    elif name == "module3":
        summary["protocol_arm_count"] = len(((payload.get("protocol_card") or {}).get("arms") or []))
        summary["qc_status"] = payload.get("qc_status")
    return summary


def run_case_through_pipeline(case: BenchmarkCase) -> PipelineRunResult:
    warnings: List[str] = []
    errors: List[str] = []
    try:
        ctx = run_pipeline(
            case.smiles,
            case.target_bond if isinstance(case.target_bond, str) else str(case.target_bond),
            requested_output=case.metadata.get("requested_output"),
            trap_target=case.metadata.get("trap_target"),
        )
        data = dict(ctx.data)
        job_card = data.get("job_card") or {}
        module_minus1 = data.get("module_minus1") or {}
        module0 = data.get("module0_strategy_router") or {}
        module1 = data.get("module1_topogate") or {}
        module2 = data.get("module2_active_site_refinement") or {}
        module3 = data.get("module3_experiment_designer") or {}
        shared_io = data.get("shared_io") or {}
        warnings.extend(module_minus1.get("warnings") or [])
        warnings.extend(module1.get("warnings") or [])
        warnings.extend(module2.get("warnings") or [])
        warnings.extend(module3.get("warnings") or [])
        errors.extend(module_minus1.get("errors") or [])
        errors.extend(module1.get("errors") or [])
        errors.extend(module2.get("errors") or [])
        errors.extend(module3.get("errors") or [])
        module_summaries = {
            "module_minus1": _module_summary("module_minus1", module_minus1),
            "module0": _module_summary("module0", module0),
            "module1": _module_summary("module1", module1),
            "module2": _module_summary("module2", module2),
            "module3": _module_summary("module3", module3),
        }
        return PipelineRunResult(
            case=case,
            ok=True,
            exception=None,
            raw_outputs=data,
            module_summaries=module_summaries,
            confidence_state=job_card.get("confidence") or {},
            energy_ledger_summary=shared_io.get("energy_ledger") or (job_card.get("energy_ledger") or {}),
            arbitration_summary=data.get("unity_arbitration") or {},
            warnings=warnings,
            errors=errors,
        )
    except Exception as exc:  # pragma: no cover - exercised by harness tests
        return PipelineRunResult(
            case=case,
            ok=False,
            exception=str(exc),
            raw_outputs={},
            module_summaries={},
            confidence_state={},
            energy_ledger_summary={},
            arbitration_summary={},
            warnings=warnings,
            errors=errors + [str(exc)],
        )
