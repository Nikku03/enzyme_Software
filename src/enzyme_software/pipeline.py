from __future__ import annotations

from typing import Iterable, List
import uuid

from enzyme_software.context import OperationalConstraints, PipelineContext
from enzyme_software.config import (
    RETENTION_WEAK_THRESHOLD,
    ROUTE_CONFIDENCE_LOW_THRESHOLD,
    TARGET_RESOLUTION_LOW_THRESHOLD,
)
from enzyme_software.modules import (
    ModuleMinus1SRE,
    Module0StrategyRouter,
    Module1TopoGate,
    Module2ActiveSiteRefinement,
    Module3ExperimentDesigner,
)
from enzyme_software.modules.base import BaseModule
from enzyme_software.unity_layer import (
    build_shared_state,
    merge_module_output,
    arbitrate_shared_state,
    consistency_market,
    export_shared_io_patch,
    validate_contract,
)


def build_pipeline() -> List[BaseModule]:
    return [
        ModuleMinus1SRE(),
        Module0StrategyRouter(),
        Module1TopoGate(),
        Module2ActiveSiteRefinement(),
        Module3ExperimentDesigner(),
    ]


def run_pipeline(
    smiles: str,
    target_bond: str,
    requested_output: str | None = None,
    trap_target: str | None = None,
    constraints: OperationalConstraints | None = None,
    modules: Iterable[BaseModule] | None = None,
) -> PipelineContext:
    ctx = PipelineContext(
        smiles=smiles,
        target_bond=target_bond,
        requested_output=requested_output,
        trap_target=trap_target,
        constraints=constraints or OperationalConstraints(),
    )
    shared_state = build_shared_state(
        smiles=smiles,
        target_bond=target_bond,
        requested_output=requested_output,
        trap_target=trap_target,
        constraints=ctx.constraints.to_dict(),
    )
    ctx.data["shared_state"] = shared_state.to_dict()
    ctx.data["unity_state"] = export_shared_io_patch(shared_state)
    pipeline_modules = list(modules) if modules is not None else build_pipeline()
    for module in pipeline_modules:
        ctx.data["unity_state"] = export_shared_io_patch(shared_state)
        ctx = module.run(ctx)
        module_id = _module_id_for(module)
        if module_id is None:
            continue
        module_output = _module_output_for(ctx, module_id)
        shared_state = merge_module_output(module_id, shared_state, module_output)
        arbitration = arbitrate_shared_state(
            shared_state,
            module_outputs=ctx.data,
            job_card=ctx.data.get("job_card"),
            stage=f"module{module_id}",
        )
        ctx.data["unity_arbitration"] = arbitration
        market = consistency_market(
            shared_state,
            module_outputs=ctx.data,
            job_card=ctx.data.get("job_card"),
        )
        ctx.data["consistency_market"] = market
        shared_state.audit["consistency_market"] = market
        penalty = market.get("penalty")
        if isinstance(penalty, (int, float)) and penalty > 0.0:
            job_card = ctx.data.get("job_card") or {}
            confidence = job_card.get("confidence") or {}
            for key in ("route", "feasibility_if_specified"):
                value = confidence.get(key)
                if isinstance(value, (int, float)):
                    adjusted = max(0.0, min(1.0, float(value) * (1.0 - float(penalty))))
                    confidence[key] = round(adjusted, 3)
            confidence["consistency_penalty"] = round(float(penalty), 3)
            job_card["confidence"] = confidence
            ctx.data["job_card"] = job_card
            module3 = ctx.data.get("module3_experiment_designer") or {}
            info = module3.get("information_gain") or {}
            if isinstance(info.get("plan_score"), (int, float)):
                info["plan_score"] = round(
                    max(0.0, min(1.0, float(info["plan_score"]) * (1.0 - 0.5 * penalty))), 3
                )
                module3["information_gain"] = info
                ctx.data["module3_experiment_designer"] = module3
        ctx.data["shared_state"] = shared_state.to_dict()
        ctx.data["unity_state"] = export_shared_io_patch(shared_state)
        shared_io = _ensure_shared_io_contract(ctx)
        if isinstance(shared_io, dict):
            shared_input = shared_io.get("input") or {}
            shared_input["unity_state"] = ctx.data["unity_state"]
            shared_io["input"] = shared_input
            ctx.data["shared_io"] = shared_io
    violations = validate_contract(
        shared_state,
        job_type=shared_state.chemistry.derived.get("job_type"),
    )
    shared_state.audit["contract_violations"] = violations
    ctx.data["shared_state"] = shared_state.to_dict()
    ctx.data["pipeline_summary"] = _build_pipeline_summary(ctx)
    return ctx


def _module_id_for(module: BaseModule) -> int | None:
    if hasattr(module, "module_id"):
        try:
            return int(getattr(module, "module_id"))
        except (TypeError, ValueError):
            return None
    if isinstance(module, Module0StrategyRouter):
        return 0
    if isinstance(module, ModuleMinus1SRE):
        return -1
    if isinstance(module, Module1TopoGate):
        return 1
    if isinstance(module, Module2ActiveSiteRefinement):
        return 2
    if isinstance(module, Module3ExperimentDesigner):
        return 3
    return None


def _get_module_minus1_payload(ctx: PipelineContext) -> dict:
    shared = ctx.data.get("shared_io") or {}
    shared_mod = (shared.get("outputs") or {}).get("module_minus1") or {}
    if shared_mod:
        return dict(shared_mod.get("result") or shared_mod)
    if "module_minus1" in ctx.data:
        return dict(ctx.data.get("module_minus1") or {})
    return {}


def _module_output_for(ctx: PipelineContext, module_id: int) -> dict:
    if module_id == 0:
        output = dict(ctx.data.get("module0_strategy_router") or {})
        job_card = ctx.data.get("job_card")
        if job_card is not None:
            output["job_card"] = job_card
        return output
    if module_id == -1:
        return _get_module_minus1_payload(ctx)
    if module_id == 1:
        return dict(ctx.data.get("module1_topogate") or {})
    if module_id == 2:
        return dict(ctx.data.get("module2_active_site_refinement") or {})
    if module_id == 3:
        return dict(ctx.data.get("module3_experiment_designer") or {})
    return {}


def _build_pipeline_summary(ctx: PipelineContext) -> dict:
    job_card = ctx.data.get("job_card") or {}
    module_minus1 = _get_module_minus1_payload(ctx)
    module0 = ctx.data.get("module0_strategy_router") or {}
    module1 = ctx.data.get("module1_topogate") or {}
    module2 = ctx.data.get("module2_active_site_refinement") or {}
    module3 = ctx.data.get("module3_experiment_designer") or {}

    module_minus1_status = module_minus1.get("status") or "n/a"

    decision = job_card.get("decision") or "UNKNOWN"
    halt_reason = job_card.get("pipeline_halt_reason")
    route = (
        (job_card.get("mechanism_route") or {}).get("primary")
        or job_card.get("chosen_route")
        or "n/a"
    )
    module1_status = module1.get("status") or "n/a"
    module2_status = module2.get("status") or "n/a"
    module3_status = module3.get("status") or "n/a"
    module1_ranked = module1.get("ranked_scaffolds") or []
    module1_confidence = module1.get("module1_confidence") or {}
    module1_retention = module1_confidence.get("retention")
    if module1_retention is None:
        module1_retention = module1_confidence.get("retention_mean")
    module2_selected = module2.get("selected_scaffold") or {}
    selected_scaffold = (
        module2_selected.get("scaffold_id")
        or module2.get("selected_scaffold_id")
        or "n/a"
    )
    best_variant = (
        (module2.get("best_variant") or {}).get("variant_id")
        or (module2_selected.get("best_variant") or {}).get("variant_id")
        or "n/a"
    )
    protocol_card = module3.get("protocol_card") or {}
    negative_control = (protocol_card.get("controls") or {}).get("negative_control_arm_id")

    bond = (job_card.get("resolved_target") or {}).get("selected_bond") or {}
    selected_bond = (
        "-".join(map(str, bond.get("atom_indices")))
        if isinstance(bond.get("atom_indices"), list)
        else "n/a"
    )

    working: list[str] = []
    if module_minus1_status == "PASS":
        constraint = module_minus1.get("primary_constraint") or "NONE"
        working.append(f"Module -1 reactivity: PASS (constraint: {constraint})")
    if decision in {"GO", "LOW_CONF_GO"}:
        working.append(f"Routing accepted ({decision})")
    if selected_bond != "n/a":
        working.append(f"Bond resolved: {selected_bond}")
    if module1_status == "PASS":
        working.append(f"Module 1 passed ({len(module1_ranked)} scaffolds ranked)")
    if module2_status in {"PASS", "DEGRADED_OK"} and selected_scaffold != "n/a":
        working.append(f"Module 2 selected {selected_scaffold}")
    if negative_control:
        working.append("Negative control assigned")
    physics_prior = (job_card.get("physics") or {}).get("prior_success_probability")
    if isinstance(physics_prior, (int, float)):
        working.append(f"Physics prior P={round(float(physics_prior), 2)}")

    needs_improvement: list[str] = []
    if decision in {"NO_GO", "LOW_CONF", "HALT"} or str(decision).startswith("HALT"):
        halt_reason = job_card.get("pipeline_halt_reason") or "unspecified"
        needs_improvement.append(f"Module 0 halted: {halt_reason}")
    if module1_status != "PASS":
        needs_improvement.append(f"Module 1 status: {module1_status}")
    if module2_status not in {"PASS", "DEGRADED_OK"}:
        needs_improvement.append(f"Module 2 status: {module2_status}")
    module3_qc_status = module3.get("qc_status") or (module3.get("qc_result") or {}).get("status")
    if module3_qc_status == "FAIL":
        needs_improvement.append("Module 3 QC failed")
    confidence = job_card.get("confidence") or {}
    route_confidence = confidence.get("route")
    if isinstance(route_confidence, (int, float)) and route_confidence < ROUTE_CONFIDENCE_LOW_THRESHOLD.value:
        needs_improvement.append("Route confidence low")
    if isinstance(module1_retention, (int, float)) and module1_retention < RETENTION_WEAK_THRESHOLD.value:
        needs_improvement.append("Retention score low")
    if not negative_control:
        needs_improvement.append("Missing negative control")

    m1_resolved = module_minus1.get("resolved_target") or {}
    m1_candidates = m1_resolved.get("candidate_bonds") or m1_resolved.get("candidate_attack_sites") or []
    handoff_checks = [
        {
            "name": "M-1 → M0",
            "ok": bool(m1_resolved) and isinstance(m1_candidates, list),
            "detail": "resolved_target + selected candidates",
        },
        {
            "name": "M0 → M1",
            "ok": bool((job_card.get("resolved_target") or {}).get("selected_bond")),
            "detail": "selected_bond",
        },
        {
            "name": "M1 → M2",
            "ok": bool((module1.get("module2_handoff") or {}).get("top_scaffolds")),
            "detail": "top_scaffolds",
        },
        {
            "name": "M2 → M3",
            "ok": bool(module2.get("module3_handoff")),
            "detail": "module3_handoff",
        },
        {
            "name": "Shared IO",
            "ok": bool(ctx.data.get("shared_io")),
            "detail": "condition_profile + telemetry",
        },
    ]

    results = {
        "decision": decision,
        "halt_reason": halt_reason,
        "route": route,
        "module_minus1_status": module_minus1_status,
        "module_minus1_primary_constraint": module_minus1.get("primary_constraint"),
        "module_minus1_confidence_prior": module_minus1.get("confidence_prior"),
        "module1_status": module1_status,
        "module2_status": module2_status,
        "module3_status": module3_status,
        "top_scaffold_count": len(module1_ranked),
        "selected_scaffold": selected_scaffold,
        "best_variant": best_variant,
        "negative_control_arm_id": negative_control,
    }
    results_list = [
        f"Module -1: {module_minus1_status} (constraint: {module_minus1.get('primary_constraint', 'n/a')})",
        f"Route: {route}",
        f"Module 1: {module1_status}",
        f"Module 2: {selected_scaffold}",
        f"Module 3 arms: {len(protocol_card.get('arms') or [])}",
    ]
    if negative_control:
        results_list.append(f"Neg control: {negative_control}")
    if halt_reason:
        results_list.append(f"Halt reason: {halt_reason}")

    return {
        "working": working,
        "needs_improvement": needs_improvement,
        "results": results,
        "results_list": results_list,
        "interconnection": {"handoff_checks": handoff_checks},
        "why_this_score": _build_why_this_score(job_card, module1, module2, module3),
        "interlink_audit": ctx.data.get("interlink_audit") or {},
    }


def _ensure_shared_io_contract(ctx: PipelineContext) -> dict:
    shared_io = ctx.data.get("shared_io")
    if not isinstance(shared_io, dict):
        run_id = str(uuid.uuid4())
        shared_io = {
            "input": {
                "schema_version": "shared_io.v2",
                "substrate_context": {"smiles": ctx.smiles, "mol_block": None},
                "bond_spec": {
                    "target_bond": ctx.target_bond,
                    "target_bond_indices": None,
                    "selection_mode": None,
                    "resolved_target": {},
                    "context": {},
                },
                "condition_profile": {
                    "pH": ctx.constraints.ph_min if ctx.constraints.ph_min is not None else ctx.constraints.ph_max,
                    "temperature_K": round(((ctx.constraints.temperature_c or 25.0) + 273.15), 2),
                    "temperature_C": ctx.constraints.temperature_c,
                    "solvent": None,
                    "ionic_strength": None,
                    "cofactors": [],
                },
                "telemetry": {"run_id": run_id, "trace": [], "warnings": []},
            },
            "outputs": {},
        }
    m1 = ctx.data.get("module_minus1") or {}
    if m1:
        shared_input = shared_io.setdefault("input", {})
        telemetry = shared_input.setdefault("telemetry", {})
        trace = telemetry.get("trace") or []
        if "module-1" not in trace:
            trace.append("module-1")
        telemetry["trace"] = trace
        shared_input["telemetry"] = telemetry
        bond_spec = shared_input.setdefault("bond_spec", {})
        if not bond_spec.get("resolved_target"):
            bond_spec["resolved_target"] = m1.get("resolved_target") or {}
        if not bond_spec.get("target_bond"):
            bond_spec["target_bond"] = ctx.target_bond
        shared_input["bond_spec"] = bond_spec
        outputs = shared_io.setdefault("outputs", {})
        outputs["module_minus1"] = outputs.get("module_minus1") or {
            "result": {
                "status": m1.get("status"),
                "resolved_target": m1.get("resolved_target"),
                "reactivity": m1.get("reactivity"),
                "primary_constraint": m1.get("primary_constraint"),
                "warnings": m1.get("warnings", []),
                "errors": m1.get("errors", []),
            },
            "sre_atr": {
                "bond360_profile": m1.get("bond360_profile"),
                "module_minus1_schema": m1.get("module_minus1_schema"),
            },
            "fragment_builder": m1.get("fragment") or {},
            "cpt": m1.get("cpt_scores") or {},
            "level1": (m1.get("cpt_scores") or {}).get("level1", {}),
            "level2": (m1.get("cpt_scores") or {}).get("level2", {}),
            "level3": (m1.get("cpt_scores") or {}).get("level3", {}),
            "ep_av": (m1.get("cpt_scores") or {}).get("epav", {}),
            "evidence_record": {
                "cpt_scores": m1.get("cpt_scores"),
                "bond360_profile": m1.get("bond360_profile"),
                "mechanism_eligibility": m1.get("mechanism_eligibility"),
                "route_bias": m1.get("route_bias"),
                "confidence_prior": m1.get("confidence_prior"),
            },
        }
        shared_io["outputs"] = outputs
        shared_io["input"] = shared_input
    return shared_io


def _build_why_this_score(
    job_card: dict,
    module1: dict,
    module2: dict,
    module3: dict,
) -> list[str]:
    reasons: list[str] = []
    confidence = job_card.get("confidence") or {}
    route_conf = confidence.get("route")
    if isinstance(route_conf, (int, float)):
        threshold = ROUTE_CONFIDENCE_LOW_THRESHOLD
        status = "below" if route_conf < threshold.value else "above"
        reasons.append(
            f"Route confidence {route_conf:.2f} {status} {threshold.value:.2f} ({threshold.rationale})"
        )
    target_resolution = confidence.get("target_resolution")
    if isinstance(target_resolution, (int, float)):
        threshold = TARGET_RESOLUTION_LOW_THRESHOLD
        status = "below" if target_resolution < threshold.value else "above"
        reasons.append(
            f"Target resolution {target_resolution:.2f} {status} {threshold.value:.2f} ({threshold.rationale})"
        )
    module1_conf = module1.get("module1_confidence") or {}
    retention = module1_conf.get("retention")
    if retention is None:
        retention = module1_conf.get("retention_mean")
    if isinstance(retention, (int, float)):
        threshold = RETENTION_WEAK_THRESHOLD
        status = "below" if retention < threshold.value else "above"
        reasons.append(
            f"Retention {retention:.2f} {status} {threshold.value:.2f} ({threshold.rationale})"
        )
    physics = job_card.get("physics_audit") or {}
    energy_ledger = job_card.get("energy_ledger") or {}
    prior = (
        physics.get("prior_success_probability_final")
        or physics.get("route_prior_target_specific")
        or energy_ledger.get("p_success_horizon")
    )
    if isinstance(prior, (int, float)):
        k_eff = energy_ledger.get("k_eff_s_inv") or physics.get("k_eff_s_inv")
        k_str = f", k_eff={k_eff:.2e} s^-1" if isinstance(k_eff, (int, float)) else ""
        reasons.append(f"Physics prior P={prior:.2f}{k_str}")
    module3_phys = module3.get("module3_physics_audit") or {}
    plan_phys = module3_phys.get("plan_phys")
    snr = module3_phys.get("snr")
    if isinstance(plan_phys, (int, float)):
        snr_str = f", SNR={snr:.1f}" if isinstance(snr, (int, float)) else ""
        reasons.append(f"Plan phys {plan_phys:.2f}{snr_str}")
    mismatch = module2.get("mechanism_mismatch") or {}
    if mismatch.get("status") == "MISMATCH":
        penalty = mismatch.get("penalty_kj_mol")
        if isinstance(penalty, (int, float)):
            reasons.append(f"Mechanism mismatch penalty {penalty:.1f} kJ/mol")
        else:
            reasons.append("Mechanism mismatch flagged")
    return reasons[:8]
