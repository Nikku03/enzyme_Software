from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

from enzyme_software.config import (
    RETENTION_WEAK_THRESHOLD,
    ROUTE_CONFIDENCE_LOW_THRESHOLD,
)


def render_pretty(payload: Dict[str, Any]) -> str:
    """Render a compact PI-facing summary (no raw JSON or warnings)."""
    normalized = _normalize_payload(payload)
    job_card = normalized["job_card"]
    module_minus1 = normalized["module_minus1"]
    module1 = normalized["module1"]
    module2 = normalized["module2"]
    module3 = normalized["module3"]

    lines: List[str] = []
    lines.extend(_pi_narrative_summary(normalized))
    lines.append("")

    decision = job_card.get("decision") or "UNKNOWN"
    route = (
        (job_card.get("mechanism_route") or {}).get("primary")
        or job_card.get("chosen_route")
        or "n/a"
    )
    confidence = job_card.get("confidence") or {}
    route_conf = _format_float(confidence.get("route"))
    target_resolution = _format_float(confidence.get("target_resolution"))
    target_bond = normalized["target_bond"]

    lines.append("Summary")
    lines.append(f"- Decision: {decision}")
    lines.append(f"- Bond: {target_bond}")
    lines.append(f"- Route: {route}")
    lines.append(f"- Confidence: route={route_conf}, target_resolution={target_resolution}")
    lines.append("")

    if module_minus1:
        m1_status = module_minus1.get("status") or "n/a"
        m1_constraint = module_minus1.get("primary_constraint") or "NONE"
        m1_conf = _format_float(module_minus1.get("confidence_prior"))
        lines.append("Substrate pre-screening (Module -1)")
        lines.append(f"- Status: {m1_status}")
        lines.append(f"- Primary constraint: {m1_constraint}")
        lines.append(f"- Confidence prior: {m1_conf}")
        rx = module_minus1.get("reactivity") or {}
        epav = _format_float(rx.get("epav_score"))
        if epav != "n/a":
            lines.append(f"- EP-AV score: {epav}")
        lines.append("")

    lines.append("Top scaffolds")
    top_scaffolds = [
        scaffold.get("scaffold_id")
        for scaffold in (module1.get("ranked_scaffolds") or [])[:3]
        if scaffold.get("scaffold_id")
    ]
    if top_scaffolds:
        for scaffold in top_scaffolds:
            lines.append(f"- {scaffold}")
    else:
        lines.append("- None available")
    lines.append("")

    lines.append("Top variant")
    best_variant = module2.get("best_variant") or {}
    best_label = best_variant.get("label") or best_variant.get("variant_id") or "n/a"
    lines.append(f"- {best_label}")
    lines.append("")

    protocol = module3.get("protocol_card") or {}
    arms = protocol.get("arms") or []
    controls = protocol.get("controls") or {}
    lines.append("Experiment plan")
    lines.append(f"- Arms: {len(arms)}")
    if controls.get("negative_control_arm_id"):
        lines.append(f"- Negative control: {controls.get('negative_control_arm_id')}")
    for arm in arms[:5]:
        arm_id = arm.get("arm_id") or arm.get("type")
        arm_type = arm.get("type") or "arm"
        conditions = arm.get("conditions") or {}
        cond_str = ", ".join(
            f"{key}={value}"
            for key, value in conditions.items()
            if value is not None
        )
        if cond_str:
            lines.append(f"- {arm_id} ({arm_type}): {cond_str}")
        else:
            lines.append(f"- {arm_id} ({arm_type})")

    return "\n".join(lines)


def render_demo(payload: Dict[str, Any]) -> str:
    """Render a compact, non-technical demo report (no raw JSON)."""
    normalized = _normalize_payload(payload)
    job_card = normalized["job_card"]
    module1 = normalized["module1"]
    module2 = normalized["module2"]
    module3 = normalized["module3"]

    decision = job_card.get("decision") or "UNKNOWN"
    route = (
        (job_card.get("mechanism_route") or {}).get("primary")
        or job_card.get("chosen_route")
        or ((normalized["shared_state"].get("bio") or {}).get("derived") or {}).get(
            "mechanism_route"
        )
        or "n/a"
    )
    target_bond = normalized["target_bond"]
    confidence = job_card.get("confidence") or {}
    route_conf = _format_float(confidence.get("route"))
    target_resolution = _format_float(confidence.get("target_resolution"))
    target_resolution = _format_float(confidence.get("target_resolution"))
    target_resolution = _format_float(confidence.get("target_resolution"))
    physics_prior = _format_float(
        (job_card.get("physics_audit") or {}).get("prior_success_probability_final")
        or (job_card.get("physics_audit") or {}).get("route_prior_target_specific")
        or ((normalized["shared_state"].get("physics") or {}).get("derived") or {}).get(
            "prior_success_probability"
        )
    )

    top_scaffolds = [
        scaffold.get("scaffold_id")
        for scaffold in (module1.get("ranked_scaffolds") or [])[:3]
        if scaffold.get("scaffold_id")
    ]
    selected_variant = (
        (module2.get("best_variant") or {}).get("label")
        or (module2.get("best_variant") or {}).get("variant_id")
        or "n/a"
    )
    protocol = module3.get("protocol_card") or {}
    arms = protocol.get("arms") or []
    controls = protocol.get("controls") or {}
    negative_control = controls.get("negative_control_arm_id")

    rationale_notes = (module3.get("information_gain") or {}).get("notes") or []
    rationale = rationale_notes[0] if rationale_notes else "Plan balances information gain and detectability."

    lines: List[str] = []
    lines.append("Demo summary")
    lines.append(f"- Decision: {decision}")
    lines.append(f"- Route: {route}")
    lines.append(f"- Bond: {target_bond}")
    lines.append("")
    lines.append("Confidence breakdown")
    lines.append(f"- Route confidence: {route_conf}")
    lines.append(f"- Target resolution: {target_resolution}")
    lines.append(f"- Physics feasibility: {physics_prior}")
    lines.append("")
    lines.append("Top scaffolds")
    if top_scaffolds:
        for scaffold in top_scaffolds:
            lines.append(f"- {scaffold}")
    else:
        lines.append("- None available")
    lines.append("")
    lines.append("Chosen variant")
    lines.append(f"- {selected_variant}")
    lines.append("")
    lines.append("Experiment plan")
    lines.append(f"- Arms: {len(arms)}")
    if negative_control:
        lines.append(f"- Negative control: {negative_control}")
    if arms:
        for arm in arms[:4]:
            arm_id = arm.get("arm_id") or arm.get("type")
            arm_type = arm.get("type") or "arm"
            conditions = arm.get("conditions") or {}
            cond_str = ", ".join(
                f"{key}={value}"
                for key, value in conditions.items()
                if value is not None
            )
            if cond_str:
                lines.append(f"- {arm_id} ({arm_type}): {cond_str}")
            else:
                lines.append(f"- {arm_id} ({arm_type})")
    lines.append("")
    lines.append("Rationale")
    lines.append(f"- {rationale}")
    return "\n".join(lines)


def render_scientist(payload: Dict[str, Any]) -> str:
    """Render a detailed scientist view with score ledger + audits."""
    normalized = _normalize_payload(payload)
    job_card = normalized["job_card"]
    shared_state = normalized["shared_state"]
    module_minus1 = normalized["module_minus1"]
    module1 = normalized["module1"]
    module2 = normalized["module2"]
    module3 = normalized["module3"]
    summary = normalized["pipeline_summary"]

    route = (
        (job_card.get("mechanism_route") or {}).get("primary")
        or job_card.get("chosen_route")
        or "n/a"
    )
    target_bond = normalized["target_bond"]
    confidence = job_card.get("confidence") or {}
    route_conf = _format_float(confidence.get("route"))
    target_resolution = _format_float(confidence.get("target_resolution"))
    decision = job_card.get("decision") or "UNKNOWN"
    halt_reason = job_card.get("pipeline_halt_reason")

    mechanism_contract = (
        (shared_state.get("mechanism") or {}).get("derived", {}).get("contract")
        or job_card.get("mechanism_contract")
        or module2.get("mechanism_contract")
        or {}
    )
    mechanism_mismatch = (
        (shared_state.get("mechanism") or {}).get("derived", {}).get("mismatch")
        or module2.get("mechanism_mismatch")
        or {}
    )
    mechanism_evidence = (
        (shared_state.get("mechanism") or {}).get("derived", {}).get("evidence")
        or module2.get("mechanism_evidence")
        or {}
    )
    target_resolution_audit = job_card.get("target_resolution_audit") or {}

    energy_ledger = (
        ((shared_state.get("physics") or {}).get("derived") or {}).get("energy_ledger")
        or {}
    )
    delta_g = _format_float(energy_ledger.get("deltaG_dagger_kJ"))
    k_eyring = _format_rate(energy_ledger.get("eyring_k_s_inv"))
    k_eff = _format_rate(energy_ledger.get("k_eff_s_inv"))

    chem = (shared_state.get("chem") or {}).get("derived") or {}
    reaction_family = chem.get("reaction_family") or "n/a"
    leaving_group = _format_float(chem.get("leaving_group_score"))

    bio = (shared_state.get("bio") or {}).get("derived") or {}
    protonation = bio.get("protonation") or {}
    protonation_factor = _format_float(protonation.get("factor"))

    risks = _collect_risks(job_card, module1, module2, module3, mechanism_mismatch, summary)
    next_actions = _next_actions(module3)
    why_this_score = summary.get("why_this_score") or []

    lines: List[str] = []
    lines.extend(_pi_narrative_summary(normalized))
    lines.append("")
    lines.append("Decision summary")
    lines.append(f"- Decision: {decision}")
    if halt_reason:
        lines.append(f"- Halt reason: {halt_reason}")
    lines.append(f"- Route: {route}")
    lines.append(f"- Target bond: {target_bond}")
    lines.append(f"- Route confidence: {route_conf}")
    lines.append(f"- Target resolution: {target_resolution}")
    lines.append("")
    if module_minus1:
        m1_status = module_minus1.get("status") or "n/a"
        m1_constraint = module_minus1.get("primary_constraint") or "NONE"
        m1_conf = _format_float(module_minus1.get("confidence_prior"))
        rx = module_minus1.get("reactivity") or {}
        rt = module_minus1.get("resolved_target") or {}
        lines.append("Substrate reactivity (Module -1)")
        lines.append(f"- Status: {m1_status}")
        lines.append(f"- Primary constraint: {m1_constraint}")
        lines.append(f"- Confidence prior: {m1_conf}")
        lines.append(f"- Bond type: {rt.get('bond_type', 'n/a')}")
        bond_idx = rt.get("bond_indices")
        if isinstance(bond_idx, list) and len(bond_idx) == 2:
            lines.append(f"- Bond indices: {bond_idx[0]}-{bond_idx[1]}")
        attack = rt.get("attack_sites") or {}
        if attack.get("electrophile") is not None:
            lines.append(f"- Electrophile index: {attack['electrophile']}")
        if attack.get("leaving_group") is not None:
            lines.append(f"- Leaving group index: {attack['leaving_group']}")
        epav_score = _format_float(rx.get("epav_score"))
        if epav_score != "n/a":
            lines.append(f"- EP-AV score: {epav_score}")
        l2_score = _format_float(rx.get("level2_score"))
        if l2_score != "n/a":
            lines.append(f"- Level 2 steric score: {l2_score}")
        l3_score = _format_float(rx.get("level3_score"))
        if l3_score != "n/a":
            lines.append(f"- Level 3 composite score: {l3_score}")
        mech_elig = module_minus1.get("mechanism_eligibility") or {}
        if mech_elig:
            eligible = [k for k, v in mech_elig.items() if v == "eligible"]
            if eligible:
                lines.append(f"- Eligible mechanisms: {', '.join(eligible)}")
        lines.append("")
    lines.append("Mechanism contract")
    lines.append(
        f"- Expected nucleophile: {mechanism_contract.get('expected_nucleophile', 'n/a')}"
    )
    lines.append(
        f"- Allowed geometries: {', '.join(mechanism_contract.get('allowed_nucleophile_geometries') or []) or 'n/a'}"
    )
    if mechanism_evidence:
        status = mechanism_evidence.get("status", "n/a")
        explanation = mechanism_evidence.get("explanation", "n/a")
        residues = mechanism_evidence.get("evidence_residues") or []
        residue_str = ", ".join(residues) if residues else "none"
        lines.append(f"- Evidence status: {status} ({explanation})")
        lines.append(f"- Evidence residues: {residue_str}")
    if mechanism_mismatch:
        lines.append(
            f"- Match status: {mechanism_mismatch.get('status', 'n/a')} ({mechanism_mismatch.get('explanation', 'n/a')})"
        )
    lines.append("")
    if target_resolution_audit:
        lines.append("Target resolution audit")
        lines.append(
            f"- match_count: {target_resolution_audit.get('match_count', 'n/a')}"
        )
        lines.append(
            f"- top_score: {_format_float(target_resolution_audit.get('top_score'))}"
        )
        lines.append(
            f"- score_gap: {_format_float(target_resolution_audit.get('score_gap'))}"
        )
        lines.append(
            f"- confidence: {_format_float(target_resolution_audit.get('confidence'))}"
        )
        lines.append(
            f"- ambiguous: {target_resolution_audit.get('ambiguous', False)}"
        )
        lines.append("")
    lines.append("Score ledger summary")
    for module_label, ledger in _collect_score_ledgers(normalized).items():
        if not ledger:
            continue
        lines.append(f"- {module_label}:")
        for term in ledger:
            lines.append(f"  - {term}")
    lines.append("")
    lines.append("Physics audit")
    lines.append(f"- Baseline ΔG‡ (kJ/mol): {delta_g}")
    lines.append(f"- Baseline k_Eyring (s^-1): {k_eyring}")
    lines.append(f"- Baseline k_eff (s^-1): {k_eff}")
    if energy_ledger.get("deltaG_dagger_variant_kJ") is not None:
        lines.append(
            f"- Variant ΔG‡ (kJ/mol): {_format_float(energy_ledger.get('deltaG_dagger_variant_kJ'))}"
        )
    if energy_ledger.get("k_variant_s_inv") is not None:
        lines.append(
            f"- Variant k (s^-1): {_format_rate(energy_ledger.get('k_variant_s_inv'))}"
        )
    prior_used = _format_float(
        (job_card.get("physics_audit") or {}).get("prior_success_probability_final")
        or (job_card.get("physics_audit") or {}).get("route_prior_target_specific")
    )
    lines.append(f"- Prior success (target-specific): {prior_used}")
    calib_sources = (job_card.get("physics_audit") or {}).get("calibration_sources")
    calib_samples = (job_card.get("physics_audit") or {}).get("calibration_samples")
    data_support = job_card.get("data_support")
    lines.append("")
    lines.append("Evidence summary")
    lines.append(f"- Data support: {_format_float(data_support)}")
    if calib_sources or calib_samples:
        lines.append(f"- Calibration sources: {calib_sources or 'n/a'}")
        lines.append(f"- Calibration samples: {calib_samples or 'n/a'}")
    lines.append("")
    lines.append("Key highlights")
    lines.append(f"- Reaction family: {reaction_family}")
    lines.append(f"- Leaving-group score: {leaving_group}")
    lines.append(f"- Protonation factor: {protonation_factor}")
    lines.append("")
    lines.append("Top risks")
    if risks:
        for risk in risks[:3]:
            lines.append(f"- {risk}")
    else:
        lines.append("- No major risks detected.")
    if why_this_score:
        lines.append("")
        lines.append("Why this score")
        for item in why_this_score[:8]:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("Next best actions")
    if next_actions:
        for action in next_actions:
            lines.append(f"- {action}")
    else:
        lines.append("- No actions available (missing module 3 plan).")
    return "\n".join(lines)


def render_debug(payload: Dict[str, Any]) -> str:
    """Render a verbose debug report with shared_state/shared_io dumps."""
    normalized = _normalize_payload(payload)
    data = normalized["data"]
    pipeline_summary = normalized["pipeline_summary"]
    shared_state = normalized["shared_state"]
    shared_io = normalized["shared_io"]
    module_minus1 = normalized["module_minus1"]
    module1 = normalized["module1"]
    module2 = normalized["module2"]
    module3 = normalized["module3"]
    job_card = normalized["job_card"]
    cache_stats = module1.get("cache_stats") or {}
    module1_weights = module1.get("weights") or job_card.get("module1_weights") or {}
    route_posteriors = job_card.get("route_posteriors") or []
    warnings = {
        "module_minus1": module_minus1.get("warnings") or [],
        "module0": job_card.get("warnings") or [],
        "module1": module1.get("warnings") or [],
        "module2": module2.get("warnings") or [],
        "module3": module3.get("warnings") or [],
    }

    lines = []
    lines.append("Debug report")
    lines.append("Flags")
    lines.append(f"- Pipeline summary: {json.dumps(pipeline_summary, ensure_ascii=False)}")
    lines.append(f"- Module 1 cache stats: {json.dumps(cache_stats, ensure_ascii=False)}")
    lines.append(f"- Module 1 weights: {json.dumps(module1_weights, ensure_ascii=False)}")
    if route_posteriors:
        lines.append(f"- Route posteriors: {json.dumps(route_posteriors, ensure_ascii=False)}")
    lines.append(f"- Module 2 status: {module2.get('status')}")
    lines.append(f"- Module 3 status: {module3.get('status')}")
    lines.append(f"- Warnings: {json.dumps(warnings, ensure_ascii=False)}")
    lines.append("")
    if module_minus1:
        lines.append("Module -1 (SRE) JSON")
        m1_debug = {
            "status": module_minus1.get("status"),
            "primary_constraint": module_minus1.get("primary_constraint"),
            "confidence_prior": module_minus1.get("confidence_prior"),
            "resolved_target": module_minus1.get("resolved_target"),
            "reactivity": module_minus1.get("reactivity"),
            "cpt_scores": module_minus1.get("cpt_scores"),
            "mechanism_eligibility": module_minus1.get("mechanism_eligibility"),
            "route_bias": module_minus1.get("route_bias"),
            "warnings": module_minus1.get("warnings", []),
            "errors": module_minus1.get("errors", []),
        }
        lines.append(json.dumps(m1_debug, indent=2, ensure_ascii=False))
        lines.append("")
    lines.append("Shared state JSON")
    lines.append(json.dumps(shared_state, indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("Shared IO JSON")
    lines.append(json.dumps(shared_io, indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("RAW JSON")
    lines.append(json.dumps(data or payload, indent=2, ensure_ascii=False))
    return "\n".join(lines)


def render_scientific_report(pipeline_result: Dict[str, Any]) -> str:
    """Backward-compatible wrapper for the scientist view."""
    return render_scientist(pipeline_result)


def render_debug_report(pipeline_result: Dict[str, Any]) -> str:
    """Backward-compatible wrapper for the debug view."""
    return render_debug(pipeline_result)


def _pi_narrative_summary(normalized: Dict[str, Any]) -> List[str]:
    job_card = normalized["job_card"]
    module3 = normalized["module3"]

    route = (
        (job_card.get("mechanism_route") or {}).get("primary")
        or job_card.get("chosen_route")
        or "n/a"
    )
    target_bond = normalized["target_bond"]
    confidence = job_card.get("confidence") or {}
    route_conf = _format_float(confidence.get("route"))
    target_resolution = _format_float(confidence.get("target_resolution"))
    token_audit = job_card.get("token_resolution_audit") or {}
    token_method = token_audit.get("method")
    token_match_count = token_audit.get("match_count")
    token_match = "n/a"
    if token_method:
        token_match = f"{token_method} (match_count={token_match_count})"
    physics_prior = _format_float(
        (job_card.get("physics_audit") or {}).get("prior_success_probability_final")
        or (job_card.get("physics_audit") or {}).get("route_prior_target_specific")
        or ((normalized["shared_state"].get("physics") or {}).get("derived") or {}).get(
            "prior_success_probability"
        )
    )

    protocol = module3.get("protocol_card") or {}
    arms = protocol.get("arms") or []
    arm_names = [arm.get("arm_id") or arm.get("type") for arm in arms[:3]]

    lines: List[str] = []
    lines.append("PI narrative summary")
    lines.append(f"- Hypothesis: {route} on {target_bond}")
    lines.append(
        f"- Evidence: token match={token_match}, route_conf={route_conf}, target_res={target_resolution}, physics_prior={physics_prior}"
    )
    if arm_names:
        lines.append(f"- Next experiment: {', '.join(arm_names)}")
    else:
        lines.append("- Next experiment: n/a")
    return lines


def _collect_risks(
    job_card: Dict[str, Any],
    module1: Dict[str, Any],
    module2: Dict[str, Any],
    module3: Dict[str, Any],
    mechanism_spec: Dict[str, Any],
    summary: Dict[str, Any],
) -> List[str]:
    risks: List[str] = []
    decision = job_card.get("decision")
    if decision and str(decision).startswith("HALT"):
        risks.append(f"Routing halted ({job_card.get('pipeline_halt_reason') or 'unspecified'})")
    mismatch_reason = mechanism_spec.get("mismatch_reason") or mechanism_spec.get("explanation")
    mismatch_status = mechanism_spec.get("status")
    mechanism_evidence = module2.get("mechanism_evidence") or {}
    if mechanism_evidence.get("status") == "UNVERIFIED":
        risks.append("Mechanism evidence unverified (no detected nucleophile)")
    if mismatch_reason:
        risks.append(f"Mechanism mismatch: {mismatch_reason}")
    elif mismatch_status == "MISMATCH":
        risks.append("Mechanism mismatch flagged")
    module1_status = module1.get("status")
    if module1_status and module1_status != "PASS":
        risks.append(f"Module 1 status {module1_status}")
    retention = (module1.get("module1_confidence") or {}).get("retention")
    if retention is None:
        retention = (module1.get("module1_confidence") or {}).get("retention_mean")
    if isinstance(retention, (int, float)) and retention < RETENTION_WEAK_THRESHOLD.value:
        risks.append(f"Retention weak (mean={_format_float(retention)})")
    route_conf = (job_card.get("confidence") or {}).get("route")
    if isinstance(route_conf, (int, float)) and route_conf < ROUTE_CONFIDENCE_LOW_THRESHOLD.value:
        risks.append(f"Route confidence low ({_format_float(route_conf)})")
    if module2.get("status") not in {None, "PASS", "DEGRADED_OK"}:
        risks.append(f"Module 2 status {module2.get('status')}")
    qc_status = module3.get("qc_status") or (module3.get("qc_result") or {}).get("status")
    if qc_status == "FAIL":
        risks.append("Module 3 QC failed")
    if not risks:
        needs = summary.get("needs_improvement") or []
        risks.extend([str(item) for item in needs][:3])
    return risks


def _next_actions(module3: Dict[str, Any]) -> List[str]:
    protocol = module3.get("protocol_card") or {}
    arms = protocol.get("arms") or []
    actions = []
    for arm in arms[:3]:
        arm_id = arm.get("arm_id") or arm.get("type")
        conditions = arm.get("conditions") or {}
        cond_str = ", ".join(
            f"{key}={value}"
            for key, value in conditions.items()
            if value is not None
        )
        if cond_str:
            actions.append(f"Run {arm_id} ({arm.get('type')}) at {cond_str}")
        else:
            actions.append(f"Run {arm_id} ({arm.get('type')})")
    return actions


def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("data") is not None:
        data = payload.get("data") or {}
    else:
        data = payload
    if "input_spec" in data and "physics" in data and "chem" in data:
        shared_state = data
        job_card = data.get("job_card") or {}
        return {
            "data": {"shared_state": shared_state},
            "job_card": job_card,
            "shared_state": shared_state,
            "shared_io": data.get("shared_io") or {},
            "module_minus1": {},
            "module1": {},
            "module2": {},
            "module3": {},
            "pipeline_summary": data.get("audit") or {},
            "target_bond": (shared_state.get("input_spec") or {}).get("target_bond") or "n/a",
        }
    job_card = (
        data.get("job_card")
        or (data.get("module0_strategy_router") or {}).get("job_card")
        or data.get("module0", {}).get("job_card")
        or {}
    )
    shared_state = data.get("shared_state") or payload.get("shared_state") or {}
    shared_io = data.get("shared_io") or payload.get("shared_io") or {}
    module_minus1 = (
        (shared_io.get("outputs") or {}).get("module_minus1", {}).get("result")
        or data.get("module_minus1")
        or {}
    )
    module1 = data.get("module1_topogate") or {}
    module2 = data.get("module2_active_site_refinement") or {}
    module3 = data.get("module3_experiment_designer") or {}
    pipeline_summary = data.get("pipeline_summary") or {}
    target_bond = payload.get("target_bond") or job_card.get("target_bond") or "n/a"
    return {
        "data": data,
        "job_card": job_card,
        "shared_state": shared_state,
        "shared_io": shared_io,
        "module_minus1": module_minus1,
        "module1": module1,
        "module2": module2,
        "module3": module3,
        "pipeline_summary": pipeline_summary,
        "target_bond": target_bond,
    }


def _collect_score_ledgers(normalized: Dict[str, Any]) -> Dict[str, List[str]]:
    ledgers: Dict[str, List[str]] = {}
    job_card = normalized["job_card"]
    module1 = normalized["module1"]
    module2 = normalized["module2"]
    module3 = normalized["module3"]

    ledger_entries = [
        ("Module 0", job_card.get("score_ledger")),
        ("Module 1", module1.get("score_ledger")),
        ("Module 2", module2.get("score_ledger")),
        ("Module 3", module3.get("score_ledger")),
    ]
    for label, ledger in ledger_entries:
        if not ledger:
            continue
        terms = ledger.get("terms") or []
        snippets: List[str] = []
        for term in terms[:3]:
            name = term.get("name")
            value = term.get("value")
            unit = term.get("unit") or ""
            formatted = _format_float(value)
            suffix = f" {unit}".rstrip()
            snippets.append(f"{name}: {formatted}{suffix}")
        ledgers[label] = snippets
    return ledgers


def _format_float(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "n/a"


def _format_rate(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    abs_val = abs(value)
    if abs_val != 0 and abs_val < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"
