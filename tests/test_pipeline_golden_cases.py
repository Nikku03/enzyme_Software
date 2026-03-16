"""Golden-case pipeline regression harness driven by JSON fixtures.

One test runs all fixture cases and emits a single consolidated failure report
with:
- case id
- likely diverged module
- key mismatches
- halt/rejection details
- candidate-bond table for fast manual diagnosis
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from enzyme_software.pipeline import run_pipeline


CASES_DIR = Path(__file__).resolve().parent / "fixtures" / "cases"
DEFAULT_REPORT_PATH = Path("artifacts") / "golden_cases_report.json"


def _load_cases() -> List[Dict[str, Any]]:
    paths = sorted(CASES_DIR.glob("*.json"))
    assert paths, f"No fixtures found under {CASES_DIR}"
    cases: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["_fixture_path"] = str(path)
        cases.append(payload)
    return cases


def _extract_observed(ctx: Any) -> Dict[str, Any]:
    summary = ctx.data.get("pipeline_summary") or {}
    results = summary.get("results") or {}
    shared = ctx.data.get("shared_io") or {}
    m1 = (shared.get("outputs") or {}).get("module_minus1") or {}
    m1_result = m1.get("result") or (ctx.data.get("module_minus1") or {})
    m1_resolved = m1_result.get("resolved_target") or {}
    m1_cpt = m1.get("cpt") or m1_result.get("cpt_scores") or {}
    job_card = ctx.data.get("job_card") or {}
    m2 = ctx.data.get("module2_active_site_refinement") or {}
    variants = m2.get("variant_set") or []
    variant0 = variants[0] if variants else {}
    resolved0 = job_card.get("resolved_target") or {}
    candidate_options = resolved0.get("candidate_bond_options") or []
    rejection_reasons = sorted(
        {
            str(opt.get("rejection_reason"))
            for opt in candidate_options
            if isinstance(opt, dict) and opt.get("rejection_reason")
        }
    )
    return {
        "decision": results.get("decision"),
        "halt_reason": results.get("halt_reason"),
        "route": results.get("route"),
        "module_minus1_status": results.get("module_minus1_status"),
        "module_minus1_track": m1_cpt.get("track"),
        "module_minus1_match_count": m1_resolved.get("match_count"),
        "module_minus1_resolution_policy": m1_resolved.get("resolution_policy"),
        "module_minus1_next_input_required": m1_resolved.get("next_input_required") or [],
        "candidate_bonds": m1_resolved.get("candidate_bonds") or [],
        "enzyme_family": (job_card.get("biology_audit") or {}).get("enzyme_family"),
        "physics_layer": job_card.get("physics_layer") or {},
        "variant0_label": variant0.get("label"),
        "variant0_policy": variant0.get("variant_policy"),
        "candidate_rejection_reasons": rejection_reasons,
    }


def _candidate_table(candidates: List[Dict[str, Any]]) -> str:
    if not candidates:
        return "  (no candidates)"
    header = "  rank | atom_indices | role/class | subclass | score"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for cand in candidates[:10]:
        rank = cand.get("rank") or cand.get("priority_rank")
        atom_indices = cand.get("atom_indices")
        role = cand.get("primary_role") or cand.get("bond_class") or cand.get("role")
        subclass = cand.get("subclass")
        score = cand.get("score_total", cand.get("score"))
        lines.append(
            f"  {rank!s:>4} | {str(atom_indices):<12} | {str(role):<10} | {str(subclass):<8} | {score}"
        )
    return "\n".join(lines)


def _check_expectations(case_id: str, expect: Dict[str, Any], obs: Dict[str, Any]) -> List[str]:
    mismatches: List[str] = []

    def _in(key: str, actual: Any) -> None:
        allowed = expect.get(key)
        if allowed is None:
            return
        if actual not in allowed:
            mismatches.append(f"{key}: expected one of {allowed}, got {actual}")

    _in("decision_in", obs["decision"])
    _in("halt_reason_in", obs["halt_reason"])
    _in("route_in", obs["route"])
    _in("enzyme_family_in", obs["enzyme_family"])

    if "module_minus1_status" in expect and obs["module_minus1_status"] != expect["module_minus1_status"]:
        mismatches.append(
            f"module_minus1_status: expected {expect['module_minus1_status']}, got {obs['module_minus1_status']}"
        )
    if "module_minus1_track" in expect and obs["module_minus1_track"] != expect["module_minus1_track"]:
        mismatches.append(
            f"module_minus1_track: expected {expect['module_minus1_track']}, got {obs['module_minus1_track']}"
        )
    if "exact_match_count" in expect:
        actual = int(obs["module_minus1_match_count"] or 0)
        if actual != int(expect["exact_match_count"]):
            mismatches.append(f"match_count: expected {expect['exact_match_count']}, got {actual}")
    if "min_match_count" in expect:
        actual = int(obs["module_minus1_match_count"] or 0)
        if actual < int(expect["min_match_count"]):
            mismatches.append(f"match_count: expected >= {expect['min_match_count']}, got {actual}")
    if "resolution_policy_prefix" in expect:
        prefix = str(expect["resolution_policy_prefix"])
        actual = str(obs["module_minus1_resolution_policy"] or "")
        if not actual.startswith(prefix):
            mismatches.append(f"resolution_policy_prefix: expected prefix {prefix}, got {actual}")
    if "variant_policy" in expect:
        actual = obs["variant0_policy"]
        if actual != expect["variant_policy"]:
            mismatches.append(f"variant_policy: expected {expect['variant_policy']}, got {actual}")
    for forbidden in expect.get("variant_label_not_contains", []) or []:
        label = str(obs["variant0_label"] or "")
        if forbidden.lower() in label.lower():
            mismatches.append(f"variant0_label contains forbidden text '{forbidden}': {label}")
    for field_name in expect.get("physics_non_null_fields", []) or []:
        val = (obs.get("physics_layer") or {}).get(field_name)
        if val is None:
            mismatches.append(f"physics_layer.{field_name}: expected non-null, got None")

    # Safety check: if Module -1 requested additional input, decision should not claim GO.
    if obs["module_minus1_next_input_required"] and obs["decision"] in {"GO", "LOW_CONF_GO"}:
        mismatches.append(
            "module_minus1_next_input_required non-empty while decision is GO/LOW_CONF_GO"
        )
    return mismatches


def _guess_diverged_module(mismatches: List[str]) -> str:
    text = " | ".join(mismatches).lower()
    if any(k in text for k in ["module_minus1", "match_count", "resolution_policy"]):
        return "Module -1"
    if any(k in text for k in ["decision", "halt_reason", "route", "enzyme_family"]):
        return "Module 0"
    if any(k in text for k in ["variant", "module2"]):
        return "Module 2"
    return "Unknown"


def test_pipeline_golden_cases() -> None:
    cases = _load_cases()
    failures: List[str] = []
    report_rows: List[Dict[str, Any]] = []

    for case in cases:
        case_id = case.get("case_id") or Path(case.get("_fixture_path", "")).stem
        inputs = case.get("input") or {}
        expect = case.get("expect") or {}

        ctx = run_pipeline(
            smiles=str(inputs.get("smiles") or ""),
            target_bond=str(inputs.get("target_bond") or ""),
            requested_output=inputs.get("requested_output"),
            trap_target=inputs.get("trap_target"),
        )
        obs = _extract_observed(ctx)
        mismatches = _check_expectations(case_id, expect, obs)
        report_rows.append(
            {
                "case_id": case_id,
                "fixture_path": case.get("_fixture_path"),
                "input": inputs,
                "expect": expect,
                "observed": obs,
                "mismatches": mismatches,
                "status": "fail" if mismatches else "pass",
            }
        )
        if mismatches:
            divergence = _guess_diverged_module(mismatches)
            report = [
                f"[{case_id}] fixture={case.get('_fixture_path')}",
                f"diverged_module={divergence}",
                f"decision={obs['decision']} halt_reason={obs['halt_reason']} route={obs['route']}",
                f"module_minus1_status={obs['module_minus1_status']} track={obs['module_minus1_track']}",
                f"module_minus1_policy={obs['module_minus1_resolution_policy']} match_count={obs['module_minus1_match_count']}",
                f"variant0={obs['variant0_label']} policy={obs['variant0_policy']}",
                f"candidate_rejection_reasons={obs['candidate_rejection_reasons']}",
                "mismatches:",
            ]
            report.extend([f"  - {item}" for item in mismatches])
            report.append("candidate_bond_table:")
            report.append(_candidate_table(obs["candidate_bonds"]))
            failures.append("\n".join(report))

    output_path = Path(os.environ.get("GOLDEN_CASES_JSON", str(DEFAULT_REPORT_PATH)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_cases": len(report_rows),
        "passed": sum(1 for row in report_rows if row["status"] == "pass"),
        "failed": sum(1 for row in report_rows if row["status"] == "fail"),
        "cases": report_rows,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    if failures:
        pytest.fail(f"JSON report: {output_path}\n\n" + "\n\n".join(failures))
