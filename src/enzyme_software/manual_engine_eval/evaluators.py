from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from .benchmark_schema import BenchmarkCase, EvaluationResult
from .failure_taxonomy import assign_failure_tags
from .heuristics import reactive_site_heuristic, route_heuristic, scaffold_heuristic
from .pipeline_runner import run_case_through_pipeline
from .validators import validate_pipeline_result


def _extract_route(run) -> str | None:
    job_card = run.raw_outputs.get("job_card") or {}
    return (
        job_card.get("chosen_route")
        or ((job_card.get("mechanism_route") or {}).get("primary"))
        or None
    )


def _route_debug(run) -> Dict[str, Any]:
    job_card = run.raw_outputs.get("job_card") or {}
    return job_card.get("router_debug") or {}


def _tag_overlap(expected: List[str], text_blobs: Iterable[str]) -> float:
    if not expected:
        return 1.0
    haystack = " ".join(blob.lower() for blob in text_blobs if blob)
    hits = sum(1 for tag in expected if tag.lower() in haystack)
    return hits / float(len(expected))


def _site_match(expected_sites, heuristic_sites) -> Dict[str, Any]:
    if not expected_sites:
        return {"known": False, "top1": None, "top3": None}
    normalized = {str(item) for item in expected_sites}
    pred = [str(item) for item in heuristic_sites]
    return {
        "known": True,
        "top1": bool(pred[:1] and pred[0] in normalized),
        "top3": any(item in normalized for item in pred[:3]),
    }


def evaluate_case(case: BenchmarkCase) -> EvaluationResult:
    run = run_case_through_pipeline(case)
    validators = validate_pipeline_result(run)
    module_minus1 = (run.raw_outputs.get("module_minus1") or {}) if run.ok else {}
    module1 = (run.raw_outputs.get("module1_topogate") or {}) if run.ok else {}
    module2 = (run.raw_outputs.get("module2_active_site_refinement") or {}) if run.ok else {}
    module3 = (run.raw_outputs.get("module3_experiment_designer") or {}) if run.ok else {}

    heuristic_block = {
        "reactive_site": reactive_site_heuristic(case, module_minus1),
        "route": route_heuristic(case, module_minus1),
        "scaffold": scaffold_heuristic(module1),
    }
    route_pred = _extract_route(run)
    site_expected = case.expected_reactive_sites
    site_match = _site_match(site_expected, heuristic_block["reactive_site"]["top_sites"])
    route_match = None
    if case.expected_route_family:
        route_match = str(route_pred or "").lower() == case.expected_route_family.lower()
    mechanism_overlap = _tag_overlap(
        case.expected_mechanism_tags,
        [
            str((run.raw_outputs.get("job_card") or {}).get("reaction_intent") or ""),
            str((run.raw_outputs.get("job_card") or {}).get("bond_context") or ""),
            str(module_minus1.get("mechanism_eligibility") or ""),
        ],
    )
    scaffold_overlap = _tag_overlap(case.expected_scaffold_tags, [str(module1.get("ranked_scaffolds") or "")])
    variant_overlap = _tag_overlap(case.expected_variant_tags, [str(module2.get("best_variant") or ""), str(module2.get("variant_set") or "")])
    experiment_overlap = _tag_overlap(case.expected_experiment_tags, [str(module3.get("protocol_card") or ""), str(module3.get("information_gain") or "")])
    module_success = {name: int(block["ok"]) for name, block in validators.items()}
    confidence = run.confidence_state
    confidence_score = next((confidence.get(key) for key in ("route", "feasibility_if_specified", "confidence_calibrated") if isinstance(confidence.get(key), (int, float))), None)
    route_debug = _route_debug(run)
    metrics = {
        "structural_validity": {
            "pipeline_ok": int(run.ok),
            "module_success_rate": sum(module_success.values()) / float(len(module_success)) if module_success else 0.0,
            "exception": run.exception,
        },
        "correctness": {
            "reactive_site_top1": site_match["top1"],
            "reactive_site_top3": site_match["top3"],
            "route_match": route_match,
            "mechanism_tag_overlap": mechanism_overlap,
            "scaffold_tag_overlap": scaffold_overlap,
            "variant_tag_overlap": variant_overlap,
            "experiment_tag_overlap": experiment_overlap,
        },
        "confidence": {
            "reported_confidence": confidence_score,
            "ambiguity_flag": route_debug.get("ambiguity_flag"),
            "route_gap": route_debug.get("route_gap"),
            "fallback_used": route_debug.get("fallback_used"),
            "calibration_mode": (route_debug.get("calibration") or {}).get("mode"),
        },
        "heuristic_comparison": {
            "heuristic_route": heuristic_block["route"]["route_family"],
            "engine_route": route_pred,
            "heuristic_top_sites": heuristic_block["reactive_site"]["top_sites"],
        },
    }
    failure_tags = assign_failure_tags(run, validators, metrics)
    if not run.ok or any(not block["ok"] for block in validators.values()):
        status = "fail"
    elif any(value in (False, 0.0) for value in [route_match] if value is not None):
        status = "partial"
    else:
        status = "pass"
    return EvaluationResult(
        case=case,
        pipeline=run,
        validators=validators,
        heuristics=heuristic_block,
        metrics=metrics,
        failure_tags=failure_tags,
        status=status,
    )


def evaluate_cases(cases: List[BenchmarkCase], *, repeat: int = 1) -> Dict[str, Any]:
    results: List[EvaluationResult] = []
    repeated_outputs = defaultdict(list)
    for case in cases:
        for _ in range(max(1, int(repeat))):
            result = evaluate_case(case)
            results.append(result)
            repeated_outputs[case.case_id].append(result)

    aggregate_failures = Counter()
    status_counts = Counter(result.status for result in results)
    module_success = Counter()
    module_counts = Counter()
    high_conf_correct = 0
    high_conf_total = 0
    low_conf_correct = 0
    low_conf_total = 0
    ambiguity_total = 0
    fallback_total = 0
    repeatability = {}

    for result in results:
        for tag in result.failure_tags:
            aggregate_failures[tag] += 1
        for module_name, block in result.validators.items():
            module_counts[module_name] += 1
            module_success[module_name] += int(block["ok"])
        confidence = result.metrics["confidence"]["reported_confidence"]
        route_match = result.metrics["correctness"]["route_match"]
        ambiguity_total += int(bool(result.metrics["confidence"].get("ambiguity_flag")))
        fallback_total += int(bool(result.metrics["confidence"].get("fallback_used")))
        if isinstance(confidence, (int, float)) and route_match is not None:
            if confidence >= 0.7:
                high_conf_total += 1
                high_conf_correct += int(bool(route_match))
            elif confidence <= 0.3:
                low_conf_total += 1
                low_conf_correct += int(bool(route_match))

    for case_id, case_results in repeated_outputs.items():
        routes = [item.metrics["heuristic_comparison"]["engine_route"] for item in case_results]
        repeatability[case_id] = {
            "repeats": len(case_results),
            "unique_routes": len(set(routes)),
            "deterministic": len(set(routes)) <= 1,
        }

    return {
        "results": [result.to_dict() for result in results],
        "aggregate": {
            "cases_run": len(results),
            "status_counts": dict(status_counts),
            "module_success_rates": {
                name: module_success[name] / float(module_counts[name]) if module_counts[name] else 0.0
                for name in module_counts
            },
            "failure_counts": dict(aggregate_failures),
            "confidence_summary": {
                "high_conf_accuracy": high_conf_correct / float(high_conf_total) if high_conf_total else None,
                "low_conf_accuracy": low_conf_correct / float(low_conf_total) if low_conf_total else None,
                "ambiguity_rate": ambiguity_total / float(len(results)) if results else None,
                "fallback_rate": fallback_total / float(len(results)) if results else None,
            },
            "repeatability": repeatability,
        },
    }
