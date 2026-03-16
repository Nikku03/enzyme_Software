from __future__ import annotations

from typing import Any, Dict, List


def assign_failure_tags(run, validators: Dict[str, Dict[str, Any]], metrics: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    if not run.ok:
        tags.append("exception_failure")
        return tags
    if not validators["module_minus1"]["ok"]:
        tags.append("target_bond_resolution_failure")
    if not validators["module0"]["ok"]:
        tags.append("route_misclassification")
    if metrics.get("correctness", {}).get("route_match") is False:
        tags.append("route_misclassification")
    if not validators["module1"]["ok"]:
        tags.append("scaffold_ranking_failure")
    if not validators["module2"]["ok"]:
        tags.append("variant_selection_failure")
    if not validators["module3"]["ok"]:
        tags.append("experiment_design_failure")
    if not validators["shared"]["ok"]:
        tags.append("cross_module_contradiction")
    if any("missing" in err for block in validators.values() for err in block["errors"]):
        tags.append("incomplete_output")
    confidence = metrics.get("confidence", {}).get("reported_confidence")
    route_match = metrics.get("correctness", {}).get("route_match")
    if isinstance(confidence, (int, float)) and confidence >= 0.8 and route_match is False:
        tags.append("overconfident_wrong_answer")
        tags.append("confidence_miscalibration")
    if not tags:
        tags.append("none")
    return sorted(set(tags))
