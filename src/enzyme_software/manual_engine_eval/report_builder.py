from __future__ import annotations

import json
from typing import Any, Dict


def build_case_summary(case_record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "case_id": case_record["case"]["case_id"],
        "status": case_record["status"],
        "route": case_record["metrics"]["heuristic_comparison"]["engine_route"],
        "confidence": case_record["metrics"]["confidence"]["reported_confidence"],
        "failure_tags": case_record["failure_tags"],
        "validator_ok": {name: block["ok"] for name, block in case_record["validators"].items()},
    }


def build_console_report(report: Dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "MANUAL ENGINE EVALUATION",
        "=" * 32,
        f"Cases run: {aggregate['cases_run']}",
        f"Status counts: {json.dumps(aggregate['status_counts'], sort_keys=True)}",
        f"Module success: {json.dumps(aggregate['module_success_rates'], sort_keys=True)}",
        f"Failure counts: {json.dumps(aggregate['failure_counts'], sort_keys=True)}",
        f"Confidence summary: {json.dumps(aggregate['confidence_summary'], sort_keys=True)}",
        "",
        "Per-case summary:",
    ]
    for case in report["results"]:
        summary = build_case_summary(case)
        lines.append(
            f"- {summary['case_id']}: {summary['status']} | "
            f"route={summary['route']} | conf={summary['confidence']} | "
            f"failures={','.join(summary['failure_tags'])}"
        )
    return "\n".join(lines)


def build_markdown_report(report: Dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# Manual Engine Evaluation",
        "",
        f"- Cases run: `{aggregate['cases_run']}`",
        f"- Status counts: `{json.dumps(aggregate['status_counts'], sort_keys=True)}`",
        f"- Module success rates: `{json.dumps(aggregate['module_success_rates'], sort_keys=True)}`",
        f"- Failure counts: `{json.dumps(aggregate['failure_counts'], sort_keys=True)}`",
        "",
        "## Cases",
    ]
    for case in report["results"]:
        summary = build_case_summary(case)
        lines.extend(
            [
                f"### {summary['case_id']}",
                f"- Status: `{summary['status']}`",
                f"- Route: `{summary['route']}`",
                f"- Confidence: `{summary['confidence']}`",
                f"- Failures: `{', '.join(summary['failure_tags'])}`",
                f"- Validators: `{json.dumps(summary['validator_ok'], sort_keys=True)}`",
                "",
            ]
        )
    return "\n".join(lines)
