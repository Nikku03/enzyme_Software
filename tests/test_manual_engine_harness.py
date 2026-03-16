from __future__ import annotations

import json

from enzyme_software.manual_engine_eval.benchmark_loader import demo_benchmark_cases, load_benchmark_cases
from enzyme_software.manual_engine_eval.failure_taxonomy import assign_failure_tags
from enzyme_software.manual_engine_eval.heuristics import reactive_site_heuristic, route_heuristic, scaffold_heuristic
from enzyme_software.manual_engine_eval.pipeline_runner import run_case_through_pipeline
from enzyme_software.manual_engine_eval.report_builder import build_console_report, build_markdown_report
from enzyme_software.manual_engine_eval.validators import validate_pipeline_result


def test_demo_cases_exist():
    cases = demo_benchmark_cases()
    assert len(cases) >= 8
    assert cases[0].case_id


def test_loader_json_roundtrip(tmp_path):
    cases = demo_benchmark_cases()[:2]
    path = tmp_path / "bench.json"
    path.write_text(json.dumps([case.to_dict() for case in cases], indent=2))
    loaded = load_benchmark_cases(str(path))
    assert len(loaded) == 2
    assert loaded[0].case_id == cases[0].case_id


def test_heuristics_are_stable():
    case = demo_benchmark_cases()[0]
    module_minus1 = {
        "resolved_target": {
            "candidate_bonds": [
                {"atom_indices": [0, 1], "bond_class": "ch__benzylic", "bde_kj_mol": 375.5},
                {"atom_indices": [1, 2], "bond_class": "ch__aryl", "bde_kj_mol": 472.2},
            ]
        }
    }
    site = reactive_site_heuristic(case, module_minus1)
    route = route_heuristic(case, module_minus1)
    scaffold = scaffold_heuristic({"ranked_scaffolds": [{"scaffold_id": "1ABC", "module1_confidence": {"total": 0.7}}]})
    assert site["top_sites"]
    assert route["route_family"]
    assert scaffold["top_scaffold"] == "1ABC"


def test_failure_taxonomy_flags_exception():
    case = demo_benchmark_cases()[0]
    run = type("Run", (), {"ok": False, "exception": "boom"})()
    tags = assign_failure_tags(run, {}, {})
    assert "exception_failure" in tags


def test_report_builder_handles_partial():
    report = {
        "results": [
            {
                "case": {"case_id": "demo"},
                "status": "partial",
                "metrics": {
                    "heuristic_comparison": {"engine_route": "p450"},
                    "confidence": {"reported_confidence": 0.5},
                },
                "failure_tags": ["incomplete_output"],
                "validators": {"module0": {"ok": True}},
            }
        ],
        "aggregate": {
            "cases_run": 1,
            "status_counts": {"partial": 1},
            "module_success_rates": {"module0": 1.0},
            "failure_counts": {"incomplete_output": 1},
            "confidence_summary": {"high_conf_accuracy": None, "low_conf_accuracy": None},
        },
    }
    assert "MANUAL ENGINE EVALUATION" in build_console_report(report)
    assert "# Manual Engine Evaluation" in build_markdown_report(report)


def test_pipeline_runner_handles_real_case_gracefully():
    case = demo_benchmark_cases()[0]
    result = run_case_through_pipeline(case)
    assert result.ok or result.exception is not None
    validators = validate_pipeline_result(result)
    assert "module0" in validators
