"""Evaluation harness for the manual mechanistic metabolism engine."""

from .benchmark_loader import load_benchmark_cases, demo_benchmark_cases
from .benchmark_schema import BenchmarkCase, EvaluationResult, PipelineRunResult
from .evaluators import evaluate_case, evaluate_cases
from .failure_taxonomy import assign_failure_tags
from .pipeline_runner import run_case_through_pipeline
from .report_builder import build_console_report, build_markdown_report
from .validators import validate_pipeline_result

__all__ = [
    "BenchmarkCase",
    "EvaluationResult",
    "PipelineRunResult",
    "assign_failure_tags",
    "build_console_report",
    "build_markdown_report",
    "demo_benchmark_cases",
    "evaluate_case",
    "evaluate_cases",
    "load_benchmark_cases",
    "run_case_through_pipeline",
    "validate_pipeline_result",
]
