from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


TargetBondType = Union[str, Dict[str, Any]]
ReactiveSiteType = Union[str, int]


@dataclass
class BenchmarkTolerances:
    topk: int = 3
    allow_loose_route_match: bool = True
    allow_partial_tags: bool = True


@dataclass
class BenchmarkCase:
    case_id: str
    smiles: str
    target_bond: TargetBondType
    expected_reactive_sites: Optional[List[ReactiveSiteType]] = None
    expected_route_family: Optional[str] = None
    expected_mechanism_tags: List[str] = field(default_factory=list)
    expected_scaffold_tags: List[str] = field(default_factory=list)
    expected_variant_tags: List[str] = field(default_factory=list)
    expected_experiment_tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tolerances: BenchmarkTolerances = field(default_factory=BenchmarkTolerances)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["target_bond"] = self.target_bond
        return payload


@dataclass
class PipelineRunResult:
    case: BenchmarkCase
    ok: bool
    exception: Optional[str]
    raw_outputs: Dict[str, Any]
    module_summaries: Dict[str, Dict[str, Any]]
    confidence_state: Dict[str, Any]
    energy_ledger_summary: Dict[str, Any]
    arbitration_summary: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    case: BenchmarkCase
    pipeline: PipelineRunResult
    validators: Dict[str, Dict[str, Any]]
    heuristics: Dict[str, Any]
    metrics: Dict[str, Any]
    failure_tags: List[str]
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "pipeline": {
                "ok": self.pipeline.ok,
                "exception": self.pipeline.exception,
                "module_summaries": self.pipeline.module_summaries,
                "confidence_state": self.pipeline.confidence_state,
                "energy_ledger_summary": self.pipeline.energy_ledger_summary,
                "arbitration_summary": self.pipeline.arbitration_summary,
                "warnings": self.pipeline.warnings,
                "errors": self.pipeline.errors,
            },
            "validators": self.validators,
            "heuristics": self.heuristics,
            "metrics": self.metrics,
            "failure_tags": self.failure_tags,
            "status": self.status,
        }
