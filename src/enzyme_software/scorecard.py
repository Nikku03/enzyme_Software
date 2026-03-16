from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ScoreCardMetric:
    name: str
    raw: Optional[float]
    calibrated: Optional[float]
    ci90: Optional[Tuple[float, float]]
    n_eff: Optional[float]
    status: str
    definition: str
    contributors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "raw": self.raw,
            "calibrated": self.calibrated,
            "ci90": list(self.ci90) if self.ci90 else None,
            "n_eff": self.n_eff,
            "status": self.status,
            "definition": self.definition,
            "contributors": self.contributors,
        }


@dataclass
class ScoreCard:
    module_id: int
    metrics: List[ScoreCardMetric] = field(default_factory=list)
    calibration_status: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "calibration_status": self.calibration_status,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


def metric_status(raw: Optional[float], n_eff: Optional[float]) -> str:
    if raw is None:
        return "missing"
    if n_eff is None:
        return "ok"
    try:
        if float(n_eff) < 5.0:
            return "low_support"
    except (TypeError, ValueError):
        return "ok"
    return "ok"


def contributors_from_features(
    feature_values: Dict[str, Any],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    items: List[Tuple[str, float]] = []
    for key, value in (feature_values or {}).items():
        if isinstance(value, (int, float)):
            items.append((key, float(value)))
    items.sort(key=lambda item: abs(item[1]), reverse=True)
    contributors = [
        {"feature": key, "delta": round(value, 3)} for key, value in items[:limit]
    ]
    return contributors


def calibration_status_from_signals(
    calibration_source: Optional[str],
    data_support: Optional[float],
    evidence_strength: Optional[float],
) -> str:
    if calibration_source:
        return "calibrated"
    support = float(data_support or 0.0)
    evidence = float(evidence_strength or 0.0)
    if support > 0.0 or evidence > 0.0:
        return "weakly_calibrated"
    return "heuristic"
