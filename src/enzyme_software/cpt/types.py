from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class CPTResult:
    cpt_id: str
    mechanism_id: str
    passed: bool
    score: float
    confidence: float
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    atoms_involved: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class MechanismProfile:
    mechanism_id: str
    feasibility_score: float
    confidence: float
    consistency: str
    primary_constraint: Optional[str]
    key_insight: str
    evidence: List[CPTResult]
