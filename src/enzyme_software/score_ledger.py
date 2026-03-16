from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScoreTerm:
    name: str
    value: Optional[float]
    unit: str
    formula: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "formula": self.formula,
            "inputs": self.inputs,
            "notes": self.notes,
        }


@dataclass
class ScoreLedger:
    module_id: int
    terms: List[ScoreTerm] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_id": self.module_id,
            "terms": [term.to_dict() for term in self.terms],
        }
