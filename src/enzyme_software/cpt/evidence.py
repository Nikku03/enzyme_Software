from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from .types import CPTResult


@dataclass
class EvidenceGraph:
    """Simple evidence store (upgradeable to real graph later)."""
    by_mechanism: Dict[str, List[CPTResult]] = field(default_factory=dict)

    def add(self, r: CPTResult) -> None:
        self.by_mechanism.setdefault(r.mechanism_id, []).append(r)

    def get(self, mech: str) -> List[CPTResult]:
        return list(self.by_mechanism.get(mech, []))
