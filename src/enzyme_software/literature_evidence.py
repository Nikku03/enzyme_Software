from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvidenceRecord:
    """Canonical literature-anchored evidence record for priors.

    Fields are intentionally minimal and auditable. All provenance is stored
    in the provenance dict to preserve traceability.
    """

    source: str
    source_id: str
    substrate_smiles: str
    reaction_family: str
    conditions: Dict[str, Any]
    catalyst_family: Optional[str]
    outcome_label: bool
    notes: Optional[str] = None
    confidence: float = 0.5
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "substrate_smiles": self.substrate_smiles,
            "reaction_family": self.reaction_family,
            "conditions": self.conditions,
            "catalyst_family": self.catalyst_family,
            "outcome_label": self.outcome_label,
            "notes": self.notes,
            "confidence": self.confidence,
            "provenance": self.provenance,
        }
