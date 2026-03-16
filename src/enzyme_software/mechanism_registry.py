from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MechanismSpec:
    route_id: str
    reaction_family: str
    expected_nucleophile: str
    allowed_nucleophile_geometries: List[str] = field(default_factory=list)
    mechanism_notes: str = ""
    mismatch_policy_default: str = "KEEP_WITH_PENALTY"

    def to_dict(self) -> Dict[str, object]:
        return {
            "route_id": self.route_id,
            "reaction_family": self.reaction_family,
            "expected_nucleophile": self.expected_nucleophile,
            "allowed_nucleophile_geometries": list(self.allowed_nucleophile_geometries),
            "mechanism_notes": self.mechanism_notes,
            "mismatch_policy_default": self.mismatch_policy_default,
        }


_MECHANISM_TABLE: Dict[str, MechanismSpec] = {
    "serine_hydrolase": MechanismSpec(
        route_id="serine_hydrolase",
        reaction_family="hydrolysis",
        expected_nucleophile="Ser",
        allowed_nucleophile_geometries=["serine_og"],
        mechanism_notes="Ser-His-Asp triad expected; acyl-enzyme mechanism.",
        mismatch_policy_default="KEEP_WITH_PENALTY",
    ),
    "cysteine_hydrolase": MechanismSpec(
        route_id="cysteine_hydrolase",
        reaction_family="hydrolysis",
        expected_nucleophile="Cys",
        allowed_nucleophile_geometries=["cysteine_thiol"],
        mechanism_notes="Cys-His-Asp/Glu catalytic triad-like motif.",
        mismatch_policy_default="KEEP_WITH_PENALTY",
    ),
    "metallo_esterase": MechanismSpec(
        route_id="metallo_esterase",
        reaction_family="hydrolysis",
        expected_nucleophile="MetalWater",
        allowed_nucleophile_geometries=["metal_water"],
        mechanism_notes="Metal-activated water nucleophile; metal-binding motif required.",
        mismatch_policy_default="FORK_HYPOTHESES",
    ),
}


def resolve_mechanism(route_id: str) -> MechanismSpec:
    """Resolve a mechanism spec for the given route id (case-insensitive)."""
    key = (route_id or "").strip().lower()
    if key in _MECHANISM_TABLE:
        return _MECHANISM_TABLE[key]
    return MechanismSpec(
        route_id=route_id or "unknown",
        reaction_family="unknown",
        expected_nucleophile="Either",
        allowed_nucleophile_geometries=[],
        mechanism_notes="Fallback mechanism; insufficient route metadata.",
        mismatch_policy_default="KEEP_WITH_PENALTY",
    )
