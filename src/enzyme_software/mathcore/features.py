from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FeatureVector:
    values: Dict[str, float]
    missing: List[str] = field(default_factory=list)
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "missing": self.missing,
            "source": self.source,
        }


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def extract_features(context: Dict[str, Any]) -> FeatureVector:
    bond_context = context.get("bond_context") or {}
    route = context.get("route") or {}
    reaction_intent = context.get("reaction_intent") or {}
    descriptor_status = context.get("descriptor_status") or {}
    warnings = context.get("warnings") or []

    features: Dict[str, float] = {}
    missing: List[str] = []

    def add_feature(name: str, value: Any) -> None:
        if value is None:
            missing.append(name)
            features[name] = 0.0
            return
        try:
            features[name] = float(value)
        except (TypeError, ValueError):
            missing.append(name)
            features[name] = 0.0

    add_feature("bond_role_confidence", bond_context.get("primary_role_confidence"))
    add_feature("target_resolution_confidence", context.get("target_resolution_confidence"))
    add_feature("intent_confidence", reaction_intent.get("intent_confidence"))
    add_feature("condition_score", context.get("condition_score"))

    features["has_primary_role"] = _bool(
        bond_context.get("primary_role") or bond_context.get("bond_role")
    )
    features["route_primary_present"] = _bool(route.get("primary"))
    features["descriptor_complete"] = _bool(descriptor_status.get("complete"))
    features["job_type_reagent"] = _bool(context.get("job_type") == "REAGENT_GENERATION")

    warning_count = len(warnings)
    features["warning_count"] = min(1.0, warning_count / 5.0)
    mechanism_count = len(route.get("mechanisms") or [])
    features["mechanism_count"] = min(1.0, mechanism_count / 4.0)

    match_count = context.get("match_count")
    if match_count is None:
        missing.append("match_count")
        features["ambiguity_flag"] = 0.0
    else:
        features["ambiguity_flag"] = 0.0 if match_count == 1 else 1.0

    novelty = context.get("novelty_penalty")
    if novelty is None:
        missing.append("novelty_penalty")
        novelty = 0.0
    features["novelty_penalty"] = float(novelty)

    data_support = context.get("data_support")
    if data_support is None:
        missing.append("data_support")
        data_support = 0.5
    features["data_support"] = float(data_support)

    return FeatureVector(values=features, missing=sorted(set(missing)), source=context.get("source"))
