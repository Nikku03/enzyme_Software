from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Threshold:
    value: float
    rationale: str


RETENTION_WEAK_THRESHOLD = Threshold(
    value=0.45,
    rationale="Retention below this implies weak binding/orientation risk.",
)
ROUTE_CONFIDENCE_LOW_THRESHOLD = Threshold(
    value=0.6,
    rationale="Route confidence below this indicates weak mechanistic support.",
)
TARGET_RESOLUTION_LOW_THRESHOLD = Threshold(
    value=0.85,
    rationale="Target resolution below this indicates ambiguous bond selection.",
)
