"""
Wave-engine fork point for analogical fusion.

This module currently mirrors the classic analogical-fusion contracts by
re-exporting the existing implementation. It exists so we can incrementally
replace the internals with equivariant or wavefield-specific logic without
modifying the stable classic path.
"""

from nexus.reasoning.analogical_fusion import (
    DEFAULT_CYP3A4_MORPHISM_ALPHA,
    HomoscedasticArbiterLoss,
    MorphismFocalLoss,
    NexusDualDecoder,
    PGWCrossAttention,
)

__all__ = [
    "DEFAULT_CYP3A4_MORPHISM_ALPHA",
    "HomoscedasticArbiterLoss",
    "MorphismFocalLoss",
    "NexusDualDecoder",
    "PGWCrossAttention",
]
