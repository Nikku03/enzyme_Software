"""
Wave-engine fork point for analogical memory.

The initial version re-exports the current hyperbolic memory-bank contract so
the trainer can be switched to a parallel package immediately. Future work can
replace retrieval features, bank payloads, and scoring with wave/equivariant
representations while preserving the external API.
"""

from nexus.reasoning.hyperbolic_memory import HyperbolicMemoryBank

__all__ = ["HyperbolicMemoryBank"]
