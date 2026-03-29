"""
Wave-engine fork point for transport.

The current transport contract is re-exported so the trainer can route through
`nexus.reasoning_wave` immediately. Wave-specific overlap costs and transport
logic should be implemented here later without changing the trainer API.
"""

from nexus.reasoning.pgw_transport import PGWTransportResult, PGWTransporter

__all__ = ["PGWTransportResult", "PGWTransporter"]
