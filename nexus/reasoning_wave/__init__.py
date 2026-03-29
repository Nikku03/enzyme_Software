"""
Parallel analogical-engine package used to stage a wave/equivariant fork
without destabilizing the classic reasoning stack.

The initial implementation mirrors the classic contracts by re-exporting the
current classes. Future wave-specific logic should live in the sibling modules
in this package so the trainer can switch engines via a single selector.
"""

from .analogical_fusion import (
    HomoscedasticArbiterLoss,
    NexusDualDecoder,
    PGWCrossAttention,
)
from .hyperbolic_memory import HyperbolicMemoryBank
from .metric_learner import (
    HGNNProjection,
    MechanismEncoder,
    PoincareMath,
    WaveQuantumDistillationHead,
    _som_class,
    encoder_supervision_loss,
    hyperbolic_supervision_loss,
    mechanism_contrastive_loss,
    quantum_distillation_loss,
)
from .pgw_transport import PGWTransportResult, PGWTransporter

__all__ = [
    "HGNNProjection",
    "HomoscedasticArbiterLoss",
    "HyperbolicMemoryBank",
    "MechanismEncoder",
    "NexusDualDecoder",
    "PGWCrossAttention",
    "PGWTransportResult",
    "PGWTransporter",
    "PoincareMath",
    "WaveQuantumDistillationHead",
    "_som_class",
    "encoder_supervision_loss",
    "hyperbolic_supervision_loss",
    "mechanism_contrastive_loss",
    "quantum_distillation_loss",
]
