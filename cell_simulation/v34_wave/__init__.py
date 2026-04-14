"""
Dark Manifold V34: Wave-Based Whole-Cell Simulation
====================================================

Integrates wave mechanics from CYP-Predict into cell simulation:
- Graph Laplacian eigendecomposition for modal dynamics
- SIREN neural fields for continuous concentration
- Green's function for gene regulatory propagation
- Tensor representations for enzyme fields

Target: 2000-5000x speedup over ODE-based V33
"""

from .modal_dynamics import ModalDynamicsEngine
from .siren_field import CellularSIRENField
from .greens_propagator import GeneRegulatoryPropagator
from .wave_cell import WaveCellSimulator

__version__ = "34.0.0"
__all__ = [
    "ModalDynamicsEngine",
    "CellularSIRENField", 
    "GeneRegulatoryPropagator",
    "WaveCellSimulator",
]
