"""
SIREN Neural Field for Cellular Concentrations
===============================================

Continuous implicit neural representation of metabolite concentrations.

Instead of: M[i] = discrete concentration of metabolite i
We use:     M(x,y,z,t) = continuous concentration field

Benefits:
- Spatial gradients computed exactly via autodiff
- Arbitrary resolution querying
- Natural handling of compartments (cytoplasm, membrane, external)
- Smooth interpolation between time points

Based on:
- nexus/field/siren_base.py (SIREN architecture)
- dark_manifold_cellular_field/src/qft/cellular_field.py (cellular adaptation)

Author: Naresh Chhillar, 2026
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math

# Try torch import - fallback to numpy-only mode if not available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# ============================================================================
# SIREN Layers (from siren_base.py)
# ============================================================================

if TORCH_AVAILABLE:
    
    SIREN_OMEGA_0 = 30.0  # Standard SIREN frequency

    class SirenLayer(nn.Module):
        """
        SIREN layer with sinusoidal activation.
        
        Uses sin(ω₀ * Wx + b) as activation, enabling the network
        to represent high-frequency signals and their derivatives.
        """
        
        def __init__(
            self,
            in_features: int,
            out_features: int,
            omega_0: float = SIREN_OMEGA_0,
            is_first: bool = False,
        ):
            super().__init__()
            
            self.in_features = in_features
            self.out_features = out_features
            self.omega_0 = omega_0
            self.is_first = is_first
            
            self.linear = nn.Linear(in_features, out_features)
            self._init_weights()
        
        def _init_weights(self):
            with torch.no_grad():
                if self.is_first:
                    bound = 1 / self.in_features
                else:
                    bound = math.sqrt(6 / self.in_features) / self.omega_0
                
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.uniform_(-bound, bound)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sin(self.omega_0 * self.linear(x))


    class SirenNetwork(nn.Module):
        """
        Full SIREN network for implicit neural representation.
        
        Learns a continuous function f: ℝ^d → ℝ^k.
        For cellular fields: f: (x, y, z, t) → [M₁, M₂, ..., Mₙ]
        """
        
        def __init__(
            self,
            in_features: int = 4,      # x, y, z, t
            hidden_features: int = 256,
            hidden_layers: int = 3,
            out_features: int = 158,   # n_metabolites for iMB155
            omega_0: float = SIREN_OMEGA_0,
        ):
            super().__init__()
            
            self.omega_0 = omega_0
            self.in_features = in_features
            self.out_features = out_features
            
            # Build network
            layers = []
            
            # First layer
            layers.append(SirenLayer(in_features, hidden_features, omega_0, is_first=True))
            
            # Hidden layers
            for _ in range(hidden_layers):
                layers.append(SirenLayer(hidden_features, hidden_features, omega_0))
            
            # Output layer (linear, no sin activation)
            layers.append(nn.Linear(hidden_features, out_features))
            
            self.layers = nn.Sequential(*layers)
        
        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            """
            Query metabolite concentrations at given spacetime coordinates.
            
            Args:
                coords: (..., 4) coordinates [x, y, z, t]
                
            Returns:
                (..., n_metabolites) concentration field values
            """
            return F.softplus(self.layers(coords))  # Concentrations must be positive


# ============================================================================
# Cellular Field with Compartments
# ============================================================================

@dataclass
class Compartment:
    """Definition of a cellular compartment."""
    name: str
    center: np.ndarray      # [3] - center position
    radius: float           # Bounding radius
    metabolite_indices: List[int]  # Which metabolites exist here
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside compartment."""
        return np.linalg.norm(point - self.center) < self.radius


@dataclass
class EnzymeSource:
    """Enzyme as a source/sink in the field."""
    position: np.ndarray    # [3] - location
    reaction_idx: int       # Which reaction it catalyzes
    activity: float         # kcat * concentration
    spread: float           # Gaussian width


class CellularSIRENField:
    """
    Continuous concentration field for the cell.
    
    The cell is modeled as a continuous 3D+time field where:
    - Each point (x,y,z,t) maps to concentration vector [M₁...Mₙ]
    - Enzymes act as localized sources/sinks
    - Compartments restrict metabolite localization
    - Diffusion is implicit in the field smoothness
    
    Key features from cellular_field.py:
    - SIREN backbone for smooth interpolation
    - Enzyme sources create/consume metabolites locally
    - Compartment constraints bound metabolites
    """
    
    def __init__(
        self,
        n_metabolites: int = 158,
        cell_radius: float = 1.0,  # Normalized cell size
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        compartments: Optional[List[Compartment]] = None,
        device: str = "cpu",
    ):
        """
        Initialize cellular SIREN field.
        
        Args:
            n_metabolites: Number of metabolite species
            cell_radius: Cell radius (for normalization)
            hidden_dim: SIREN hidden dimension
            hidden_layers: Number of SIREN hidden layers
            compartments: List of compartments (default: cytoplasm only)
            device: Compute device
        """
        self.n_metabolites = n_metabolites
        self.cell_radius = cell_radius
        self.device = device
        
        # Default compartments
        if compartments is None:
            compartments = [
                Compartment(
                    name="cytoplasm",
                    center=np.array([0.0, 0.0, 0.0]),
                    radius=cell_radius,
                    metabolite_indices=list(range(n_metabolites))
                )
            ]
        self.compartments = compartments
        
        # Create SIREN network if torch available
        if TORCH_AVAILABLE:
            self.siren = SirenNetwork(
                in_features=4,  # x, y, z, t
                hidden_features=hidden_dim,
                hidden_layers=hidden_layers,
                out_features=n_metabolites,
            ).to(device)
        else:
            self.siren = None
            
        # Enzyme sources
        self.enzyme_sources: List[EnzymeSource] = []
        
        # Reference concentrations (for initialization)
        self.reference_concentrations = np.ones(n_metabolites)
        
        # Time state
        self.current_time = 0.0
    
    def add_enzyme(
        self,
        position: np.ndarray,
        reaction_idx: int,
        activity: float,
        spread: float = 0.1,
    ) -> None:
        """Add enzyme source to the field."""
        self.enzyme_sources.append(EnzymeSource(
            position=np.array(position),
            reaction_idx=reaction_idx,
            activity=activity,
            spread=spread,
        ))
    
    def set_reference_concentrations(self, concentrations: np.ndarray) -> None:
        """Set reference concentrations for field initialization."""
        self.reference_concentrations = np.array(concentrations)
    
    def query(
        self,
        points: np.ndarray,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Query concentrations at given points.
        
        Args:
            points: [N, 3] spatial coordinates
            time: Query time (default: current_time)
            
        Returns:
            [N, n_metabolites] concentration values
        """
        if time is None:
            time = self.current_time
        
        N = points.shape[0]
        
        if TORCH_AVAILABLE and self.siren is not None:
            # Use SIREN network
            coords = np.zeros((N, 4))
            coords[:, :3] = points / self.cell_radius  # Normalize
            coords[:, 3] = time
            
            with torch.no_grad():
                coords_t = torch.tensor(coords, dtype=torch.float32, device=self.device)
                conc_t = self.siren(coords_t)
                concentrations = conc_t.cpu().numpy()
        else:
            # Fallback: Gaussian mixture from enzyme sources
            concentrations = np.tile(self.reference_concentrations, (N, 1))
            
            # Modulate by distance to enzyme sources
            for enzyme in self.enzyme_sources:
                dist = np.linalg.norm(points - enzyme.position, axis=1)
                influence = np.exp(-0.5 * (dist / enzyme.spread) ** 2)
                # Enzyme affects its reaction's metabolites
                concentrations[:, enzyme.reaction_idx % self.n_metabolites] *= (
                    1 + enzyme.activity * influence[:, np.newaxis]
                ).flatten()
        
        # Apply compartment constraints
        for comp in self.compartments:
            for i, point in enumerate(points):
                if not comp.contains(point):
                    # Zero out metabolites not in this compartment
                    for m_idx in range(self.n_metabolites):
                        if m_idx not in comp.metabolite_indices:
                            concentrations[i, m_idx] = 0.0
        
        return np.maximum(concentrations, 0)  # Non-negative
    
    def query_single(self, x: float, y: float, z: float, t: Optional[float] = None) -> np.ndarray:
        """Query concentration at a single point."""
        return self.query(np.array([[x, y, z]]), t)[0]
    
    def gradient(
        self,
        points: np.ndarray,
        metabolite_idx: int,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute spatial gradient of concentration.
        
        Uses autodiff if torch available, finite differences otherwise.
        
        Args:
            points: [N, 3] spatial coordinates
            metabolite_idx: Which metabolite
            time: Query time
            
        Returns:
            [N, 3] gradient vectors
        """
        if time is None:
            time = self.current_time
        
        N = points.shape[0]
        
        if TORCH_AVAILABLE and self.siren is not None:
            # Use autodiff
            coords = np.zeros((N, 4))
            coords[:, :3] = points / self.cell_radius
            coords[:, 3] = time
            
            coords_t = torch.tensor(coords, dtype=torch.float32, device=self.device, requires_grad=True)
            conc_t = self.siren(coords_t)[:, metabolite_idx]
            
            grad = torch.autograd.grad(
                conc_t.sum(),
                coords_t,
                create_graph=False,
            )[0]
            
            return grad[:, :3].detach().cpu().numpy() / self.cell_radius
        else:
            # Finite differences
            eps = 0.01 * self.cell_radius
            gradients = np.zeros((N, 3))
            
            for d in range(3):
                points_plus = points.copy()
                points_plus[:, d] += eps
                points_minus = points.copy()
                points_minus[:, d] -= eps
                
                c_plus = self.query(points_plus, time)[:, metabolite_idx]
                c_minus = self.query(points_minus, time)[:, metabolite_idx]
                
                gradients[:, d] = (c_plus - c_minus) / (2 * eps)
            
            return gradients
    
    def laplacian(
        self,
        points: np.ndarray,
        metabolite_idx: int,
        time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Compute Laplacian (for diffusion).
        
        Returns:
            [N] Laplacian values
        """
        if time is None:
            time = self.current_time
        
        # Finite differences for Laplacian
        eps = 0.01 * self.cell_radius
        c_center = self.query(points, time)[:, metabolite_idx]
        
        laplacian = np.zeros(len(points))
        for d in range(3):
            points_plus = points.copy()
            points_plus[:, d] += eps
            points_minus = points.copy()
            points_minus[:, d] -= eps
            
            c_plus = self.query(points_plus, time)[:, metabolite_idx]
            c_minus = self.query(points_minus, time)[:, metabolite_idx]
            
            laplacian += (c_plus + c_minus - 2 * c_center) / (eps ** 2)
        
        return laplacian
    
    def update_time(self, dt: float) -> None:
        """Advance current time."""
        self.current_time += dt
    
    def fit_to_concentrations(
        self,
        concentrations: np.ndarray,
        positions: Optional[np.ndarray] = None,
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """
        Fit SIREN to match given concentration data.
        
        Args:
            concentrations: [N, n_metabolites] target concentrations
            positions: [N, 3] positions (default: random in cell)
            n_epochs: Training epochs
            lr: Learning rate
            
        Returns:
            Final loss value
        """
        if not TORCH_AVAILABLE or self.siren is None:
            self.reference_concentrations = concentrations.mean(axis=0)
            return 0.0
        
        N = len(concentrations)
        
        if positions is None:
            positions = np.random.randn(N, 3) * self.cell_radius * 0.5
        
        # Prepare data
        coords = np.zeros((N, 4))
        coords[:, :3] = positions / self.cell_radius
        coords[:, 3] = self.current_time
        
        coords_t = torch.tensor(coords, dtype=torch.float32, device=self.device)
        target_t = torch.tensor(concentrations, dtype=torch.float32, device=self.device)
        
        # Train
        optimizer = torch.optim.Adam(self.siren.parameters(), lr=lr)
        
        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred = self.siren(coords_t)
            loss = F.mse_loss(pred, target_t)
            loss.backward()
            optimizer.step()
        
        return float(loss.item())
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        return {
            "n_metabolites": self.n_metabolites,
            "cell_radius": self.cell_radius,
            "n_compartments": len(self.compartments),
            "n_enzymes": len(self.enzyme_sources),
            "current_time": self.current_time,
            "torch_available": TORCH_AVAILABLE,
            "siren_params": sum(p.numel() for p in self.siren.parameters()) if self.siren else 0,
        }


def create_syn3a_field(
    n_metabolites: int = 158,
    cell_radius: float = 0.3,  # ~300nm for syn3A
) -> CellularSIRENField:
    """
    Create a field configured for JCVI-syn3A minimal cell.
    
    syn3A is roughly spherical with ~300nm radius.
    Only has cytoplasm (no internal compartments).
    """
    compartments = [
        Compartment(
            name="cytoplasm",
            center=np.array([0.0, 0.0, 0.0]),
            radius=cell_radius,
            metabolite_indices=list(range(n_metabolites))
        ),
        Compartment(
            name="membrane",
            center=np.array([0.0, 0.0, 0.0]),
            radius=cell_radius * 1.05,  # Thin shell
            metabolite_indices=[]  # Lipids, not metabolites
        ),
    ]
    
    return CellularSIRENField(
        n_metabolites=n_metabolites,
        cell_radius=cell_radius,
        hidden_dim=128,  # Smaller for minimal cell
        hidden_layers=2,
        compartments=compartments,
    )
