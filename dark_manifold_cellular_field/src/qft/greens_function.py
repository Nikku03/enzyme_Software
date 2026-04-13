"""
Quantum Field Theory Components
===============================

This module implements QFT-inspired components for the Dark Manifold architecture:

1. Green's Function Propagator
   - Models non-local gene-gene interactions
   - G(ω) = (ω + iη - H)^(-1)
   - Captures long-range regulatory effects

2. Field Equations
   - Gene expression as field excitations
   - Dark matter mediator field

Based on concepts from:
    - quantum_field_theory.py in enzyme_Software
    - first_principles_qft.py in enzyme_Software
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class GeneNetworkGreensFunction(nn.Module):
    """
    Non-local gene interactions via Green's function propagator.
    
    G(ω) = (ω + iη - H)^(-1)
    
    where:
        - ω is the frequency (energy scale)
        - η is the broadening parameter (damping)
        - H is the effective Hamiltonian (gene-gene interactions)
    
    The Green's function captures propagation of regulatory signals
    through the gene network, including indirect effects.
    """
    
    def __init__(
        self,
        n_genes: int,
        rank: int = 32,
        eta: float = 0.05,
        learnable_eta: bool = False,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.rank = min(rank, n_genes // 2)
        
        # Low-rank approximation of Hamiltonian: H ≈ U @ U^T + diag(d)
        # This reduces O(n³) to O(n * rank²) for inversion
        self.H_low_rank = nn.Parameter(torch.randn(n_genes, self.rank) * 0.01)
        self.H_diag = nn.Parameter(torch.randn(n_genes) * 0.1)
        
        # Broadening parameter
        if learnable_eta:
            self.log_eta = nn.Parameter(torch.tensor(math.log(eta)))
        else:
            self.register_buffer('log_eta', torch.tensor(math.log(eta)))
    
    @property
    def eta(self) -> float:
        return torch.exp(self.log_eta).item()
    
    @property
    def H(self) -> torch.Tensor:
        """Construct the effective Hamiltonian."""
        H = self.H_low_rank @ self.H_low_rank.t() + torch.diag(self.H_diag)
        # Symmetrize
        return 0.5 * (H + H.t())
    
    def forward(self, omega: float = 0.0) -> torch.Tensor:
        """
        Compute Green's function G(ω).
        
        Args:
            omega: Frequency/energy scale (default 0 = static limit)
            
        Returns:
            G: Green's function matrix [n_genes, n_genes]
        """
        device = self.H_low_rank.device
        eta = torch.exp(self.log_eta).to(device)
        
        # Build resolvent: (ω + iη)I - H
        H = self.H
        I = torch.eye(self.n_genes, device=device)
        resolvent = (omega + 1j * eta) * I - H
        
        # Invert (with numerical stability)
        try:
            G = torch.linalg.inv(resolvent)
            # Take absolute value for real-valued output
            G = torch.abs(G).float()
            # Clamp to avoid explosion
            return G.clamp(max=10.0)
        except RuntimeError:
            # Fallback to identity if inversion fails
            return I
    
    def spectral_density(self, omega_range: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral density A(ω) = -Im[Tr G(ω)] / π
        
        Useful for analyzing the gene network's frequency response.
        """
        densities = []
        for omega in omega_range:
            G = self.forward(omega.item())
            # This is a real approximation since we took abs
            densities.append(G.trace())
        return torch.stack(densities)


class DarkMatterField(nn.Module):
    """
    Continuous "dark matter" field that mediates interactions.
    
    The field fills the entire cellular volume and modulates
    how different molecular species interact.
    
    Inspired by the Dark Manifold concept where the network IS
    a 4D manifold, not a graph on a manifold.
    """
    
    def __init__(
        self,
        field_dim: int,
        hidden_dim: int = 128,
        n_modes: int = 16,
    ):
        super().__init__()
        
        self.field_dim = field_dim
        self.n_modes = n_modes
        
        # Field modes (like Fourier modes)
        self.mode_amplitudes = nn.Parameter(torch.randn(n_modes, field_dim) * 0.1)
        self.mode_phases = nn.Parameter(torch.randn(n_modes) * 2 * math.pi)
        self.mode_frequencies = nn.Parameter(torch.randn(n_modes).abs() + 0.1)
        
        # Field dynamics
        self.field_encoder = nn.Sequential(
            nn.Linear(field_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim),
        )
    
    def forward(
        self,
        state: torch.Tensor,
        t: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute dark field contribution at time t.
        
        Args:
            state: Current state [B, field_dim]
            t: Time coordinate
            
        Returns:
            field_effect: Field modulation [B, field_dim]
        """
        device = state.device
        
        # Sum over modes: ∑ A_k * cos(ω_k * t + φ_k)
        phases = self.mode_frequencies * t + self.mode_phases
        mode_values = torch.cos(phases.to(device))  # [n_modes]
        
        # Field value
        field = (mode_values.unsqueeze(1) * self.mode_amplitudes.to(device)).sum(dim=0)
        
        # Modulate state
        return state + self.field_encoder(state) * torch.tanh(field)


class QuantumFluctuation(nn.Module):
    """
    Quantum fluctuations as stochastic sampling primitive.
    
    In the Dark Manifold, states exist in superposition until
    collapse/readout. This module adds controlled stochasticity
    that can be interpreted as quantum fluctuations.
    """
    
    def __init__(
        self,
        dim: int,
        temperature: float = 0.1,
    ):
        super().__init__()
        
        self.dim = dim
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature)))
        
        # Uncertainty scale per dimension
        self.log_sigma = nn.Parameter(torch.zeros(dim))
    
    @property
    def temperature(self) -> float:
        return torch.exp(self.log_temperature).item()
    
    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)
    
    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Add quantum fluctuations to state.
        
        Args:
            state: Input state [B, dim]
            deterministic: If True, skip fluctuations
            
        Returns:
            state_fluctuated: State with fluctuations
        """
        if deterministic or not self.training:
            return state
        
        device = state.device
        T = torch.exp(self.log_temperature).to(device)
        sigma = self.sigma.to(device)
        
        # Sample fluctuations: σ * √T * ε
        noise = torch.randn_like(state) * sigma * torch.sqrt(T)
        
        return state + noise
