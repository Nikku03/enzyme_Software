"""
Unified Cognitive Metabolism Engine (UCME)
==========================================

A unified architecture that combines:
1. WAVE FIELD - Electron density dynamics during reaction
2. LIQUID CORE - Adaptive continuous-time GNN
3. ANALOGICAL REASONER - Human-like reasoning over similar molecules

The three modules work together:
- Wave Field provides electronic reactivity signals
- Liquid Core learns adaptive representations with variable "thinking time"
- Analogical Reasoner retrieves similar cases and reasons about differences

All integrated through learned gating for final SoM prediction.

Author: CYP-Predict Team
Version: 1.0
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdFMCS, Descriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UCMEConfig:
    """Configuration for UCME model."""
    # Dimensions
    hidden_dim: int = 128
    atom_feature_dim: int = 64
    edge_feature_dim: int = 16
    
    # Wave Field
    wave_hidden_dim: int = 64
    wave_num_shells: int = 4
    wave_ode_steps: int = 10
    
    # Liquid Core
    liquid_num_layers: int = 4
    liquid_ode_steps: int = 8
    liquid_tau_min: float = 0.1
    liquid_tau_max: float = 2.0
    liquid_adaptive_steps: bool = True
    liquid_max_steps: int = 20
    liquid_convergence_threshold: float = 0.01
    
    # Analogical Reasoner
    memory_capacity: int = 4096
    memory_topk: int = 16
    hyperbolic_dim: int = 64
    hyperbolic_curvature: float = 1.0
    use_mcs_alignment: bool = True
    max_mcs_time: float = 1.0  # seconds
    
    # Integration
    dropout: float = 0.1
    num_heads: int = 4
    
    # Training
    use_wave: bool = True
    use_liquid: bool = True
    use_analogical: bool = True


# =============================================================================
# WAVE FIELD MODULE - Electron Dynamics
# =============================================================================

if TORCH_AVAILABLE:
    
    class SIRENLayer(nn.Module):
        """SIREN layer with sinusoidal activation."""
        
        def __init__(self, in_dim: int, out_dim: int, omega_0: float = 30.0, is_first: bool = False):
            super().__init__()
            self.omega_0 = omega_0
            self.linear = nn.Linear(in_dim, out_dim)
            self._init_weights(is_first)
            
        def _init_weights(self, is_first: bool):
            with torch.no_grad():
                if is_first:
                    self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                                  1 / self.linear.in_features)
                else:
                    self.linear.weight.uniform_(
                        -math.sqrt(6 / self.linear.in_features) / self.omega_0,
                        math.sqrt(6 / self.linear.in_features) / self.omega_0
                    )
                    
        def forward(self, x: Tensor) -> Tensor:
            return torch.sin(self.omega_0 * self.linear(x))
    
    
    class ElectronDensityField(nn.Module):
        """
        Neural network representing electron density ρ(r) in 3D space.
        Uses SIREN for continuous, differentiable density field.
        """
        
        def __init__(self, hidden_dim: int = 64, num_layers: int = 4):
            super().__init__()
            self.layers = nn.ModuleList()
            # Input: 3D position + atom context
            self.layers.append(SIRENLayer(3 + hidden_dim, hidden_dim, is_first=True))
            for _ in range(num_layers - 2):
                self.layers.append(SIRENLayer(hidden_dim, hidden_dim))
            # Output: density value
            self.output = nn.Linear(hidden_dim, 1)
            
        def forward(self, positions: Tensor, atom_context: Tensor) -> Tensor:
            """
            Compute electron density at given positions.
            
            Args:
                positions: [N, 3] - 3D coordinates to sample
                atom_context: [N, hidden_dim] - context from nearest atom
                
            Returns:
                density: [N] - electron density values
            """
            x = torch.cat([positions, atom_context], dim=-1)
            for layer in self.layers:
                x = layer(x)
            return self.output(x).squeeze(-1)
    
    
    class PerturbationField(nn.Module):
        """
        Models the perturbation from CYP Fe=O attacking a site.
        V(r, t) = perturbation potential at position r and time t
        """
        
        def __init__(self, hidden_dim: int = 32):
            super().__init__()
            # Input: distance to attack site, time, atom features
            self.net = nn.Sequential(
                nn.Linear(2 + hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            
        def forward(self, distance_to_attack: Tensor, time: Tensor, 
                    atom_features: Tensor) -> Tensor:
            """
            Compute perturbation potential.
            
            Args:
                distance_to_attack: [N] - distance from each position to attack site
                time: scalar - current time in reaction
                atom_features: [N, hidden_dim] - local atom features
                
            Returns:
                potential: [N] - perturbation potential
            """
            time_expanded = time.expand(distance_to_attack.shape[0])
            x = torch.cat([
                distance_to_attack.unsqueeze(-1),
                time_expanded.unsqueeze(-1),
                atom_features
            ], dim=-1)
            return self.net(x).squeeze(-1)
    
    
    class WaveDynamics(nn.Module):
        """
        Neural ODE for electron density evolution.
        dρ/dt = F(ρ, V, t; θ)
        
        Models how electron density changes during H-abstraction reaction.
        """
        
        def __init__(self, hidden_dim: int = 64):
            super().__init__()
            # Dynamics network: predicts dρ/dt given current density and perturbation
            self.dynamics_net = nn.Sequential(
                nn.Linear(hidden_dim + 2, hidden_dim),  # density_features + perturbation + time
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
        def forward(self, density_features: Tensor, perturbation: Tensor, 
                    time: Tensor) -> Tensor:
            """
            Compute time derivative of density features.
            
            Args:
                density_features: [N, hidden_dim] - current density state
                perturbation: [N] - perturbation potential at each point
                time: scalar - current time
                
            Returns:
                d_density/dt: [N, hidden_dim] - rate of change
            """
            time_expanded = time.expand(density_features.shape[0], 1)
            x = torch.cat([
                density_features,
                perturbation.unsqueeze(-1),
                time_expanded
            ], dim=-1)
            return self.dynamics_net(x)
    
    
    class WaveFieldModule(nn.Module):
        """
        Complete Wave Field module for electron dynamics.
        
        Workflow:
        1. Compute ground state electron density
        2. For each candidate site, simulate Fe=O attack
        3. Track how electron density flows/reorganizes
        4. Score sites by ease of electron reorganization
        """
        
        def __init__(self, config: UCMEConfig):
            super().__init__()
            self.config = config
            
            # Ground state density network
            self.density_field = ElectronDensityField(
                hidden_dim=config.wave_hidden_dim,
                num_layers=4
            )
            
            # Perturbation model
            self.perturbation = PerturbationField(hidden_dim=config.wave_hidden_dim)
            
            # Dynamics
            self.dynamics = WaveDynamics(hidden_dim=config.wave_hidden_dim)
            
            # Atom feature encoder for wave field
            self.atom_encoder = nn.Sequential(
                nn.Linear(config.atom_feature_dim, config.wave_hidden_dim),
                nn.SiLU(),
                nn.Linear(config.wave_hidden_dim, config.wave_hidden_dim),
            )
            
            # Shell radii for probing density around atoms
            self.register_buffer(
                'shell_radii',
                torch.linspace(0.0, 2.0, config.wave_num_shells)
            )
            
            # Output projection
            self.output_proj = nn.Linear(config.wave_hidden_dim * 2, config.hidden_dim)
            
        def compute_ground_state(self, atom_coords: Tensor, atom_features: Tensor,
                                  batch: Tensor) -> Tensor:
            """
            Compute ground state density features for each atom.
            
            Args:
                atom_coords: [N, 3] - 3D coordinates
                atom_features: [N, atom_feature_dim] - atom features
                batch: [N] - batch indices
                
            Returns:
                density_features: [N, wave_hidden_dim] - density features per atom
            """
            encoded = self.atom_encoder(atom_features)
            
            # Probe density at shell points around each atom
            N = atom_coords.shape[0]
            device = atom_coords.device
            
            # Create probe points at different radii
            # For simplicity, use 6 directions (±x, ±y, ±z)
            directions = torch.tensor([
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1],
            ], dtype=torch.float32, device=device)
            
            shell_features = []
            for r in self.shell_radii:
                if r == 0:
                    # Center point
                    probe_points = atom_coords
                    density = self.density_field(probe_points, encoded)
                    shell_features.append(density.unsqueeze(-1))
                else:
                    # Points on shell
                    densities = []
                    for d in directions:
                        probe_points = atom_coords + r * d.unsqueeze(0)
                        density = self.density_field(probe_points, encoded)
                        densities.append(density)
                    # Average over shell
                    shell_density = torch.stack(densities, dim=-1).mean(dim=-1)
                    shell_features.append(shell_density.unsqueeze(-1))
            
            # Combine shell features
            shell_tensor = torch.cat(shell_features, dim=-1)  # [N, num_shells]
            
            # Project to hidden dim
            density_features = encoded * shell_tensor.mean(dim=-1, keepdim=True)
            
            return density_features
        
        def simulate_attack(self, atom_idx: int, atom_coords: Tensor,
                           density_features: Tensor, atom_features: Tensor,
                           num_steps: int = 10, dt: float = 0.1) -> Tensor:
            """
            Simulate electron flow when Fe=O attacks atom_idx.
            
            Returns:
                flow_score: scalar - how easily electrons reorganize
            """
            attack_pos = atom_coords[atom_idx]
            encoded = self.atom_encoder(atom_features)
            
            # Initial state
            state = density_features.clone()
            
            # Simulate dynamics
            for step in range(num_steps):
                t = torch.tensor(step * dt, device=atom_coords.device)
                
                # Distance from attack site
                distances = (atom_coords - attack_pos).norm(dim=-1)
                
                # Perturbation from approaching Fe=O
                # Closer = stronger perturbation, increases with time
                perturbation = self.perturbation(distances, t, encoded)
                
                # Compute derivative
                d_state = self.dynamics(state, perturbation, t)
                
                # Euler step (could use RK4 for higher accuracy)
                state = state + dt * d_state
            
            # Measure flow: how much density changed at attack site
            delta = (state - density_features).norm(dim=-1)
            
            # Flow score for the attacked atom
            return delta[atom_idx]
        
        def forward(self, atom_coords: Tensor, atom_features: Tensor,
                   batch: Tensor, candidate_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
            """
            Compute wave field features for all atoms.
            
            Args:
                atom_coords: [N, 3] - 3D coordinates
                atom_features: [N, atom_feature_dim] - atom features
                batch: [N] - batch indices
                candidate_mask: [N] - which atoms to consider as candidates
                
            Returns:
                Dict with:
                - wave_features: [N, hidden_dim] - per-atom wave features
                - flow_scores: [N] - electron flow susceptibility
            """
            # Ground state
            density_features = self.compute_ground_state(atom_coords, atom_features, batch)
            
            # Simulate attack at each candidate and measure flow
            N = atom_coords.shape[0]
            flow_scores = torch.zeros(N, device=atom_coords.device)
            
            if candidate_mask is None:
                candidate_mask = torch.ones(N, dtype=torch.bool, device=atom_coords.device)
            
            # For efficiency, only simulate for candidates
            candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
            
            for idx in candidate_indices:
                flow_scores[idx] = self.simulate_attack(
                    idx.item(), atom_coords, density_features, atom_features,
                    num_steps=self.config.wave_ode_steps
                )
            
            # Combine density features with flow scores
            combined = torch.cat([
                density_features,
                flow_scores.unsqueeze(-1).expand(-1, self.config.wave_hidden_dim)
            ], dim=-1)
            
            wave_features = self.output_proj(combined)
            
            return {
                'wave_features': wave_features,
                'flow_scores': flow_scores,
                'density_features': density_features,
            }


# =============================================================================
# LIQUID CORE MODULE - Adaptive Learning
# =============================================================================

if TORCH_AVAILABLE:
    
    class AdaptiveTauPredictor(nn.Module):
        """
        Predicts per-atom time constants τ based on local environment.
        Complex atoms get larger τ → more ODE iterations → "think longer"
        """
        
        def __init__(self, hidden_dim: int, tau_min: float = 0.1, tau_max: float = 2.0):
            super().__init__()
            self.tau_min = tau_min
            self.tau_max = tau_max
            
            self.net = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),  # atom + neighbors + global
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            
        def forward(self, atom_state: Tensor, neighbor_agg: Tensor,
                    mol_context: Tensor) -> Tensor:
            """
            Predict τ for each atom.
            
            Args:
                atom_state: [N, hidden_dim] - current atom states
                neighbor_agg: [N, hidden_dim] - aggregated neighbor features
                mol_context: [N, hidden_dim] - molecular context (broadcast to atoms)
                
            Returns:
                tau: [N] - time constants (in [tau_min, tau_max])
            """
            x = torch.cat([atom_state, neighbor_agg, mol_context], dim=-1)
            raw = self.net(x).squeeze(-1)
            # Sigmoid to [0, 1], then scale to [tau_min, tau_max]
            tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(raw)
            return tau
    
    
    class LiquidMessagePassing(nn.Module):
        """Edge-aware message passing layer."""
        
        def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
            super().__init__()
            self.hidden_dim = hidden_dim
            
            self.edge_encoder = nn.Linear(edge_dim, hidden_dim) if edge_dim > 0 else None
            
            self.message_net = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
        def forward(self, h: Tensor, edge_index: Tensor, 
                    edge_attr: Optional[Tensor] = None) -> Tensor:
            """
            Compute aggregated messages for each atom.
            
            Args:
                h: [N, hidden_dim] - atom states
                edge_index: [2, E] - edge indices (src, dst)
                edge_attr: [E, edge_dim] - optional edge features
                
            Returns:
                messages: [N, hidden_dim] - aggregated messages
            """
            if edge_index.numel() == 0:
                return torch.zeros_like(h)
            
            src, dst = edge_index[0], edge_index[1]
            
            # Edge features
            if edge_attr is not None and self.edge_encoder is not None:
                edge_feat = self.edge_encoder(edge_attr)
            else:
                edge_feat = torch.zeros(src.shape[0], self.hidden_dim, device=h.device)
            
            # Messages
            messages = self.message_net(torch.cat([h[src], h[dst], edge_feat], dim=-1))
            
            # Aggregate by destination
            out = torch.zeros_like(h)
            out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
            
            # Normalize by degree
            degree = torch.bincount(dst, minlength=h.shape[0]).clamp(min=1).float()
            out = out / degree.unsqueeze(-1)
            
            return out
    
    
    class LiquidCoreModule(nn.Module):
        """
        Liquid Neural Network core with adaptive dynamics.
        
        Key features:
        1. Continuous-time ODE dynamics (not discrete steps)
        2. Per-atom adaptive time constants τ
        3. Adaptive number of steps (converge when uncertain atoms stabilize)
        4. Uncertainty estimation for "think longer" on hard atoms
        """
        
        def __init__(self, config: UCMEConfig):
            super().__init__()
            self.config = config
            
            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(config.atom_feature_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.SiLU(),
            )
            
            # Message passing layers
            self.mp_layers = nn.ModuleList([
                LiquidMessagePassing(config.hidden_dim, config.edge_feature_dim, config.dropout)
                for _ in range(config.liquid_num_layers)
            ])
            
            # Tau predictors per layer
            self.tau_predictors = nn.ModuleList([
                AdaptiveTauPredictor(config.hidden_dim, config.liquid_tau_min, config.liquid_tau_max)
                for _ in range(config.liquid_num_layers)
            ])
            
            # Target networks (for ODE: dh/dt = (-h + f(h)) / τ)
            self.target_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.Tanh(),
                )
                for _ in range(config.liquid_num_layers)
            ])
            
            # Layer norms
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(config.hidden_dim)
                for _ in range(config.liquid_num_layers)
            ])
            
            # Uncertainty head (for adaptive steps)
            self.uncertainty_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Softplus(),
            )
            
            self.dropout = nn.Dropout(config.dropout)
            
        def _compute_mol_context(self, h: Tensor, batch: Tensor) -> Tensor:
            """Compute per-molecule context and expand to atoms."""
            num_mols = batch.max().item() + 1 if batch.numel() > 0 else 0
            
            # Mean pooling per molecule
            mol_context = torch.zeros(num_mols, h.shape[-1], device=h.device)
            counts = torch.bincount(batch, minlength=num_mols).clamp(min=1).float()
            mol_context.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)
            mol_context = mol_context / counts.unsqueeze(-1)
            
            # Expand back to atoms
            return mol_context[batch]
        
        def _ode_step_rk4(self, h: Tensor, messages: Tensor, tau: Tensor,
                          target_net: nn.Module, dt: float) -> Tensor:
            """Single RK4 step for the liquid dynamics."""
            
            def dynamics(state):
                target = target_net(torch.cat([state, messages], dim=-1))
                return (-state + target) / tau.unsqueeze(-1)
            
            k1 = dynamics(h)
            k2 = dynamics(h + 0.5 * dt * k1)
            k3 = dynamics(h + 0.5 * dt * k2)
            k4 = dynamics(h + dt * k3)
            
            return h + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        def forward(self, atom_features: Tensor, edge_index: Tensor,
                   batch: Tensor, edge_attr: Optional[Tensor] = None) -> Dict[str, Tensor]:
            """
            Process atoms through liquid dynamics.
            
            Args:
                atom_features: [N, atom_feature_dim]
                edge_index: [2, E]
                batch: [N]
                edge_attr: [E, edge_dim] optional
                
            Returns:
                Dict with:
                - atom_states: [N, hidden_dim] - final atom representations
                - tau_history: list of [N] tensors - τ values per layer
                - uncertainties: [N] - per-atom uncertainty
                - num_steps: int - total ODE steps taken
            """
            h = self.dropout(self.input_proj(atom_features))
            
            tau_history = []
            total_steps = 0
            
            for layer_idx, (mp, tau_pred, target_net, ln) in enumerate(
                zip(self.mp_layers, self.tau_predictors, self.target_nets, self.layer_norms)
            ):
                # Get messages
                messages = mp(h, edge_index, edge_attr)
                
                # Get molecular context
                mol_context = self._compute_mol_context(h, batch)
                
                # Predict per-atom tau
                tau = tau_pred(h, messages, mol_context)
                tau_history.append(tau)
                
                # ODE integration
                dt = 1.0 / self.config.liquid_ode_steps
                
                if self.config.liquid_adaptive_steps and self.training:
                    # Adaptive: continue until convergence or max steps
                    prev_h = h
                    for step in range(self.config.liquid_max_steps):
                        h = self._ode_step_rk4(h, messages, tau, target_net, dt)
                        total_steps += 1
                        
                        # Check convergence
                        delta = (h - prev_h).abs().max()
                        if delta < self.config.liquid_convergence_threshold:
                            break
                        prev_h = h
                else:
                    # Fixed number of steps
                    for _ in range(self.config.liquid_ode_steps):
                        h = self._ode_step_rk4(h, messages, tau, target_net, dt)
                        total_steps += 1
                
                # Residual + norm
                h = ln(h + self.dropout(messages))
            
            # Compute uncertainty
            uncertainties = self.uncertainty_head(h).squeeze(-1)
            
            return {
                'atom_states': h,
                'tau_history': tau_history,
                'uncertainties': uncertainties,
                'num_steps': total_steps,
            }


# =============================================================================
# ANALOGICAL REASONER MODULE - Human-like Reasoning
# =============================================================================

if TORCH_AVAILABLE and RDKIT_AVAILABLE:
    
    class HyperbolicEmbedding(nn.Module):
        """
        Poincaré ball model for hyperbolic embeddings.
        Better for hierarchical molecular similarity.
        """
        
        def __init__(self, dim: int, curvature: float = 1.0):
            super().__init__()
            self.dim = dim
            self.c = curvature  # curvature
            
        def exp_map(self, v: Tensor, x: Optional[Tensor] = None) -> Tensor:
            """Exponential map from tangent space to Poincaré ball."""
            if x is None:
                x = torch.zeros_like(v)
            
            v_norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-10)
            sqrt_c = math.sqrt(self.c)
            
            # Möbius addition and exp map
            lambda_x = 2 / (1 - self.c * (x * x).sum(dim=-1, keepdim=True)).clamp(min=1e-10)
            second_term = torch.tanh(sqrt_c * lambda_x * v_norm / 2) * v / (sqrt_c * v_norm)
            
            return self._mobius_add(x, second_term)
        
        def _mobius_add(self, x: Tensor, y: Tensor) -> Tensor:
            """Möbius addition in Poincaré ball."""
            x2 = (x * x).sum(dim=-1, keepdim=True)
            y2 = (y * y).sum(dim=-1, keepdim=True)
            xy = (x * y).sum(dim=-1, keepdim=True)
            
            num = (1 + 2*self.c*xy + self.c*y2) * x + (1 - self.c*x2) * y
            denom = 1 + 2*self.c*xy + self.c**2 * x2 * y2
            
            return num / denom.clamp(min=1e-10)
        
        def distance(self, x: Tensor, y: Tensor) -> Tensor:
            """Hyperbolic distance between points."""
            diff = self._mobius_add(-x, y)
            diff_norm = diff.norm(dim=-1).clamp(max=1-1e-5)
            
            sqrt_c = math.sqrt(self.c)
            return 2 / sqrt_c * torch.atanh(sqrt_c * diff_norm)
        
        def project(self, x: Tensor, eps: float = 1e-5) -> Tensor:
            """Project points to stay inside the Poincaré ball."""
            norm = x.norm(dim=-1, keepdim=True)
            max_norm = (1 - eps) / math.sqrt(self.c)
            return x * (max_norm / norm.clamp(min=max_norm))
    
    
    class StructureAligner:
        """
        Aligns molecular structures using Maximum Common Substructure (MCS).
        Returns atom-to-atom mapping between query and reference molecules.
        """
        
        def __init__(self, max_time: float = 1.0):
            self.max_time = max_time
            
        def align(self, query_mol: Chem.Mol, ref_mol: Chem.Mol) -> Dict[int, int]:
            """
            Find atom mapping from query to reference using MCS.
            
            Args:
                query_mol: Query molecule
                ref_mol: Reference molecule
                
            Returns:
                mapping: Dict[query_atom_idx -> ref_atom_idx]
            """
            try:
                # Find MCS with timeout
                mcs_result = rdFMCS.FindMCS(
                    [query_mol, ref_mol],
                    timeout=int(self.max_time),
                    completeRingsOnly=True,
                    ringMatchesRingOnly=True,
                    atomCompare=rdFMCS.AtomCompare.CompareAny,
                    bondCompare=rdFMCS.BondCompare.CompareAny,
                )
                
                if mcs_result.numAtoms == 0:
                    return {}
                
                # Get                # Get the MCS pattern
                mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                if mcs_mol is None:
                    return {}
                
                # Find matches in both molecules
                query_match = query_mol.GetSubstructMatch(mcs_mol)
                ref_match = ref_mol.GetSubstructMatch(mcs_mol)
                
                if not query_match or not ref_match:
                    return {}
                
                # Create mapping
                mapping = {q: r for q, r in zip(query_match, ref_match)}
                return mapping
                
            except Exception:
                return {}
    
    
    class EnvironmentComparer(nn.Module):
        """
        Compares local chemical environments between aligned atoms.
        Outputs difference features that indicate why sites might behave differently.
        """
        
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim
            
            # Encode local environment
            self.env_encoder = nn.Sequential(
                nn.Linear(hidden_dim + 8, hidden_dim),  # features + descriptors
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            
            # Compare environments
            self.diff_net = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
            )
            
        def compute_env_descriptors(self, mol: Chem.Mol, atom_idx: int) -> torch.Tensor:
            """Compute local environment descriptors for an atom."""
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # Steric: number of heavy neighbors, ring membership
            n_neighbors = len([n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1])
            in_ring = 1.0 if atom.IsInRing() else 0.0
            ring_size = min(atom.GetRingSize(), 8) / 8.0 if atom.IsInRing() else 0.0
            
            # Electronic: formal charge, hybridization
            charge = float(atom.GetFormalCharge())
            hybridization = {
                Chem.HybridizationType.SP: 0.25,
                Chem.HybridizationType.SP2: 0.5,
                Chem.HybridizationType.SP3: 0.75,
            }.get(atom.GetHybridization(), 0.0)
            
            # Aromaticity
            aromatic = 1.0 if atom.GetIsAromatic() else 0.0
            
            # Neighbor electronegativities (simplified)
            neighbor_en = 0.0
            for n in atom.GetNeighbors():
                neighbor_en += {6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 16: 2.58, 17: 3.16}.get(
                    n.GetAtomicNum(), 2.5
                )
            neighbor_en = neighbor_en / max(n_neighbors, 1) / 4.0  # Normalize
            
            # H count
            h_count = float(atom.GetTotalNumHs()) / 4.0
            
            return torch.tensor([
                n_neighbors / 4.0, in_ring, ring_size, charge / 2.0,
                hybridization, aromatic, neighbor_en, h_count
            ], dtype=torch.float32)
        
        def forward(self, query_features: Tensor, ref_features: Tensor,
                   query_mol: Chem.Mol, ref_mol: Chem.Mol,
                   mapping: Dict[int, int]) -> Tuple[Tensor, Tensor]:
            """
            Compare environments of aligned atoms.
            
            Args:
                query_features: [N_query, hidden_dim] - query atom features
                ref_features: [N_ref, hidden_dim] - reference atom features
                query_mol: Query RDKit molecule
                ref_mol: Reference RDKit molecule
                mapping: query_idx -> ref_idx
                
            Returns:
                diff_features: [N_query, hidden_dim//2] - environment differences
                aligned_mask: [N_query] - which atoms have aligned counterparts
            """
            N = query_features.shape[0]
            device = query_features.device
            
            diff_features = torch.zeros(N, self.hidden_dim // 2, device=device)
            aligned_mask = torch.zeros(N, dtype=torch.bool, device=device)
            
            for q_idx, r_idx in mapping.items():
                if q_idx >= N or r_idx >= ref_features.shape[0]:
                    continue
                    
                # Get descriptors
                q_desc = self.compute_env_descriptors(query_mol, q_idx).to(device)
                r_desc = self.compute_env_descriptors(ref_mol, r_idx).to(device)
                
                # Encode environments
                q_env = self.env_encoder(torch.cat([query_features[q_idx], q_desc]))
                r_env = self.env_encoder(torch.cat([ref_features[r_idx], r_desc]))
                
                # Compute difference
                diff = self.diff_net(torch.cat([q_env, r_env]))
                
                diff_features[q_idx] = diff
                aligned_mask[q_idx] = True
            
            return diff_features, aligned_mask
    
    
    class CounterfactualReasoner(nn.Module):
        """
        Performs counterfactual reasoning:
        "If reference was metabolized at site X, and query site Y is aligned to X,
         but Y has different properties, how should we adjust the prediction?"
        """
        
        def __init__(self, hidden_dim: int):
            super().__init__()
            
            # Learn how environment differences affect SoM likelihood
            self.adjustment_net = nn.Sequential(
                nn.Linear(hidden_dim // 2 + 1, hidden_dim // 2),  # diff + ref_score
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            
            # Alternative site finder
            self.alternative_net = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            
        def forward(self, query_features: Tensor, diff_features: Tensor,
                   aligned_mask: Tensor, ref_scores: Tensor,
                   mapping: Dict[int, int]) -> Tuple[Tensor, List[str]]:
            """
            Compute counterfactual score adjustments.
            
            Args:
                query_features: [N, hidden_dim] - query atom features
                diff_features: [N, hidden_dim//2] - environment differences
                aligned_mask: [N] - which atoms are aligned
                ref_scores: [N_ref] - reference SoM scores
                mapping: query_idx -> ref_idx
                
            Returns:
                adjustments: [N] - score adjustments per atom
                explanations: list of strings explaining reasoning
            """
            N = query_features.shape[0]
            device = query_features.device
            
            adjustments = torch.zeros(N, device=device)
            explanations = []
            
            for q_idx, r_idx in mapping.items():
                if q_idx >= N or r_idx >= ref_scores.shape[0]:
                    continue
                
                if not aligned_mask[q_idx]:
                    continue
                
                # Get reference score for aligned atom
                r_score = ref_scores[r_idx].unsqueeze(0)
                
                # Compute adjustment based on environment difference
                adj_input = torch.cat([diff_features[q_idx], r_score])
                adjustment = self.adjustment_net(adj_input).squeeze()
                
                adjustments[q_idx] = adjustment
                
                # Generate explanation
                if adjustment > 0.1:
                    explanations.append(
                        f"Site {q_idx}: boosted (similar to ref site {r_idx}, favorable env)"
                    )
                elif adjustment < -0.1:
                    explanations.append(
                        f"Site {q_idx}: penalized (aligned to ref site {r_idx}, but hindered)"
                    )
            
            # Find alternative sites when aligned sites are hindered
            hindered_mask = (adjustments < -0.1) & aligned_mask
            if hindered_mask.any():
                # Score unaligned atoms as potential alternatives
                unaligned_mask = ~aligned_mask
                if unaligned_mask.any():
                    for q_idx in hindered_mask.nonzero(as_tuple=True)[0]:
                        # Combine features from hindered site with unaligned candidates
                        hindered_feat = diff_features[q_idx]
                        
                        for alt_idx in unaligned_mask.nonzero(as_tuple=True)[0]:
                            alt_input = torch.cat([
                                query_features[alt_idx], hindered_feat
                            ])
                            alt_score = self.alternative_net(alt_input).squeeze()
                            adjustments[alt_idx] = adjustments[alt_idx] + 0.3 * alt_score
                            
                            if alt_score > 0.2:
                                explanations.append(
                                    f"Site {alt_idx.item()}: alternative to hindered site {q_idx.item()}"
                                )
            
            return adjustments, explanations
    
    
    @dataclass
    class MemoryEntry:
        """Single entry in the analogical memory bank."""
        smiles: str
        mol: Any  # RDKit Mol
        som_sites: List[int]
        atom_features: Optional[Tensor] = None
        embedding: Optional[Tensor] = None
        scores: Optional[Tensor] = None
    
    
    class AnalogicalMemoryBank(nn.Module):
        """
        Memory bank storing molecules and their SoM sites.
        Supports fast retrieval via learned hyperbolic embeddings.
        """
        
        def __init__(self, config: UCMEConfig):
            super().__init__()
            self.config = config
            
            # Hyperbolic embedding space
            self.hyperbolic = HyperbolicEmbedding(config.hyperbolic_dim, config.hyperbolic_curvature)
            
            # Molecule encoder (SMILES -> embedding)
            self.mol_encoder = nn.Sequential(
                nn.Linear(2048, 512),  # Morgan FP size
                nn.SiLU(),
                nn.Linear(512, 256),
                nn.SiLU(),
                nn.Linear(256, config.hyperbolic_dim),
            )
            
            # Memory storage
            self.memory: List[MemoryEntry] = []
            self.memory_embeddings: Optional[Tensor] = None
            
            # For projecting to hyperbolic space
            self.to_tangent = nn.Linear(config.hyperbolic_dim, config.hyperbolic_dim)
            
        def encode_molecule(self, mol: Chem.Mol) -> Tensor:
            """Encode molecule to hyperbolic embedding."""
            fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_array = torch.tensor([fp.GetBit(i) for i in range(2048)], dtype=torch.float32)
            
            # Project to tangent space, then exp map to hyperbolic
            tangent = self.to_tangent(self.mol_encoder(fp_array))
            return self.hyperbolic.exp_map(tangent)
        
        def add(self, smiles: str, som_sites: List[int], 
                atom_features: Optional[Tensor] = None,
                scores: Optional[Tensor] = None):
            """Add a molecule to memory."""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return
            
            with torch.no_grad():
                embedding = self.encode_molecule(mol)
            
            entry = MemoryEntry(
                smiles=smiles,
                mol=mol,
                som_sites=som_sites,
                atom_features=atom_features,
                embedding=embedding,
                scores=scores,
            )
            self.memory.append(entry)
            
            # Update embedding matrix
            self._update_embedding_matrix()
        
        def _update_embedding_matrix(self):
            """Update the stacked embedding matrix for fast retrieval."""
            if not self.memory:
                self.memory_embeddings = None
                return
            
            embeddings = [e.embedding for e in self.memory if e.embedding is not None]
            if embeddings:
                self.memory_embeddings = torch.stack(embeddings)
        
        def retrieve(self, query_mol: Chem.Mol, k: int = 16) -> List[Tuple[MemoryEntry, float]]:
            """
            Retrieve top-k similar molecules from memory.
            
            Returns:
                List of (entry, similarity_score) tuples
            """
            if not self.memory or self.memory_embeddings is None:
                return []
            
            with torch.no_grad():
                query_emb = self.encode_molecule(query_mol)
                
                # Compute hyperbolic distances
                distances = self.hyperbolic.distance(
                    query_emb.unsqueeze(0),
                    self.memory_embeddings
                )
                
                # Convert to similarity (closer = more similar)
                similarities = torch.exp(-distances)
                
                # Get top-k
                k = min(k, len(self.memory))
                top_k = torch.topk(similarities, k)
            
            results = []
            for idx, sim in zip(top_k.indices.tolist(), top_k.values.tolist()):
                results.append((self.memory[idx], sim))
            
            return results
    
    
    class AnalogicalReasonerModule(nn.Module):
        """
        Complete Analogical Reasoning module.
        
        Workflow:
        1. Retrieve similar molecules from memory
        2. Align structures using MCS
        3. Compare environments of aligned atoms
        4. Apply counterfactual reasoning
        5. Output adjusted scores with explanations
        """
        
        def __init__(self, config: UCMEConfig):
            super().__init__()
            self.config = config
            
            # Components
            self.memory = AnalogicalMemoryBank(config)
            self.aligner = StructureAligner(max_time=config.max_mcs_time)
            self.comparer = EnvironmentComparer(config.hidden_dim)
            self.reasoner = CounterfactualReasoner(config.hidden_dim)
            
            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(config.hidden_dim + config.hidden_dim // 2 + 1, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            
        def forward(self, smiles: str, atom_features: Tensor,
                   batch: Optional[Tensor] = None) -> Dict[str, Any]:
            """
            Perform analogical reasoning for a molecule.
            
            Args:
                smiles: Query molecule SMILES
                atom_features: [N, hidden_dim] - atom features from liquid core
                batch: [N] - batch indices (for batched processing)
                
            Returns:
                Dict with:
                - analogical_features: [N, hidden_dim]
                - analogical_scores: [N]
                - explanations: list of strings
            """
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                N = atom_features.shape[0]
                device = atom_features.device
                return {
                    'analogical_features': torch.zeros(N, self.config.hidden_dim, device=device),
                    'analogical_scores': torch.zeros(N, device=device),
                    'explanations': [],
                }
            
            N = atom_features.shape[0]
            device = atom_features.device
            
            # Retrieve similar molecules
            similar = self.memory.retrieve(mol, k=self.config.memory_topk)
            
            if not similar:
                return {
                    'analogical_features': torch.zeros(N, self.config.hidden_dim, device=device),
                    'analogical_scores': torch.zeros(N, device=device),
                    'explanations': ["No similar molecules in memory"],
                }
            
            # Aggregate reasoning over similar molecules
            all_adjustments = []
            all_explanations = []
            all_diff_features = []
            
            for entry, similarity in similar:
                # Align structures
                if self.config.use_mcs_alignment:
                    mapping = self.aligner.align(mol, entry.mol)
                else:
                    mapping = {}
                
                if not mapping:
                    continue
                
                # Get reference features (use stored or zeros)
                if entry.atom_features is not None:
                    ref_features = entry.atom_features.to(device)
                else:
                    ref_features = torch.zeros(
                        entry.mol.GetNumAtoms(), self.config.hidden_dim, device=device
                    )
                
                # Compare environments
                diff_features, aligned_mask = self.comparer(
                    atom_features, ref_features, mol, entry.mol, mapping
                )
                
                # Get reference scores
                if entry.scores is not None:
                    ref_scores = entry.scores.to(device)
                else:
                    # Create scores from SoM sites
                    ref_scores = torch.zeros(entry.mol.GetNumAtoms(), device=device)
                    for site in entry.som_sites:
                        if site < len(ref_scores):
                            ref_scores[site] = 1.0
                
                # Counterfactual reasoning
                adjustments, explanations = self.reasoner(
                    atom_features, diff_features, aligned_mask, ref_scores, mapping
                )
                
                # Weight by similarity
                all_adjustments.append(similarity * adjustments)
                all_diff_features.append(diff_features)
                all_explanations.extend(explanations)
            
            # Aggregate
            if all_adjustments:
                analogical_scores = torch.stack(all_adjustments).sum(dim=0)
                agg_diff = torch.stack(all_diff_features).mean(dim=0)
            else:
                analogical_scores = torch.zeros(N, device=device)
                agg_diff = torch.zeros(N, self.config.hidden_dim // 2, device=device)
            
            # Create output features
            combined = torch.cat([
                atom_features,
                agg_diff,
                analogical_scores.unsqueeze(-1)
            ], dim=-1)
            
            analogical_features = self.output_proj(combined)
            
            return {
                'analogical_features': analogical_features,
                'analogical_scores': analogical_scores,
                'explanations': all_explanations,
            }


# =============================================================================
# UNIFIED COGNITIVE METABOLISM ENGINE
# =============================================================================

if TORCH_AVAILABLE:
    
    class GatedFusion(nn.Module):
        """Learns to fuse multiple signal streams with per-atom gating."""
        
        def __init__(self, hidden_dim: int, num_streams: int):
            super().__init__()
            self.gate_net = nn.Sequential(
                nn.Linear(hidden_dim * num_streams, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_streams),
            )
            
        def forward(self, *streams: Tensor) -> Tuple[Tensor, Tensor]:
            """
            Fuse multiple streams with learned gates.
            
            Args:
                *streams: Variable number of [N, hidden_dim] tensors
                
            Returns:
                fused: [N, hidden_dim] - fused features
                gates: [N, num_streams] - gate values (for interpretability)
            """
            combined = torch.cat(streams, dim=-1)
            gate_logits = self.gate_net(combined)
            gates = F.softmax(gate_logits, dim=-1)
            
            # Stack streams and apply gates
            stacked = torch.stack(streams, dim=-1)  # [N, hidden_dim, num_streams]
            fused = (stacked * gates.unsqueeze(1)).sum(dim=-1)  # [N, hidden_dim]
            
            return fused, gates
    
    
    class SoMHead(nn.Module):
        """Final SoM prediction head with ListMLE ranking loss."""
        
        def __init__(self, hidden_dim: int, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            
        def forward(self, features: Tensor) -> Tensor:
            """Compute SoM scores for each atom."""
            return self.net(features).squeeze(-1)
        
        def listmle_loss(self, scores: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
            """
            ListMLE ranking loss.
            
            Args:
                scores: [N] - predicted scores
                labels: [N] - ground truth (1 for SoM, 0 otherwise)
                mask: [N] - which atoms are valid candidates
                
            Returns:
                loss: scalar
            """
            if not mask.any():
                return scores.new_zeros(1)
            
            # Filter to valid candidates
            valid_scores = scores[mask]
            valid_labels = labels[mask]
            
            if valid_labels.sum() == 0:
                return scores.new_zeros(1)
            
            # Sort by ground truth (positive first)
            sorted_indices = torch.argsort(valid_labels, descending=True)
            sorted_scores = valid_scores[sorted_indices]
            
            # ListMLE: -sum(log(softmax(remaining scores)))
            n = len(sorted_scores)
            loss = 0.0
            for i in range(n):
                remaining = sorted_scores[i:]
                log_sum_exp = torch.logsumexp(remaining, dim=0)
                loss = loss - sorted_scores[i] + log_sum_exp
            
            return loss / n
    
    
    class UCME(nn.Module):
        """
        Unified Cognitive Metabolism Engine.
        
        Combines Wave Field (electron dynamics), Liquid Core (adaptive GNN),
        and Analogical Reasoner (human-like reasoning) for SoM prediction.
        """
        
        def __init__(self, config: Optional[UCMEConfig] = None):
            super().__init__()
            self.config = config or UCMEConfig()
            
            # Input embedding
            self.atom_embedding = nn.Embedding(100, self.config.atom_feature_dim)  # Atomic numbers
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.config.atom_feature_dim + 32, self.config.atom_feature_dim),
                nn.LayerNorm(self.config.atom_feature_dim),
                nn.SiLU(),
            )
            
            # Core modules
            if self.config.use_wave:
                self.wave_field = WaveFieldModule(self.config)
            else:
                self.wave_field = None
                
            if self.config.use_liquid:
                self.liquid_core = LiquidCoreModule(self.config)
            else:
                self.liquid_core = None
            
            if self.config.use_analogical and RDKIT_AVAILABLE:
                self.analogical = AnalogicalReasonerModule(self.config)
            else:
                self.analogical = None
            
            # Determine number of active streams
            num_streams = sum([
                self.config.use_wave,
                self.config.use_liquid,
                self.config.use_analogical
            ])
            num_streams = max(num_streams, 1)  # At least one stream
            
            # Integration
            self.fusion = GatedFusion(self.config.hidden_dim, num_streams)
            
            # Output head
            self.som_head = SoMHead(self.config.hidden_dim, self.config.dropout)
            
        def encode_atoms(self, mol: Chem.Mol) -> Tuple[Tensor, Tensor, Tensor]:
            """
            Encode molecule atoms to features.
            
            Returns:
                atom_features: [N, atom_feature_dim]
                edge_index: [2, E]
                atom_coords: [N, 3]
            """
            # Get atomic numbers
            atomic_nums = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
            
            # Embed
            embeddings = self.atom_embedding(atomic_nums)
            
            # Add additional features
            extra_features = []
            for atom in mol.GetAtoms():
                feat = [
                    atom.GetTotalDegree() / 4.0,
                    atom.GetTotalNumHs() / 4.0,
                    1.0 if atom.GetIsAromatic() else 0.0,
                    1.0 if atom.IsInRing() else 0.0,
                    atom.GetFormalCharge() / 2.0,
                    {
                        Chem.HybridizationType.SP: 0.25,
                        Chem.HybridizationType.SP2: 0.5,
                        Chem.HybridizationType.SP3: 0.75,
                    }.get(atom.GetHybridization(), 0.0),
                ]
                # Pad to 32
                feat = feat + [0.0] * (32 - len(feat))
                extra_features.append(feat)
            
            extra = torch.tensor(extra_features, dtype=torch.float32)
            atom_features = self.feature_encoder(torch.cat([embeddings, extra], dim=-1))
            
            # Build edge index
            edges_src, edges_dst = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
            
            if edges_src:
                edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
            
            # Generate 3D coordinates
            mol_3d = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol_3d, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_3d)
                conf = mol_3d.GetConformer()
                coords = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                atom_coords = torch.tensor(coords, dtype=torch.float32)
            except Exception:
                # Fallback: random coordinates
                atom_coords = torch.randn(mol.GetNumAtoms(), 3)
            
            return atom_features, edge_index, atom_coords
        
        def forward(self, smiles: str, labels: Optional[Tensor] = None,
                   candidate_mask: Optional[Tensor] = None) -> Dict[str, Any]:
            """
            Forward pass for a single molecule.
            
            Args:
                smiles: Molecule SMILES
                labels: [N] - ground truth SoM labels (optional, for training)
                candidate_mask: [N] - which atoms are candidates (optional)
                
            Returns:
                Dict with predictions, features, and explanations
            """
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Encode atoms
            atom_features, edge_index, atom_coords = self.encode_atoms(mol)
            N = atom_features.shape[0]
            batch = torch.zeros(N, dtype=torch.long, device=atom_features.device)
            
            if candidate_mask is None:
                # Default: carbons with H are candidates
                candidate_mask = torch.tensor([
                    a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0
                    for a in mol.GetAtoms()
                ], dtype=torch.bool)
            
            # Process through modules
            streams = []
            module_outputs = {}
            
            # Wave Field
            if self.wave_field is not None:
                wave_out = self.wave_field(
                    atom_coords, atom_features, batch, candidate_mask
                )
                streams.append(wave_out['wave_features'])
                module_outputs['wave'] = wave_out
            
            # Liquid Core
            if self.liquid_core is not None:
                liquid_out = self.liquid_core(
                    atom_features, edge_index, batch
                )
                streams.append(liquid_out['atom_states'])
                module_outputs['liquid'] = liquid_out
            
            # Analogical Reasoner
            if self.analogical is not None:
                # Use liquid features if available, else atom features
                input_features = streams[-1] if streams else atom_features
                ana_out = self.analogical(smiles, input_features, batch)
                streams.append(ana_out['analogical_features'])
                module_outputs['analogical'] = ana_out
            
            # Fuse streams
            if len(streams) == 0:
                fused = atom_features
                gates = None
            elif len(streams) == 1:
                fused = streams[0]
                gates = torch.ones(N, 1)
            else:
                fused, gates = self.fusion(*streams)
            
            # Predict SoM scores
            scores = self.som_head(fused)
            
            # Compute loss if labels provided
            loss = None
            if labels is not None:
                loss = self.som_head.listmle_loss(scores, labels, candidate_mask)
            
            # Collect explanations
            explanations = []
            if 'analogical' in module_outputs:
                explanations.extend(module_outputs['analogical'].get('explanations', []))
            
            return {
                'scores': scores,
                'fused_features': fused,
                'gates': gates,
                'loss': loss,
                'explanations': explanations,
                'module_outputs': module_outputs,
                'candidate_mask': candidate_mask,
            }
        
        def predict(self, smiles: str, top_k: int = 3) -> List[Tuple[int, float, List[str]]]:
            """
            Predict top-k SoM sites for a molecule.
            
            Args:
                smiles: Molecule SMILES
                top_k: Number of sites to return
                
            Returns:
                List of (atom_idx, score, explanations) tuples
            """
            self.eval()
            with torch.no_grad():
                out = self.forward(smiles)
            
            scores = out['scores']
            mask = out['candidate_mask']
            explanations = out['explanations']
            
            # Get top-k among candidates
            candidate_scores = scores.clone()
            candidate_scores[~mask] = float('-inf')
            
            top_k = min(top_k, mask.sum().item())
            top_indices = torch.topk(candidate_scores, top_k).indices
            
            results = []
            for idx in top_indices:
                idx = idx.item()
                score = scores[idx].item()
                # Filter explanations relevant to this site
                site_explanations = [e for e in explanations if f"Site {idx}" in e]
                results.append((idx, score, site_explanations))
            
            return results
        
        def add_to_memory(self, smiles: str, som_sites: List[int]):
            """Add a molecule to the analogical memory bank."""
            if self.analogical is not None:
                self.analogical.memory.add(smiles, som_sites)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_ucme(
    hidden_dim: int = 128,
    use_wave: bool = True,
    use_liquid: bool = True,
    use_analogical: bool = True,
    **kwargs
) -> "UCME":
    """
    Create a UCME model with specified configuration.
    
    Args:
        hidden_dim: Hidden dimension for all modules
        use_wave: Enable Wave Field module
        use_liquid: Enable Liquid Core module
        use_analogical: Enable Analogical Reasoner module
        **kwargs: Additional config parameters
        
    Returns:
        UCME model
    """
    config = UCMEConfig(
        hidden_dim=hidden_dim,
        use_wave=use_wave,
        use_liquid=use_liquid,
        use_analogical=use_analogical,
        **kwargs
    )
    return UCME(config)
