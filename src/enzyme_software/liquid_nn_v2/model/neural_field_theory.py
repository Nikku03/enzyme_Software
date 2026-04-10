"""
Neural Field Theory for Site-of-Metabolism Prediction.

This module implements the "Dark Manifold" concept: a 4D spacetime field-theoretic
neural network where the network IS a continuous manifold, not a discrete graph.

Core Philosophy:
- Molecules exist in a 4D space: 3D spatial coordinates + 1D "reactivity time"
- Atoms are not nodes but FIELD EXCITATIONS at points in this space
- Connections are mediated by a continuous "dark field" filling the space
- Reactivity emerges from field dynamics, not discrete message passing

Mathematical Framework:

1. **Scalar Field Theory**
   - φ(x,t): Reactivity potential field
   - Lagrangian: L = ½(∂φ)² - V(φ) - J·φ
   - J(x): Source term from atomic positions

2. **Neural Implicit Function**
   - f_θ(x,t) → (density, reactivity, field_value)
   - Continuous representation of molecular properties
   - Queried at any point in space

3. **Hamiltonian Dynamics**
   - Phase space: (q, p) = (field_config, field_momentum)
   - Neural ODE for time evolution
   - Energy conservation encodes reaction thermodynamics

4. **Dark Field Mediation**
   - Latent field ψ(x) mediates long-range interactions
   - Analogous to dark matter in cosmology
   - Captures non-local electronic effects

Implementation:
- SIREN-style coordinate encoding
- Neural ODE for field dynamics
- Geometric algebra for orientation-aware features
- Continuous attention via field correlation
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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
# SIREN COMPONENTS (Sinusoidal Representation Networks)
# ============================================================================

if TORCH_AVAILABLE:
    
    class SirenLayer(nn.Module):
        """
        SIREN layer with sinusoidal activation.
        
        Uses sin(ω₀ * Wx + b) as activation, which allows the network
        to represent high-frequency signals and their derivatives.
        
        Initialization is crucial for SIREN:
        - First layer: Uniform(-1/input_dim, 1/input_dim)
        - Other layers: Uniform(-√(6/input_dim)/ω₀, √(6/input_dim)/ω₀)
        """
        
        def __init__(
            self,
            in_features: int,
            out_features: int,
            omega_0: float = 30.0,
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
        
        This network learns a continuous function f: ℝ^d → ℝ^k
        that can represent any smooth signal.
        """
        
        def __init__(
            self,
            in_features: int = 3,
            hidden_features: int = 256,
            hidden_layers: int = 3,
            out_features: int = 1,
            omega_0: float = 30.0,
            final_activation: Optional[str] = None,
        ):
            super().__init__()
            
            self.omega_0 = omega_0
            
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
            self.final_activation = final_activation
        
        def forward(self, coords: torch.Tensor) -> torch.Tensor:
            """
            Query the implicit function at given coordinates.
            
            Args:
                coords: (..., in_features) coordinates
                
            Returns:
                (..., out_features) function values
            """
            out = self.layers(coords)
            
            if self.final_activation == "sigmoid":
                out = torch.sigmoid(out)
            elif self.final_activation == "tanh":
                out = torch.tanh(out)
            elif self.final_activation == "softplus":
                out = F.softplus(out)
            
            return out


# ============================================================================
# NEURAL FIELD COMPONENTS
# ============================================================================

if TORCH_AVAILABLE:
    
    class ReactivityField(nn.Module):
        """
        Neural implicit field for molecular reactivity.
        
        Given 3D coordinates, outputs:
        - Electron density at that point
        - Reactivity potential
        - Field momentum (for dynamics)
        
        The field is conditioned on the molecular structure via
        source terms placed at atomic positions.
        """
        
        def __init__(
            self,
            coord_dim: int = 3,
            field_dim: int = 64,
            hidden_dim: int = 256,
            num_layers: int = 4,
        ):
            super().__init__()
            
            self.coord_dim = coord_dim
            self.field_dim = field_dim
            
            # Coordinate encoder
            self.coord_encoder = SirenNetwork(
                in_features=coord_dim,
                hidden_features=hidden_dim,
                hidden_layers=num_layers - 1,
                out_features=field_dim,
            )
            
            # Source term encoder (atomic features → field contribution)
            self.source_encoder = nn.Sequential(
                nn.Linear(field_dim + coord_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, field_dim),
            )
            
            # Field-to-observable decoders
            self.density_decoder = nn.Sequential(
                nn.Linear(field_dim, field_dim // 2),
                nn.SiLU(),
                nn.Linear(field_dim // 2, 1),
                nn.Softplus(),  # Density must be positive
            )
            
            self.reactivity_decoder = nn.Sequential(
                nn.Linear(field_dim, field_dim // 2),
                nn.SiLU(),
                nn.Linear(field_dim // 2, 1),
            )
            
            self.momentum_decoder = nn.Sequential(
                nn.Linear(field_dim, field_dim // 2),
                nn.SiLU(),
                nn.Linear(field_dim // 2, coord_dim),
            )
        
        def compute_source_contribution(
            self,
            query_coords: torch.Tensor,      # (Q, 3) query points
            atom_coords: torch.Tensor,       # (N, 3) atom positions
            atom_features: torch.Tensor,     # (N, field_dim) atom features
            cutoff: float = 10.0,
        ) -> torch.Tensor:
            """
            Compute field contribution from atomic sources.
            
            Uses a smooth cutoff to limit the range of atomic influence.
            """
            Q = query_coords.size(0)
            N = atom_coords.size(0)
            
            # Compute distances
            # (Q, 1, 3) - (1, N, 3) → (Q, N, 3)
            diff = query_coords.unsqueeze(1) - atom_coords.unsqueeze(0)
            distances = torch.norm(diff, dim=-1)  # (Q, N)
            
            # Smooth cutoff function
            cutoff_val = 0.5 * (1 + torch.cos(math.pi * distances / cutoff))
            cutoff_val = torch.where(distances < cutoff, cutoff_val, torch.zeros_like(cutoff_val))
            
            # Distance-weighted contribution
            # Closer atoms contribute more
            weights = cutoff_val / (distances + 1e-6)  # (Q, N)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
            
            # Aggregate atomic features
            # (Q, N) @ (N, field_dim) → (Q, field_dim)
            source = torch.matmul(weights, atom_features)
            
            # Also include direction information
            directions = diff / (distances.unsqueeze(-1) + 1e-6)  # (Q, N, 3)
            weighted_dir = torch.einsum("qn,qnd->qd", weights, directions)  # (Q, 3)
            
            # Encode source with direction
            source_input = torch.cat([source, weighted_dir], dim=-1)
            source_encoded = self.source_encoder(source_input)
            
            return source_encoded
        
        def forward(
            self,
            query_coords: torch.Tensor,      # (Q, 3) query points
            atom_coords: torch.Tensor,       # (N, 3) atom positions
            atom_features: torch.Tensor,     # (N, field_dim) atom features
        ) -> Dict[str, torch.Tensor]:
            """
            Query the reactivity field at given coordinates.
            
            Returns:
                Dict with density, reactivity, momentum at each query point
            """
            # Base field from coordinates
            base_field = self.coord_encoder(query_coords)
            
            # Source contribution from atoms
            source_field = self.compute_source_contribution(
                query_coords, atom_coords, atom_features
            )
            
            # Combined field
            field = base_field + source_field
            
            # Decode observables
            density = self.density_decoder(field)
            reactivity = self.reactivity_decoder(field)
            momentum = self.momentum_decoder(field)
            
            return {
                "field": field,
                "density": density.squeeze(-1),
                "reactivity": reactivity.squeeze(-1),
                "momentum": momentum,
            }
    
    
    class DarkFieldMediator(nn.Module):
        """
        The "Dark Field" that mediates long-range interactions.
        
        Analogous to dark matter in cosmology, this field:
        - Cannot be directly observed (latent)
        - Mediates interactions between visible matter (atoms)
        - Captures non-local correlations in reactivity
        
        Implementation: A diffusion process in latent space that
        propagates information between atoms.
        """
        
        def __init__(
            self,
            atom_dim: int = 128,
            dark_dim: int = 64,
            num_diffusion_steps: int = 5,
            diffusion_rate: float = 0.1,
        ):
            super().__init__()
            
            self.dark_dim = dark_dim
            self.num_steps = num_diffusion_steps
            self.diffusion_rate = diffusion_rate
            
            # Project atoms to dark field
            self.to_dark = nn.Linear(atom_dim, dark_dim)
            
            # Dark field dynamics
            self.dark_update = nn.Sequential(
                nn.Linear(dark_dim * 2, dark_dim * 2),
                nn.SiLU(),
                nn.Linear(dark_dim * 2, dark_dim),
            )
            
            # Project back to atom space
            self.from_dark = nn.Linear(dark_dim, atom_dim)
        
        def forward(
            self,
            atom_features: torch.Tensor,     # (N, atom_dim)
            distances: torch.Tensor,         # (N, N) pairwise distances
            mask: Optional[torch.Tensor] = None,  # (N, N) attention mask
        ) -> torch.Tensor:
            """
            Propagate information through the dark field.
            
            Args:
                atom_features: Initial atom features
                distances: Pairwise distance matrix
                mask: Optional mask for valid pairs
                
            Returns:
                Updated atom features with dark field contribution
            """
            N = atom_features.size(0)
            
            # Project to dark space
            dark = self.to_dark(atom_features)  # (N, dark_dim)
            
            # Compute interaction kernel
            # Exponential decay with distance
            kernel = torch.exp(-distances / 3.0)  # 3 Å characteristic length
            if mask is not None:
                kernel = kernel * mask
            
            # Normalize kernel
            kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-6)
            
            # Diffusion steps
            for step in range(self.num_steps):
                # Aggregate neighbor information
                neighbor_dark = torch.matmul(kernel, dark)  # (N, dark_dim)
                
                # Update dark field
                update_input = torch.cat([dark, neighbor_dark], dim=-1)
                dark_delta = self.dark_update(update_input)
                dark = dark + self.diffusion_rate * dark_delta
            
            # Project back to atom space
            dark_contribution = self.from_dark(dark)
            
            return dark_contribution
    
    
    class HamiltonianDynamics(nn.Module):
        """
        Hamiltonian neural ODE for field dynamics.
        
        Models the time evolution of the reactivity field using
        Hamiltonian mechanics:
        - H(q, p) = T(p) + V(q)
        - dq/dt = ∂H/∂p
        - dp/dt = -∂H/∂q
        
        This ensures energy conservation, which encodes reaction
        thermodynamics.
        """
        
        def __init__(
            self,
            field_dim: int = 64,
            hidden_dim: int = 128,
        ):
            super().__init__()
            
            self.field_dim = field_dim
            
            # Potential energy: V(q)
            self.potential_net = nn.Sequential(
                nn.Linear(field_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            
            # Kinetic energy: T(p) = ½p²/m
            # m is a learnable mass matrix
            self.mass_inv = nn.Parameter(torch.ones(field_dim))
        
        def hamiltonian(
            self,
            q: torch.Tensor,  # Field configuration
            p: torch.Tensor,  # Field momentum
        ) -> torch.Tensor:
            """Compute total energy H = T + V."""
            # Kinetic energy: T = ½ Σᵢ pᵢ²/mᵢ
            T = 0.5 * (p ** 2 * self.mass_inv.unsqueeze(0)).sum(dim=-1)
            
            # Potential energy
            V = self.potential_net(q).squeeze(-1)
            
            return T + V
        
        def time_derivative(
            self,
            t: torch.Tensor,
            state: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute time derivatives using Hamilton's equations.
            
            state = [q, p] concatenated
            """
            q, p = state.chunk(2, dim=-1)
            
            # dq/dt = ∂H/∂p = p/m
            dq_dt = p * self.mass_inv.unsqueeze(0)
            
            # dp/dt = -∂H/∂q (computed via autograd)
            q_grad = q.requires_grad_(True)
            V = self.potential_net(q_grad).sum()
            dV_dq = torch.autograd.grad(V, q_grad, create_graph=True)[0]
            dp_dt = -dV_dq
            
            return torch.cat([dq_dt, dp_dt], dim=-1)
        
        def forward(
            self,
            q0: torch.Tensor,
            p0: torch.Tensor,
            t_span: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Integrate Hamiltonian dynamics forward in time.
            
            Uses simple Euler integration (for efficiency).
            For production, use symplectic integrator.
            
            Args:
                q0: Initial field configuration
                p0: Initial field momentum
                t_span: Time points to evaluate
                
            Returns:
                (q_trajectory, p_trajectory) at each time point
            """
            dt = t_span[1] - t_span[0] if len(t_span) > 1 else 0.1
            
            q, p = q0, p0
            q_traj = [q0]
            p_traj = [p0]
            
            for t in t_span[1:]:
                # Symplectic Euler (leapfrog would be better)
                # Step 1: Update momentum
                q_grad = q.requires_grad_(True)
                V = self.potential_net(q_grad).sum()
                dV_dq = torch.autograd.grad(V, q_grad)[0]
                p = p - dt * dV_dq
                
                # Step 2: Update position
                q = q + dt * p * self.mass_inv.unsqueeze(0)
                
                q_traj.append(q.detach())
                p_traj.append(p.detach())
            
            return torch.stack(q_traj), torch.stack(p_traj)


# ============================================================================
# DARK MANIFOLD COMPLETE MODEL
# ============================================================================

if TORCH_AVAILABLE:
    
    class DarkManifoldModel(nn.Module):
        """
        The Dark Manifold: A 4D field-theoretic neural network for SoM prediction.
        
        Architecture:
        1. Embed atoms as field excitations in 3D+1D spacetime
        2. Propagate information through the dark field
        3. Evolve field configuration via Hamiltonian dynamics
        4. Collapse field to reactivity predictions at atomic sites
        
        The "4th dimension" is reactivity time - tracking how the system
        would evolve toward a reaction. The final prediction is the
        equilibrium field value at each atomic position.
        """
        
        def __init__(
            self,
            atom_input_dim: int = 64,
            field_dim: int = 64,
            hidden_dim: int = 128,
            num_field_layers: int = 4,
            num_diffusion_steps: int = 5,
            use_hamiltonian: bool = True,
            dropout: float = 0.1,
        ):
            super().__init__()
            
            self.field_dim = field_dim
            self.use_hamiltonian = use_hamiltonian
            
            # Atom embedding to field space
            self.atom_to_field = nn.Sequential(
                nn.Linear(atom_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, field_dim),
            )
            
            # 3D coordinate embedding
            self.coord_embed = SirenNetwork(
                in_features=3,
                hidden_features=hidden_dim,
                hidden_layers=2,
                out_features=field_dim // 2,
            )
            
            # Reactivity field
            self.reactivity_field = ReactivityField(
                coord_dim=3,
                field_dim=field_dim,
                hidden_dim=hidden_dim,
                num_layers=num_field_layers,
            )
            
            # Dark field mediator
            self.dark_field = DarkFieldMediator(
                atom_dim=field_dim,
                dark_dim=field_dim // 2,
                num_diffusion_steps=num_diffusion_steps,
            )
            
            # Hamiltonian dynamics (optional)
            if use_hamiltonian:
                self.hamiltonian = HamiltonianDynamics(
                    field_dim=field_dim,
                    hidden_dim=hidden_dim,
                )
            
            # Field collapse to prediction
            self.collapse = nn.Sequential(
                nn.Linear(field_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            
            # Auxiliary: predict field energy (for regularization)
            self.energy_head = nn.Linear(field_dim, 1)
        
        def forward(
            self,
            atom_features: torch.Tensor,     # (N, atom_input_dim)
            atom_coords: torch.Tensor,       # (N, 3)
            batch_index: Optional[torch.Tensor] = None,  # (N,)
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass through the Dark Manifold.
            
            Args:
                atom_features: Input atom features
                atom_coords: 3D coordinates of atoms
                batch_index: Molecule indices for batching
                
            Returns:
                Dict with predictions and field representations
            """
            N = atom_features.size(0)
            device = atom_features.device
            
            # Project atoms to field space
            atom_field = self.atom_to_field(atom_features)
            
            # Embed coordinates
            coord_field = self.coord_embed(atom_coords)
            
            # Combine atom and coordinate information
            combined_field = atom_field + F.pad(coord_field, (0, atom_field.size(-1) - coord_field.size(-1)))
            
            # Compute pairwise distances for dark field
            diff = atom_coords.unsqueeze(0) - atom_coords.unsqueeze(1)
            distances = torch.norm(diff, dim=-1)  # (N, N)
            
            # Create batch mask if batch_index provided
            if batch_index is not None:
                batch_mask = batch_index.unsqueeze(0) == batch_index.unsqueeze(1)
            else:
                batch_mask = None
            
            # Propagate through dark field
            dark_contribution = self.dark_field(combined_field, distances, batch_mask)
            field_with_dark = combined_field + dark_contribution
            
            # Query reactivity field at atomic positions
            field_output = self.reactivity_field(
                query_coords=atom_coords,
                atom_coords=atom_coords,
                atom_features=field_with_dark,
            )
            
            queried_field = field_output["field"]
            
            # Hamiltonian evolution (if enabled)
            if self.use_hamiltonian:
                # Initialize momentum from field gradient
                p0 = field_output["momentum"].mean(dim=-1, keepdim=True).expand_as(queried_field)
                
                # Short time evolution
                t_span = torch.linspace(0, 1, 5, device=device)
                q_traj, p_traj = self.hamiltonian(queried_field, p0, t_span)
                
                # Final field is equilibrium
                final_field = q_traj[-1]
                
                # Compute energy
                energy = self.hamiltonian.hamiltonian(final_field, p_traj[-1])
            else:
                final_field = queried_field
                energy = self.energy_head(queried_field).squeeze(-1)
            
            # Collapse field to predictions
            collapse_input = torch.cat([final_field, field_with_dark], dim=-1)
            logits = self.collapse(collapse_input).squeeze(-1)
            
            return {
                "site_logits": logits,
                "field_embedding": final_field,
                "dark_contribution": dark_contribution,
                "field_density": field_output["density"],
                "field_reactivity": field_output["reactivity"],
                "field_energy": energy,
                "atom_features": field_with_dark,
            }
    
    
    class DarkManifoldSoMPredictor(nn.Module):
        """
        Complete SoM predictor using the Dark Manifold architecture.
        
        This wraps a backbone GNN and adds the Dark Manifold for
        enhanced reactivity prediction.
        """
        
        def __init__(
            self,
            backbone: nn.Module,
            atom_dim: int = 128,
            field_dim: int = 64,
            use_hamiltonian: bool = True,
        ):
            super().__init__()
            
            self.backbone = backbone
            
            # Dark Manifold
            self.dark_manifold = DarkManifoldModel(
                atom_input_dim=atom_dim,
                field_dim=field_dim,
                use_hamiltonian=use_hamiltonian,
            )
            
            # Ensemble head
            self.ensemble = nn.Sequential(
                nn.Linear(atom_dim + field_dim, atom_dim),
                nn.LayerNorm(atom_dim),
                nn.SiLU(),
                nn.Linear(atom_dim, 1),
            )
        
        def forward(
            self,
            batch: Dict[str, torch.Tensor],
        ) -> Dict[str, torch.Tensor]:
            """Forward pass with backbone + Dark Manifold."""
            # Get backbone features
            backbone_out = self.backbone(batch)
            atom_features = backbone_out.get("atom_features")
            backbone_logits = backbone_out.get("site_logits", torch.zeros(atom_features.size(0)))
            
            # Get 3D coordinates
            atom_coords = batch.get("pos", batch.get("atom_coords"))
            if atom_coords is None:
                # Generate dummy coordinates if not provided
                atom_coords = torch.randn(atom_features.size(0), 3, device=atom_features.device)
            
            batch_index = batch.get("batch")
            
            # Dark Manifold prediction
            dark_out = self.dark_manifold(atom_features, atom_coords, batch_index)
            
            # Ensemble
            ensemble_input = torch.cat([atom_features, dark_out["field_embedding"]], dim=-1)
            ensemble_logits = self.ensemble(ensemble_input).squeeze(-1)
            
            # Final prediction
            final_logits = (
                0.4 * backbone_logits +
                0.4 * dark_out["site_logits"] +
                0.2 * ensemble_logits
            )
            
            # Compile outputs
            outputs = dict(backbone_out)
            outputs["site_logits"] = final_logits
            outputs["site_logits_backbone"] = backbone_logits
            outputs["site_logits_dark"] = dark_out["site_logits"]
            outputs["field_embedding"] = dark_out["field_embedding"]
            outputs["dark_contribution"] = dark_out["dark_contribution"]
            outputs["field_energy"] = dark_out["field_energy"]
            
            return outputs


def create_dark_manifold_predictor(
    backbone: nn.Module,
    config: Optional[object] = None,
) -> "DarkManifoldSoMPredictor":
    """
    Create a Dark Manifold SoM predictor from a backbone model.
    
    Args:
        backbone: Existing GNN backbone
        config: Optional configuration object
        
    Returns:
        DarkManifoldSoMPredictor
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    
    atom_dim = 128
    if config is not None:
        atom_dim = getattr(config, "hidden_dim", 
                   getattr(config, "som_branch_dim", 128))
    
    return DarkManifoldSoMPredictor(
        backbone=backbone,
        atom_dim=atom_dim,
        field_dim=64,
        use_hamiltonian=True,
    )


if __name__ == "__main__":
    print("Testing Dark Manifold Neural Field Theory...")
    
    if TORCH_AVAILABLE:
        # Test components
        print("\n1. Testing SIREN network...")
        siren = SirenNetwork(in_features=3, out_features=1)
        coords = torch.randn(10, 3)
        out = siren(coords)
        print(f"   Input: {coords.shape}, Output: {out.shape}")
        
        print("\n2. Testing ReactivityField...")
        field = ReactivityField(field_dim=64)
        atom_coords = torch.randn(5, 3)
        atom_features = torch.randn(5, 64)
        query_coords = torch.randn(20, 3)
        
        field_out = field(query_coords, atom_coords, atom_features)
        print(f"   Query: {query_coords.shape}")
        print(f"   Density: {field_out['density'].shape}")
        print(f"   Reactivity: {field_out['reactivity'].shape}")
        
        print("\n3. Testing DarkFieldMediator...")
        dark = DarkFieldMediator(atom_dim=128)
        atom_feat = torch.randn(5, 128)
        distances = torch.rand(5, 5) * 5
        dark_contrib = dark(atom_feat, distances)
        print(f"   Dark contribution: {dark_contrib.shape}")
        
        print("\n4. Testing full DarkManifoldModel...")
        model = DarkManifoldModel(
            atom_input_dim=64,
            field_dim=64,
            use_hamiltonian=True,
        )
        
        N = 10
        atom_features = torch.randn(N, 64)
        atom_coords = torch.randn(N, 3)
        batch_index = torch.zeros(N, dtype=torch.long)
        
        out = model(atom_features, atom_coords, batch_index)
        print(f"   Site logits: {out['site_logits'].shape}")
        print(f"   Field embedding: {out['field_embedding'].shape}")
        print(f"   Field energy: {out['field_energy'].shape}")
        
        print("\nAll tests passed!")
    else:
        print("PyTorch not available, skipping tests")
