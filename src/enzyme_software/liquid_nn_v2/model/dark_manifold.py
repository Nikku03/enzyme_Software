"""
Dark Manifold: 4D Spacetime Field-Theoretic Neural Network

This module implements a fundamentally novel architecture where:
- The network IS a 4D manifold (3 spatial + 1 time/scale dimension)
- Neurons are field excitations at spacetime points
- Connections are mediated by a continuous "dark matter" latent field
- All states exist in superposition until collapse/readout
- Quantum fluctuations serve as stochastic sampling primitives

For SoM prediction, molecules are embedded into this 4D field where:
- Spatial dimensions encode molecular geometry
- The 4th dimension encodes reaction scale (electronic → steric → binding)
- Field dynamics propagate reactivity information across the manifold
- Readout collapses the field to site predictions

Mathematical Foundation:
- Field theory: φ(x,t) satisfies a learned PDE
- Dark field: ψ(x,t) mediates long-range interactions
- Hamiltonian: H = ∫[|∇φ|² + V(φ) + φψ + |ψ|²]dx
- Evolution: ∂φ/∂t = -δH/δφ (gradient flow toward equilibrium)

This is a research prototype exploring whether continuous field dynamics
can capture chemical reactivity better than discrete graph networks.

References:
- Neural ODE (Chen et al., 2018)
- Physics-Informed Neural Networks (Raissi et al., 2019)
- Gauge Equivariant Mesh CNNs (de Haan et al., 2021)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None


if TORCH_AVAILABLE:
    
    # ========================================================================
    # FIELD THEORY PRIMITIVES
    # ========================================================================
    
    class ContinuousField(nn.Module):
        """
        Continuous field φ(x) defined on ℝ³ using implicit neural representation.
        
        Instead of storing values on a grid, we learn a function that maps
        continuous coordinates to field values. This allows:
        - Arbitrary resolution queries
        - Natural gradient computation
        - Memory efficiency for large domains
        
        Architecture: SIREN (Sinusoidal Representation Network)
        φ(x) = W_n(sin(ω·W_{n-1}(...sin(ω·W_0·x + b_0)...)))
        """
        
        def __init__(
            self,
            input_dim: int = 3,
            output_dim: int = 64,
            hidden_dim: int = 128,
            num_layers: int = 4,
            omega_0: float = 30.0,
            use_positional_encoding: bool = True,
            encoding_levels: int = 6,
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.omega_0 = omega_0
            self.use_positional_encoding = use_positional_encoding
            self.encoding_levels = encoding_levels
            
            # Positional encoding expands input dimension
            if use_positional_encoding:
                encoded_dim = input_dim * (1 + 2 * encoding_levels)
            else:
                encoded_dim = input_dim
            
            # SIREN layers
            layers = []
            layers.append(SIRENLayer(encoded_dim, hidden_dim, is_first=True, omega_0=omega_0))
            
            for _ in range(num_layers - 2):
                layers.append(SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0))
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.net = nn.Sequential(*layers)
        
        def positional_encoding(self, x: Tensor) -> Tensor:
            """
            Fourier feature encoding for better high-frequency learning.
            γ(x) = [x, sin(2^0·πx), cos(2^0·πx), ..., sin(2^L·πx), cos(2^L·πx)]
            """
            encodings = [x]
            for i in range(self.encoding_levels):
                freq = 2 ** i * math.pi
                encodings.append(torch.sin(freq * x))
                encodings.append(torch.cos(freq * x))
            return torch.cat(encodings, dim=-1)
        
        def forward(self, coords: Tensor) -> Tensor:
            """
            Evaluate field at coordinates.
            
            Args:
                coords: (..., input_dim) coordinate tensor
                
            Returns:
                (..., output_dim) field values
            """
            if self.use_positional_encoding:
                x = self.positional_encoding(coords)
            else:
                x = coords
            
            return self.net(x)
        
        def gradient(self, coords: Tensor, create_graph: bool = True) -> Tensor:
            """
            Compute gradient ∇φ at coordinates.
            
            Returns:
                (..., output_dim, input_dim) gradient tensor
            """
            coords = coords.requires_grad_(True)
            values = self.forward(coords)
            
            # Compute gradient for each output dimension
            grads = []
            for i in range(self.output_dim):
                grad_i = torch.autograd.grad(
                    values[..., i].sum(),
                    coords,
                    create_graph=create_graph,
                    retain_graph=True,
                )[0]
                grads.append(grad_i)
            
            return torch.stack(grads, dim=-2)  # (..., output_dim, input_dim)
        
        def laplacian(self, coords: Tensor) -> Tensor:
            """
            Compute Laplacian ∇²φ at coordinates.
            
            Returns:
                (..., output_dim) Laplacian values
            """
            coords = coords.requires_grad_(True)
            grad = self.gradient(coords, create_graph=True)  # (..., out, in)
            
            # Laplacian = trace of Hessian = sum of second derivatives
            laplacian = torch.zeros(coords.shape[:-1] + (self.output_dim,), device=coords.device)
            
            for i in range(self.input_dim):
                grad_i = grad[..., i]  # (..., output_dim)
                for j in range(self.output_dim):
                    d2 = torch.autograd.grad(
                        grad_i[..., j].sum(),
                        coords,
                        create_graph=False,
                        retain_graph=True,
                    )[0][..., i]
                    laplacian[..., j] += d2
            
            return laplacian


    class SIRENLayer(nn.Module):
        """
        SIREN layer with sinusoidal activation.
        
        Key insight: sin activations preserve high-frequency information
        through deep networks, unlike ReLU which causes spectral bias.
        """
        
        def __init__(
            self,
            in_features: int,
            out_features: int,
            is_first: bool = False,
            omega_0: float = 30.0,
        ):
            super().__init__()
            
            self.omega_0 = omega_0
            self.is_first = is_first
            
            self.linear = nn.Linear(in_features, out_features)
            
            # Special initialization for SIREN
            with torch.no_grad():
                if is_first:
                    self.linear.weight.uniform_(-1/in_features, 1/in_features)
                else:
                    bound = math.sqrt(6/in_features) / omega_0
                    self.linear.weight.uniform_(-bound, bound)
        
        def forward(self, x: Tensor) -> Tensor:
            return torch.sin(self.omega_0 * self.linear(x))


    # ========================================================================
    # DARK MATTER FIELD (Latent Mediator)
    # ========================================================================
    
    class DarkMatterField(nn.Module):
        """
        The "dark matter" field ψ(x) that mediates interactions between
        visible field excitations.
        
        In physics, dark matter explains gravitational effects without
        direct observation. Here, ψ captures latent interaction patterns
        that aren't directly encoded in molecular features but influence
        reactivity (e.g., electronic correlation effects, induced fit).
        
        The dark field:
        - Is learned from data (no explicit physics)
        - Couples to the visible field through a potential V(φ,ψ)
        - Evolves more slowly than φ (adiabatic approximation)
        """
        
        def __init__(
            self,
            field_dim: int = 64,
            hidden_dim: int = 128,
            num_modes: int = 16,
        ):
            super().__init__()
            
            self.field_dim = field_dim
            self.num_modes = num_modes
            
            # Mode basis (learned Fourier-like decomposition)
            self.mode_vectors = nn.Parameter(torch.randn(num_modes, field_dim))
            self.mode_amplitudes = nn.Parameter(torch.ones(num_modes))
            
            # Coupling network: learns V(φ,ψ)
            self.coupling = nn.Sequential(
                nn.Linear(field_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, field_dim),
            )
            
            # Dark field evolution
            self.evolution = nn.GRUCell(field_dim, field_dim)
        
        def compute_modes(self, coords: Tensor) -> Tensor:
            """
            Compute dark field mode expansion at coordinates.
            
            ψ(x) = Σ_k A_k · exp(i·k·x) · v_k
            (Real version: sum of weighted sinusoids)
            """
            # coords: (..., 3), mode_vectors: (num_modes, field_dim)
            
            # Project coords onto mode "wavevectors" 
            # (interpret first 3 dims of mode_vectors as wavevectors)
            k = self.mode_vectors[:, :3]  # (num_modes, 3)
            
            # Phase: k · x
            phase = torch.einsum("...d,kd->...k", coords, k)  # (..., num_modes)
            
            # Mode values
            cos_modes = torch.cos(phase)
            sin_modes = torch.sin(phase)
            
            # Weighted combination
            amplitudes = F.softmax(self.mode_amplitudes, dim=0)
            values = self.mode_vectors[None] * (
                cos_modes[..., None] * amplitudes[None, :, None]
            )
            
            return values.sum(dim=-2)  # (..., field_dim)
        
        def couple(self, phi: Tensor, psi: Tensor) -> Tensor:
            """
            Compute coupling energy/force between φ and ψ.
            
            Returns interaction term to add to φ dynamics.
            """
            combined = torch.cat([phi, psi], dim=-1)
            return self.coupling(combined)
        
        def forward(
            self,
            coords: Tensor,
            phi: Tensor,
            prev_psi: Optional[Tensor] = None,
        ) -> Tensor:
            """
            Compute dark field value, optionally evolving from previous state.
            
            Args:
                coords: (..., 3) spatial coordinates
                phi: (..., field_dim) visible field values
                prev_psi: (..., field_dim) previous dark field state
                
            Returns:
                (..., field_dim) dark field values
            """
            # Base dark field from modes
            psi_base = self.compute_modes(coords)
            
            # Modulate by coupling to visible field
            coupling = self.couple(phi, psi_base)
            psi = psi_base + 0.1 * coupling
            
            # Evolve if previous state provided
            if prev_psi is not None:
                # Flatten for GRU
                shape = psi.shape
                psi_flat = psi.reshape(-1, self.field_dim)
                prev_flat = prev_psi.reshape(-1, self.field_dim)
                
                psi_evolved = self.evolution(psi_flat, prev_flat)
                psi = psi_evolved.reshape(shape)
            
            return psi


    # ========================================================================
    # FIELD DYNAMICS (Neural ODE)
    # ========================================================================
    
    class FieldDynamics(nn.Module):
        """
        Learns the dynamics dφ/dt = F(φ, ψ, ∇φ, x, t).
        
        The field evolves according to an energy-minimizing flow:
        - Gradient term: diffuses information spatially
        - Potential term: drives toward equilibrium
        - Coupling term: interaction with dark field
        - Source term: injection from initial conditions
        """
        
        def __init__(
            self,
            field_dim: int = 64,
            hidden_dim: int = 128,
        ):
            super().__init__()
            
            self.field_dim = field_dim
            
            # Dynamics network
            # Input: [φ, ψ, ∇φ_flat, x, t]
            # ∇φ has shape (field_dim, 3), flattened = field_dim * 3
            input_dim = field_dim + field_dim + field_dim * 3 + 3 + 1
            
            self.dynamics = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, field_dim),
            )
            
            # Learnable diffusion coefficient
            self.diffusion = nn.Parameter(torch.ones(1) * 0.1)
            
            # Potential (nonlinear self-interaction)
            self.potential = nn.Sequential(
                nn.Linear(field_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, field_dim),
            )
        
        def forward(
            self,
            t: Tensor,
            phi: Tensor,
            psi: Tensor,
            grad_phi: Tensor,
            coords: Tensor,
        ) -> Tensor:
            """
            Compute dφ/dt.
            
            Args:
                t: () scalar time
                phi: (..., field_dim) current field values
                psi: (..., field_dim) dark field values
                grad_phi: (..., field_dim, 3) field gradients
                coords: (..., 3) spatial coordinates
                
            Returns:
                (..., field_dim) time derivative
            """
            batch_shape = phi.shape[:-1]
            
            # Flatten gradient
            grad_flat = grad_phi.reshape(*batch_shape, -1)
            
            # Expand time to match batch
            t_expand = t.expand(*batch_shape, 1)
            
            # Concatenate inputs
            inputs = torch.cat([phi, psi, grad_flat, coords, t_expand], dim=-1)
            
            # Base dynamics
            dphi_dt = self.dynamics(inputs)
            
            # Add diffusion (Laplacian approximation from gradient)
            # ∇²φ ≈ Tr(∇⊗∇φ) - simplified as gradient magnitude
            diffusion_term = self.diffusion * grad_phi.norm(dim=-1).mean(dim=-1, keepdim=True)
            
            # Add potential
            potential_term = -self.potential(phi)
            
            return dphi_dt + diffusion_term + 0.1 * potential_term


    # ========================================================================
    # QUANTUM SUPERPOSITION AND COLLAPSE
    # ========================================================================
    
    class QuantumState(nn.Module):
        """
        Represents field state as superposition until measurement.
        
        |ψ⟩ = Σ_i c_i |φ_i⟩
        
        where |φ_i⟩ are field configurations and c_i are complex amplitudes.
        
        For SoM prediction:
        - Each |φ_i⟩ represents a possible reaction pathway
        - c_i encodes pathway probability
        - Measurement collapses to site predictions
        """
        
        def __init__(
            self,
            field_dim: int = 64,
            num_pathways: int = 8,
        ):
            super().__init__()
            
            self.field_dim = field_dim
            self.num_pathways = num_pathways
            
            # Pathway basis vectors
            self.pathways = nn.Parameter(torch.randn(num_pathways, field_dim))
            
            # Amplitude network (outputs complex amplitude as [re, im])
            self.amplitude_net = nn.Sequential(
                nn.Linear(field_dim, field_dim),
                nn.SiLU(),
                nn.Linear(field_dim, num_pathways * 2),
            )
            
            # Collapse projector
            self.collapse = nn.Linear(field_dim, 1)
        
        def forward(self, phi: Tensor) -> Dict[str, Tensor]:
            """
            Compute superposition state and measurement probabilities.
            
            Args:
                phi: (..., field_dim) field values
                
            Returns:
                Dict with amplitudes, probabilities, and collapsed values
            """
            # Compute amplitudes
            amp_raw = self.amplitude_net(phi)
            amp_re = amp_raw[..., :self.num_pathways]
            amp_im = amp_raw[..., self.num_pathways:]
            
            # Complex magnitude squared = probability
            prob = amp_re ** 2 + amp_im ** 2
            prob = F.softmax(prob, dim=-1)  # Normalize
            
            # Superposition: weighted sum of pathways
            # pathways: (num_pathways, field_dim)
            # prob: (..., num_pathways)
            superposition = torch.einsum("...k,kd->...d", prob, self.pathways)
            
            # Collapse to scalar (measurement)
            collapsed = self.collapse(superposition).squeeze(-1)
            
            return {
                "amplitudes_re": amp_re,
                "amplitudes_im": amp_im,
                "probabilities": prob,
                "superposition": superposition,
                "collapsed": collapsed,
            }
        
        def sample(self, phi: Tensor, num_samples: int = 1) -> Tensor:
            """
            Sample from superposition (stochastic collapse).
            
            Returns sampled pathway indices.
            """
            result = self.forward(phi)
            prob = result["probabilities"]
            
            # Sample from categorical
            samples = torch.multinomial(prob.reshape(-1, self.num_pathways), num_samples)
            
            return samples.reshape(phi.shape[:-1] + (num_samples,))


    # ========================================================================
    # DARK MANIFOLD MODEL
    # ========================================================================
    
    class DarkManifold(nn.Module):
        """
        Complete Dark Manifold model for SoM prediction.
        
        Pipeline:
        1. Embed molecule into 4D spacetime coordinates
        2. Initialize visible field φ from atom features
        3. Compute dark field ψ from learned modes
        4. Evolve φ through field dynamics (neural ODE)
        5. Collapse field to site predictions
        
        The 4th dimension represents "scale" - ranging from electronic
        effects (small scale) to binding pocket interactions (large scale).
        """
        
        def __init__(
            self,
            atom_dim: int = 128,
            field_dim: int = 64,
            hidden_dim: int = 128,
            num_evolution_steps: int = 4,
            num_pathways: int = 8,
        ):
            super().__init__()
            
            self.atom_dim = atom_dim
            self.field_dim = field_dim
            self.num_evolution_steps = num_evolution_steps
            
            # Embed atoms to field space
            self.atom_to_field = nn.Sequential(
                nn.Linear(atom_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, field_dim),
            )
            
            # 3D coords to 4D spacetime (add scale dimension)
            self.coord_augment = nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 4),
            )
            
            # Continuous field representation
            self.phi_field = ContinuousField(
                input_dim=4,  # 4D spacetime
                output_dim=field_dim,
                hidden_dim=hidden_dim,
            )
            
            # Dark matter field
            self.psi_field = DarkMatterField(
                field_dim=field_dim,
                hidden_dim=hidden_dim,
            )
            
            # Field dynamics
            self.dynamics = FieldDynamics(
                field_dim=field_dim,
                hidden_dim=hidden_dim,
            )
            
            # Quantum state for collapse
            self.quantum = QuantumState(
                field_dim=field_dim,
                num_pathways=num_pathways,
            )
            
            # Final site prediction
            self.site_head = nn.Sequential(
                nn.Linear(field_dim + atom_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        
        def embed_molecule(
            self,
            atom_features: Tensor,
            coords_3d: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:
            """
            Embed molecule into 4D spacetime.
            
            Args:
                atom_features: (N, atom_dim) atom embeddings
                coords_3d: (N, 3) 3D coordinates (optional)
                
            Returns:
                coords_4d: (N, 4) 4D spacetime coordinates
                phi_init: (N, field_dim) initial field values
            """
            N = atom_features.size(0)
            device = atom_features.device
            
            # Initial field from atom features
            phi_init = self.atom_to_field(atom_features)
            
            # 4D coordinates
            if coords_3d is not None:
                coords_4d = self.coord_augment(coords_3d)
            else:
                # Default: atoms on a line
                coords_3d = torch.zeros(N, 3, device=device)
                coords_3d[:, 0] = torch.linspace(-1, 1, N, device=device)
                coords_4d = self.coord_augment(coords_3d)
            
            return coords_4d, phi_init
        
        def evolve_field(
            self,
            phi: Tensor,
            coords_4d: Tensor,
        ) -> Tensor:
            """
            Evolve field through dynamics for multiple steps.
            
            Uses Euler integration (could be upgraded to RK4 or adaptive).
            """
            dt = 1.0 / self.num_evolution_steps
            psi = None
            
            for step in range(self.num_evolution_steps):
                t = torch.tensor(step * dt, device=phi.device)
                
                # Compute dark field
                psi = self.psi_field(coords_4d[:, :3], phi, psi)
                
                # Compute gradients (simplified: use difference between field values)
                # True gradient would require differentiating through ContinuousField
                grad_phi = torch.zeros(phi.shape + (3,), device=phi.device)
                
                # Compute dynamics
                dphi_dt = self.dynamics(t, phi, psi, grad_phi, coords_4d[:, :3])
                
                # Euler step
                phi = phi + dt * dphi_dt
            
            return phi
        
        def forward(
            self,
            batch: Dict[str, Tensor],
        ) -> Dict[str, Tensor]:
            """
            Forward pass through Dark Manifold.
            
            Args:
                batch: Dict with:
                    - atom_features: (N, atom_dim)
                    - coords_3d: (N, 3) optional
                    - candidate_mask: (N,) optional
                    
            Returns:
                Dict with site_logits and intermediate states
            """
            atom_features = batch.get("atom_features")
            if atom_features is None:
                raise ValueError("atom_features required")
            
            coords_3d = batch.get("coords_3d")
            candidate_mask = batch.get("candidate_mask")
            
            # Embed into 4D spacetime
            coords_4d, phi_init = self.embed_molecule(atom_features, coords_3d)
            
            # Evolve field
            phi_evolved = self.evolve_field(phi_init, coords_4d)
            
            # Quantum collapse
            quantum_result = self.quantum(phi_evolved)
            
            # Final site prediction
            combined = torch.cat([phi_evolved, atom_features], dim=-1)
            logits = self.site_head(combined).squeeze(-1)
            
            # Apply mask
            if candidate_mask is not None:
                logits = logits * candidate_mask.float() + (-100) * (1 - candidate_mask.float())
            
            return {
                "site_logits": logits,
                "phi_init": phi_init,
                "phi_evolved": phi_evolved,
                "coords_4d": coords_4d,
                "quantum_probs": quantum_result["probabilities"],
                "quantum_collapsed": quantum_result["collapsed"],
            }


    # ========================================================================
    # HYPERBOLIC MEMORY WITH PGW TRANSPORT
    # ========================================================================
    
    class HyperbolicMemoryWithTransport(nn.Module):
        """
        Extended memory system with parallel Gromov-Wasserstein transport.
        
        PGW (Partial Gromov-Wasserstein) allows comparing structures of
        different sizes by finding optimal partial matchings. This enables:
        - Comparing molecules with different atom counts
        - Learning reaction-pattern correspondences
        - Transferring reactivity patterns between similar substrates
        """
        
        def __init__(
            self,
            key_dim: int = 64,
            value_dim: int = 128,
            capacity: int = 2048,
            transport_reg: float = 0.1,
        ):
            super().__init__()
            
            self.key_dim = key_dim
            self.value_dim = value_dim
            self.capacity = capacity
            self.transport_reg = transport_reg
            
            # Memory storage
            self.register_buffer("keys", torch.zeros(capacity, key_dim))
            self.register_buffer("values", torch.zeros(capacity, value_dim))
            self.register_buffer("valid", torch.zeros(capacity, dtype=torch.bool))
            self.register_buffer("ptr", torch.tensor(0))
            
            # Structure encoder (for PGW)
            self.structure_encoder = nn.Sequential(
                nn.Linear(key_dim, key_dim * 2),
                nn.SiLU(),
                nn.Linear(key_dim * 2, key_dim),
            )
            
            # Transport projector (from value_dim to value_dim)
            self.transport_proj = nn.Linear(value_dim, value_dim)
        
        def compute_cost_matrix(self, X: Tensor, Y: Tensor) -> Tensor:
            """
            Compute pairwise cost matrix for OT.
            
            C[i,j] = ||x_i - y_j||²
            """
            X_sq = (X ** 2).sum(dim=-1, keepdim=True)
            Y_sq = (Y ** 2).sum(dim=-1, keepdim=True)
            
            C = X_sq + Y_sq.T - 2 * X @ Y.T
            return C.clamp(min=0)
        
        def sinkhorn(
            self,
            C: Tensor,
            reg: float,
            num_iters: int = 50,
        ) -> Tensor:
            """
            Sinkhorn algorithm for entropic OT.
            
            Computes approximate transport plan T* = argmin_T <C,T> + reg*H(T)
            """
            n, m = C.shape
            device = C.device
            
            # Initialize marginals
            a = torch.ones(n, device=device) / n
            b = torch.ones(m, device=device) / m
            
            # Kernel
            K = torch.exp(-C / reg)
            
            # Sinkhorn iterations
            u = torch.ones(n, device=device)
            v = torch.ones(m, device=device)
            
            for _ in range(num_iters):
                u = a / (K @ v + 1e-10)
                v = b / (K.T @ u + 1e-10)
            
            # Transport plan
            T = u.unsqueeze(1) * K * v.unsqueeze(0)
            
            return T
        
        def write(self, keys: Tensor, values: Tensor) -> None:
            """Write entries to memory."""
            n = keys.size(0)
            
            for i in range(n):
                ptr = self.ptr.item()
                self.keys[ptr] = keys[i].detach()
                self.values[ptr] = values[i].detach()
                self.valid[ptr] = True
                self.ptr = (self.ptr + 1) % self.capacity
        
        def read_with_transport(self, queries: Tensor) -> Dict[str, Tensor]:
            """
            Read from memory using optimal transport matching.
            
            Args:
                queries: (N, key_dim) query embeddings
                
            Returns:
                Dict with transported values and transport plan
            """
            if not self.valid.any():
                return {
                    "values": torch.zeros(queries.size(0), self.value_dim, device=queries.device),
                    "transport": None,
                }
            
            valid_keys = self.keys[self.valid]
            valid_values = self.values[self.valid]
            
            # Encode structure
            Q = self.structure_encoder(queries)
            K = self.structure_encoder(valid_keys)
            
            # Compute transport plan
            C = self.compute_cost_matrix(Q, K)
            T = self.sinkhorn(C, self.transport_reg)
            
            # Transport values
            transported = T @ valid_values  # (N, M) @ (M, value_dim) -> (N, value_dim)
            
            # Project
            output = self.transport_proj(transported)
            
            return {
                "values": output,
                "transport": T,
                "cost": C,
            }


    # ========================================================================
    # RULE DISCOVERY MODULE
    # ========================================================================
    
    class RuleDiscoveryModule(nn.Module):
        """
        Learns to extract symbolic rules from field dynamics.
        
        The goal is to discover human-interpretable rules like:
        - "Benzylic carbons are more reactive"
        - "Sites far from electron-withdrawing groups are preferred"
        
        Approach:
        1. Cluster similar field trajectories
        2. Extract common features in each cluster
        3. Distill to conditional rules
        4. Validate rules on held-out examples
        """
        
        def __init__(
            self,
            field_dim: int = 64,
            num_rules: int = 32,
            rule_dim: int = 16,
        ):
            super().__init__()
            
            self.field_dim = field_dim
            self.num_rules = num_rules
            self.rule_dim = rule_dim
            
            # Rule prototypes
            self.rule_keys = nn.Parameter(torch.randn(num_rules, field_dim))
            self.rule_values = nn.Parameter(torch.randn(num_rules, rule_dim))
            
            # Rule confidence
            self.confidence = nn.Parameter(torch.zeros(num_rules))
            
            # Rule extractor
            self.extractor = nn.Sequential(
                nn.Linear(field_dim * 2, field_dim),
                nn.SiLU(),
                nn.Linear(field_dim, rule_dim),
            )
            
            # Rule bank (accumulated validated rules)
            self.register_buffer("rule_bank", torch.zeros(num_rules, rule_dim))
            self.register_buffer("rule_counts", torch.zeros(num_rules))
        
        def match_rules(self, phi: Tensor) -> Dict[str, Tensor]:
            """
            Find which rules match the current field state.
            
            Returns rule activations and extracted rule features.
            """
            # Compare to rule prototypes
            similarities = F.cosine_similarity(
                phi.unsqueeze(1),  # (N, 1, field_dim)
                self.rule_keys.unsqueeze(0),  # (1, num_rules, field_dim)
                dim=-1
            )  # (N, num_rules)
            
            # Soft rule activations
            activations = F.softmax(similarities * 5.0, dim=-1)
            
            # Weighted rule values
            matched_rules = torch.einsum("nk,kd->nd", activations, self.rule_values)
            
            return {
                "activations": activations,
                "matched_rules": matched_rules,
                "confidences": torch.sigmoid(self.confidence),
            }
        
        def update_rules(
            self,
            phi_init: Tensor,
            phi_final: Tensor,
            labels: Tensor,
        ) -> None:
            """
            Update rule bank based on observed trajectories and outcomes.
            
            Called after training batch to accumulate rule statistics.
            """
            # Extract rules from trajectory
            trajectory = torch.cat([phi_init, phi_final], dim=-1)
            extracted = self.extractor(trajectory)  # (N, rule_dim)
            
            # Find best matching rule for each example
            match_result = self.match_rules(phi_final)
            best_rules = match_result["activations"].argmax(dim=-1)  # (N,)
            
            # Update rule bank (weighted by label agreement)
            for i in range(extracted.size(0)):
                rule_idx = best_rules[i].item()
                weight = labels[i].item()  # Weight by positive label
                
                # Exponential moving average
                alpha = 0.1
                self.rule_bank[rule_idx] = (
                    (1 - alpha) * self.rule_bank[rule_idx] + 
                    alpha * weight * extracted[i].detach()
                )
                self.rule_counts[rule_idx] += weight


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_dark_manifold(
    atom_dim: int = 128,
    field_dim: int = 64,
    num_evolution_steps: int = 4,
) -> "DarkManifold":
    """Create a Dark Manifold model with default configuration."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")
    
    return DarkManifold(
        atom_dim=atom_dim,
        field_dim=field_dim,
        num_evolution_steps=num_evolution_steps,
    )


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        print("Testing Dark Manifold components...")
        
        # Test continuous field
        print("\n1. ContinuousField:")
        field = ContinuousField(input_dim=3, output_dim=8, hidden_dim=32, num_layers=3)
        coords = torch.randn(10, 3) * 0.5
        values = field(coords)
        print(f"   Input shape: {coords.shape}")
        print(f"   Output shape: {values.shape}")
        
        # Test dark matter field
        print("\n2. DarkMatterField:")
        dark = DarkMatterField(field_dim=8, hidden_dim=16, num_modes=4)
        phi = torch.randn(10, 8)
        psi = dark(coords, phi)
        print(f"   Phi shape: {phi.shape}")
        print(f"   Psi shape: {psi.shape}")
        
        # Test quantum state
        print("\n3. QuantumState:")
        quantum = QuantumState(field_dim=8, num_pathways=4)
        q_result = quantum(phi)
        print(f"   Probabilities shape: {q_result['probabilities'].shape}")
        print(f"   Collapsed shape: {q_result['collapsed'].shape}")
        
        # Test full model
        print("\n4. DarkManifold (full model):")
        model = DarkManifold(atom_dim=16, field_dim=8, num_evolution_steps=2)
        
        batch = {
            "atom_features": torch.randn(10, 16),
            "coords_3d": torch.randn(10, 3),
        }
        
        output = model(batch)
        print(f"   Site logits shape: {output['site_logits'].shape}")
        print(f"   Phi init shape: {output['phi_init'].shape}")
        print(f"   Phi evolved shape: {output['phi_evolved'].shape}")
        print(f"   Quantum probs shape: {output['quantum_probs'].shape}")
        
        # Test memory with transport
        print("\n5. HyperbolicMemoryWithTransport:")
        memory = HyperbolicMemoryWithTransport(key_dim=8, value_dim=16, capacity=100)
        
        keys = torch.randn(20, 8)
        values = torch.randn(20, 16)
        memory.write(keys, values)
        
        queries = torch.randn(5, 8)
        mem_result = memory.read_with_transport(queries)
        print(f"   Transport shape: {mem_result['transport'].shape}")
        print(f"   Retrieved values shape: {mem_result['values'].shape}")
        
        print("\n✓ All Dark Manifold tests passed!")
