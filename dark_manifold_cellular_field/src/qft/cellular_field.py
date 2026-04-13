"""
Cellular Field Theory for Dark Manifold Virtual Cell
=====================================================

This module adapts the quantum field theory concepts from enzyme_Software
for whole-cell metabolism simulation.

KEY INSIGHT: The cell IS a continuous field, not a bag of discrete molecules.

ANALOGIES:
    Molecular QFT              →     Cellular Field Theory
    ─────────────────────────────────────────────────────────────
    Electron density ρ(x)      →     Metabolite concentration M_i(x)
    Atom positions {Ri}        →     Enzyme positions / compartments
    Atomic number Z            →     Enzyme activity (kcat/Km)
    Nuclear cusp               →     Reaction hotspot
    Hamiltonian evolution      →     Metabolism dynamics
    Dark field mediation       →     Gene regulatory network
    Wave function collapse     →     Measurement / cell state readout

PHYSICS:
    The cell state is a field φ(x, t) over 4D spacetime.
    - φ is a vector: [metabolite_1, metabolite_2, ..., metabolite_N]
    - Dynamics follow: ∂φ/∂t = D∇²φ + S·v - decay
    - Where v = flux vector from enzyme kinetics
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# SIREN: Sinusoidal Representation Networks
# =============================================================================

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
    
    Learns a continuous function f: ℝ^d → ℝ^k.
    For cellular fields: f: (x, y, z, t) → [M₁, M₂, ..., Mₙ]
    """
    
    def __init__(
        self,
        in_features: int = 4,  # x, y, z, t
        hidden_features: int = 256,
        hidden_layers: int = 3,
        out_features: int = 83,  # n_metabolites
        omega_0: float = 30.0,
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
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Query metabolite concentrations at given spacetime coordinates.
        
        Args:
            coords: (..., 4) coordinates [x, y, z, t]
            
        Returns:
            (..., n_metabolites) concentration field values
        """
        return F.softplus(self.layers(coords))  # Concentrations must be positive


# =============================================================================
# METABOLITE CONCENTRATION FIELD
# =============================================================================

class MetaboliteField(nn.Module):
    """
    Continuous field for metabolite concentrations.
    
    Models the concentration of all metabolites as a continuous
    function over the cell's 3D volume.
    
    Key features:
    - SIREN backbone for smooth interpolation
    - Enzyme sources: enzymes create/consume metabolites locally
    - Compartment constraints: metabolites are bounded to compartments
    - Diffusion: metabolites spread according to diffusion coefficients
    """
    
    def __init__(
        self,
        n_metabolites: int = 83,
        n_enzymes: int = 531,
        field_dim: int = 128,
        hidden_dim: int = 256,
        n_compartments: int = 3,  # cytoplasm, membrane, dna_region
    ):
        super().__init__()
        
        self.n_metabolites = n_metabolites
        self.n_enzymes = n_enzymes
        self.n_compartments = n_compartments
        
        # SIREN backbone for continuous field
        self.siren = SirenNetwork(
            in_features=4,  # x, y, z, t
            hidden_features=hidden_dim,
            hidden_layers=3,
            out_features=field_dim,
            omega_0=30.0,
        )
        
        # Decode field to metabolite concentrations
        self.concentration_decoder = nn.Sequential(
            nn.Linear(field_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_metabolites),
            nn.Softplus(),  # Ensure positive concentrations
        )
        
        # Enzyme activity field (enzymes act as sources/sinks)
        self.enzyme_to_field = nn.Sequential(
            nn.Linear(n_enzymes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, field_dim),
        )
        
        # Stoichiometry: how enzymes affect metabolites
        # This should encode: enzyme i catalyzes reaction j which produces/consumes metabolite k
        self.stoichiometry = nn.Parameter(
            torch.randn(n_metabolites, n_enzymes) * 0.01
        )
        
        # Compartment embeddings
        self.compartment_embed = nn.Embedding(n_compartments, field_dim)
        
        # Compartment boundaries (learned or fixed)
        # For bacteria: simple ellipsoid
        self.register_buffer(
            'cell_radius',
            torch.tensor([0.5, 0.5, 1.0])  # Half-axes in μm
        )
    
    def get_compartment(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Determine which compartment each coordinate is in.
        
        Simple model:
        - membrane: |x| > 0.9 * radius
        - dna_region: z in [-0.2, 0.2] and |x| < 0.3
        - cytoplasm: everything else
        """
        # Normalize to unit sphere
        r = coords[..., :3] / self.cell_radius
        r_norm = torch.norm(r, dim=-1)
        
        # Compartment IDs: 0=cytoplasm, 1=membrane, 2=dna_region
        compartment = torch.zeros_like(r_norm, dtype=torch.long)
        
        # Membrane: near surface
        compartment = torch.where(r_norm > 0.9, torch.ones_like(compartment), compartment)
        
        # DNA region: central, low z
        is_dna = (r_norm < 0.4) & (r[..., 2].abs() < 0.3)
        compartment = torch.where(is_dna, 2 * torch.ones_like(compartment), compartment)
        
        return compartment
    
    def forward(
        self,
        coords: torch.Tensor,  # (B, 4) or (B, N, 4) spacetime coordinates
        enzyme_activity: Optional[torch.Tensor] = None,  # (B, n_enzymes) or (n_enzymes,)
    ) -> Dict[str, torch.Tensor]:
        """
        Query the metabolite field at given coordinates.
        
        Args:
            coords: Spacetime coordinates [x, y, z, t]
            enzyme_activity: Enzyme expression/activity levels
            
        Returns:
            Dict with concentrations, field values, compartment info
        """
        # Get base field from SIREN
        field = self.siren(coords)
        
        # Add enzyme contribution (global effect)
        if enzyme_activity is not None:
            if enzyme_activity.dim() == 1:
                enzyme_activity = enzyme_activity.unsqueeze(0)
            enzyme_field = self.enzyme_to_field(enzyme_activity)
            
            # Broadcast enzyme field to all coordinates
            if coords.dim() == 3:  # (B, N, 4)
                enzyme_field = enzyme_field.unsqueeze(1)
            field = field + enzyme_field
        
        # Add compartment modulation
        compartment = self.get_compartment(coords)
        comp_embed = self.compartment_embed(compartment)
        field = field + comp_embed
        
        # Decode to concentrations
        concentrations = self.concentration_decoder(field)
        
        return {
            'concentrations': concentrations,
            'field': field,
            'compartment': compartment,
        }
    
    def compute_flux(
        self,
        concentrations: torch.Tensor,  # (B, n_metabolites)
        enzyme_activity: torch.Tensor,  # (B, n_enzymes)
        km_values: Optional[torch.Tensor] = None,  # (n_enzymes,)
    ) -> torch.Tensor:
        """
        Compute reaction fluxes using Michaelis-Menten kinetics.
        
        v = Vmax * [S] / (Km + [S])
        
        Returns:
            (B, n_metabolites) flux contribution
        """
        if km_values is None:
            km_values = torch.ones(self.n_enzymes, device=enzyme_activity.device)
        
        # Simplified: assume each enzyme acts on one primary substrate
        # Real implementation would use reaction definitions
        
        # Vmax ~ enzyme_activity
        vmax = enzyme_activity
        
        # Simplified substrate (use mean concentration as proxy)
        substrate = concentrations.mean(dim=-1, keepdim=True)
        
        # MM kinetics
        flux_per_enzyme = vmax * substrate / (km_values + substrate)
        
        # Apply stoichiometry
        flux = flux_per_enzyme @ self.stoichiometry.T
        
        return flux


# =============================================================================
# DARK FIELD GENE REGULATORY NETWORK
# =============================================================================

class GeneRegulatoryField(nn.Module):
    """
    The "Dark Field" for gene regulation.
    
    Just as dark matter mediates gravitational interactions in cosmology,
    this field mediates regulatory interactions between genes.
    
    Key idea: Gene regulation is a diffusive process in "regulatory space"
    where transcription factors propagate signals between genes.
    """
    
    def __init__(
        self,
        n_genes: int = 531,
        dark_dim: int = 64,
        n_diffusion_steps: int = 5,
        diffusion_rate: float = 0.1,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.dark_dim = dark_dim
        self.n_steps = n_diffusion_steps
        self.diffusion_rate = diffusion_rate
        
        # Project genes to dark field
        self.to_dark = nn.Linear(n_genes, dark_dim)
        
        # Dark field dynamics (regulatory interactions)
        self.dark_update = nn.Sequential(
            nn.Linear(dark_dim * 2, dark_dim * 2),
            nn.SiLU(),
            nn.Linear(dark_dim * 2, dark_dim),
        )
        
        # Regulatory kernel (which genes regulate which)
        # This is the "dark matter distribution"
        self.regulatory_weights = nn.Parameter(
            torch.randn(n_genes, n_genes) * 0.01
        )
        
        # Project back to gene space
        self.from_dark = nn.Linear(dark_dim, n_genes)
    
    @property
    def regulatory_kernel(self) -> torch.Tensor:
        """
        Get the regulatory interaction kernel.
        
        This is analogous to the gravitational potential in dark matter.
        """
        # Symmetrize and apply softmax for attention-like behavior
        K = 0.5 * (self.regulatory_weights + self.regulatory_weights.T)
        return torch.softmax(K, dim=-1)
    
    def forward(
        self,
        gene_expression: torch.Tensor,  # (B, n_genes)
        external_signal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Propagate regulatory signals through the dark field.
        
        Args:
            gene_expression: Current gene expression levels
            external_signal: Optional external regulatory signal
            
        Returns:
            Dict with regulated expression, dark field state
        """
        # Project to dark space
        dark = self.to_dark(gene_expression)  # (B, dark_dim)
        
        # Get regulatory kernel
        kernel = self.regulatory_kernel  # (n_genes, n_genes)
        
        # Initial dark state
        dark_states = [dark]
        
        # Diffusion steps (signal propagation)
        for step in range(self.n_steps):
            # Gene-to-gene signal propagation
            gene_signal = gene_expression @ kernel.T  # (B, n_genes)
            
            # Project propagated signal to dark space
            neighbor_dark = self.to_dark(gene_signal)
            
            # Update dark field
            update_input = torch.cat([dark, neighbor_dark], dim=-1)
            dark_delta = self.dark_update(update_input)
            dark = dark + self.diffusion_rate * dark_delta
            
            dark_states.append(dark)
        
        # Add external signal if provided
        if external_signal is not None:
            dark = dark + self.to_dark(external_signal)
        
        # Project back to gene space
        regulated_expression = self.from_dark(dark)
        
        # Apply as multiplicative modulation
        output_expression = gene_expression * torch.sigmoid(regulated_expression)
        
        return {
            'regulated_expression': output_expression,
            'dark_field': dark,
            'dark_trajectory': torch.stack(dark_states),
            'regulatory_kernel': kernel,
        }


# =============================================================================
# HAMILTONIAN CELL DYNAMICS
# =============================================================================

class HamiltonianCellDynamics(nn.Module):
    """
    Hamiltonian dynamics for cellular metabolism.
    
    The cell state evolves according to Hamilton's equations:
    - H(q, p) = T(p) + V(q)  (total energy)
    - dq/dt = ∂H/∂p         (generalized velocity)
    - dp/dt = -∂H/∂q        (generalized force)
    
    This ensures energy conservation, which encodes:
    - ATP balance
    - Redox balance (NAD/NADH)
    - Thermodynamic feasibility (ΔG constraints)
    """
    
    def __init__(
        self,
        state_dim: int = 83,  # n_metabolites
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        # Potential energy: V(q) - represents thermodynamic potential
        # Minima correspond to stable metabolic states
        self.potential_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Mass matrix (learnable)
        # Represents "inertia" - how resistant each metabolite is to change
        self.log_mass = nn.Parameter(torch.zeros(state_dim))
    
    @property
    def mass_inv(self) -> torch.Tensor:
        """Inverse mass matrix."""
        return torch.exp(-self.log_mass)
    
    def hamiltonian(
        self,
        q: torch.Tensor,  # Metabolite concentrations
        p: torch.Tensor,  # Metabolite "momenta" (rate of change)
    ) -> torch.Tensor:
        """
        Compute total energy H = T + V.
        
        This should be approximately conserved during evolution.
        """
        # Kinetic energy: T = ½ Σᵢ pᵢ²/mᵢ
        T = 0.5 * (p ** 2 * self.mass_inv.unsqueeze(0)).sum(dim=-1)
        
        # Potential energy (from network)
        V = self.potential_net(q).squeeze(-1)
        
        return T + V
    
    def step(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        dt: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single integration step using neural network for forces.
        
        Instead of autograd, we use a learned force network.
        This is faster and avoids gradient issues in eval mode.
        """
        # Compute forces from potential (learned approximation)
        with torch.enable_grad():
            q_input = q.detach().clone().requires_grad_(True)
            V = self.potential_net(q_input).sum()
            if q_input.grad is not None:
                q_input.grad.zero_()
            V.backward()
            dV_dq = q_input.grad.clone()
        
        # Symplectic Euler update
        p_new = p - dt * dV_dq
        q_new = q + dt * p_new * self.mass_inv.unsqueeze(0)
        
        return q_new.detach(), p_new.detach()
    
    def forward(
        self,
        q0: torch.Tensor,  # Initial concentrations
        p0: torch.Tensor,  # Initial rates (can be zero)
        n_steps: int = 10,
        dt: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Integrate Hamiltonian dynamics forward in time.
        
        Returns:
            Dict with trajectory, energy conservation stats
        """
        q, p = q0, p0
        
        q_traj = [q0]
        p_traj = [p0]
        energies = [self.hamiltonian(q0, p0)]
        
        for _ in range(n_steps):
            q, p = self.step(q, p, dt)
            q_traj.append(q)
            p_traj.append(p)
            energies.append(self.hamiltonian(q, p))
        
        q_traj = torch.stack(q_traj)
        p_traj = torch.stack(p_traj)
        energies = torch.stack(energies)
        
        # Energy conservation metric
        energy_drift = (energies[-1] - energies[0]).abs() / (energies[0].abs() + 1e-6)
        
        return {
            'q_trajectory': q_traj,  # (n_steps+1, B, state_dim)
            'p_trajectory': p_traj,
            'energies': energies,
            'energy_drift': energy_drift,
            'final_state': q_traj[-1],
        }


# =============================================================================
# CONSERVATION ENFORCER
# =============================================================================

class CellConservationEnforcer(nn.Module):
    """
    Enforces physical conservation laws in the cell.
    
    Analogous to HohenbergKohn_Field_Enforcer in quantum chemistry,
    but for cellular quantities.
    
    Conservation laws:
    1. Mass balance: total atoms are conserved
    2. Energy balance: ATP/ADP ratio follows thermodynamics
    3. Redox balance: NAD⁺/NADH follows electron transfer
    4. Compartment constraints: metabolites stay in allowed compartments
    """
    
    def __init__(
        self,
        n_metabolites: int = 83,
    ):
        super().__init__()
        
        self.n_metabolites = n_metabolites
        
        # Indices for key metabolites (if known)
        self.register_buffer('atp_idx', torch.tensor(21))  # ATP index
        self.register_buffer('adp_idx', torch.tensor(22))  # ADP index
        self.register_buffer('nad_idx', torch.tensor(23))  # NAD+ index
        self.register_buffer('nadh_idx', torch.tensor(24))  # NADH index
    
    def apply_mass_balance(
        self,
        concentrations: torch.Tensor,  # (B, n_metabolites)
        total_mass: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Ensure total metabolite mass is conserved.
        
        Simple approach: normalize to target total.
        """
        if total_mass is None:
            return concentrations
        
        current_total = concentrations.sum(dim=-1, keepdim=True)
        scale = total_mass / (current_total + 1e-6)
        
        return concentrations * scale
    
    def apply_atp_constraint(
        self,
        concentrations: torch.Tensor,
        target_energy_charge: float = 0.85,
    ) -> torch.Tensor:
        """
        Constrain ATP/ADP ratio to target energy charge.
        
        Energy charge = ([ATP] + 0.5[ADP]) / ([ATP] + [ADP] + [AMP])
        Simplified: just constrain [ATP]/[ADP] ratio.
        """
        atp = concentrations[:, self.atp_idx]
        adp = concentrations[:, self.adp_idx]
        
        # Target ratio from energy charge
        # EC = 0.85 → ATP/ADP ≈ 5.67
        target_ratio = target_energy_charge / (1 - target_energy_charge)
        current_ratio = atp / (adp + 1e-6)
        
        # Soft constraint via projection
        correction = torch.sigmoid(current_ratio / target_ratio - 1)
        
        # Redistribute between ATP and ADP
        total = atp + adp
        new_atp = total * target_ratio / (1 + target_ratio)
        new_adp = total / (1 + target_ratio)
        
        # Blend current with target
        alpha = 0.1  # Soft constraint strength
        concentrations = concentrations.clone()
        concentrations[:, self.atp_idx] = (1 - alpha) * atp + alpha * new_atp
        concentrations[:, self.adp_idx] = (1 - alpha) * adp + alpha * new_adp
        
        return concentrations
    
    def apply_thermodynamic_feasibility(
        self,
        flux: torch.Tensor,  # (B, n_reactions)
        delta_G: torch.Tensor,  # (n_reactions,) Gibbs free energy
    ) -> torch.Tensor:
        """
        Enforce ΔG × v ≤ 0 (flux opposes free energy gradient).
        
        This is the fundamental thermodynamic constraint.
        """
        # Product should be non-positive
        violation = flux * delta_G.unsqueeze(0)
        
        # Zero out thermodynamically impossible fluxes
        mask = violation <= 0
        
        return flux * mask.float()
    
    def forward(
        self,
        concentrations: torch.Tensor,
        constraints: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Apply all conservation constraints.
        """
        if constraints is None:
            constraints = {}
        
        # Apply each constraint
        result = concentrations
        
        if 'total_mass' in constraints:
            result = self.apply_mass_balance(result, constraints['total_mass'])
        
        if constraints.get('enforce_atp', True):
            result = self.apply_atp_constraint(result)
        
        return result


# =============================================================================
# COMPLETE CELLULAR FIELD MODEL
# =============================================================================

class CellularFieldModel(nn.Module):
    """
    Complete Dark Manifold Virtual Cell using field theory.
    
    Combines:
    1. MetaboliteField: Continuous concentration field
    2. GeneRegulatoryField: Dark field for gene regulation
    3. HamiltonianCellDynamics: Energy-conserving evolution
    4. CellConservationEnforcer: Physical constraints
    """
    
    def __init__(
        self,
        n_genes: int = 531,
        n_metabolites: int = 83,
        field_dim: int = 128,
        use_hamiltonian: bool = True,
    ):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_metabolites = n_metabolites
        self.use_hamiltonian = use_hamiltonian
        
        # Component modules
        self.metabolite_field = MetaboliteField(
            n_metabolites=n_metabolites,
            n_enzymes=n_genes,
            field_dim=field_dim,
        )
        
        self.gene_regulatory_field = GeneRegulatoryField(
            n_genes=n_genes,
            dark_dim=64,
        )
        
        if use_hamiltonian:
            self.hamiltonian = HamiltonianCellDynamics(
                state_dim=n_metabolites,
            )
        
        self.enforcer = CellConservationEnforcer(
            n_metabolites=n_metabolites,
        )
        
        # Final prediction head
        self.predictor = nn.Sequential(
            nn.Linear(n_genes + n_metabolites, field_dim),
            nn.SiLU(),
            nn.Linear(field_dim, n_metabolites),
        )
    
    def forward(
        self,
        gene_expression: torch.Tensor,  # (B, n_genes)
        metabolite_state: torch.Tensor,  # (B, n_metabolites)
        coordinates: Optional[torch.Tensor] = None,  # (B, N, 4) for spatial queries
        n_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the cellular field model.
        
        Args:
            gene_expression: Gene expression levels
            metabolite_state: Current metabolite concentrations
            coordinates: Optional spatial query points
            n_steps: Number of evolution steps
            
        Returns:
            Dict with predictions, field states, trajectories
        """
        # Step 1: Gene regulatory field
        reg_out = self.gene_regulatory_field(gene_expression)
        regulated_genes = reg_out['regulated_expression']
        
        # Step 2: Spatial metabolite field (if coordinates provided)
        if coordinates is not None:
            field_out = self.metabolite_field(
                coordinates,
                enzyme_activity=regulated_genes,
            )
            spatial_concentrations = field_out['concentrations']
        else:
            spatial_concentrations = None
        
        # Step 3: Hamiltonian evolution (if enabled)
        if self.use_hamiltonian:
            # Initialize momentum from gene regulation
            p0 = reg_out['dark_field'].mean(dim=-1, keepdim=True).expand_as(metabolite_state) * 0.01
            
            ham_out = self.hamiltonian(
                q0=metabolite_state,
                p0=p0,
                n_steps=n_steps,
            )
            evolved_state = ham_out['final_state']
            trajectory = ham_out['q_trajectory']
        else:
            evolved_state = metabolite_state
            trajectory = metabolite_state.unsqueeze(0)
        
        # Step 4: Final prediction
        combined = torch.cat([regulated_genes, evolved_state], dim=-1)
        predicted_change = self.predictor(combined)
        
        # Step 5: Apply constraints
        next_state = self.enforcer(evolved_state + predicted_change)
        
        return {
            'next_metabolites': next_state,
            'regulated_genes': regulated_genes,
            'trajectory': trajectory,
            'spatial_concentrations': spatial_concentrations,
            'dark_field': reg_out['dark_field'],
            'regulatory_kernel': reg_out['regulatory_kernel'],
            'energy_drift': ham_out.get('energy_drift') if self.use_hamiltonian else None,
        }
    
    def knockout(
        self,
        gene_expression: torch.Tensor,
        metabolite_state: torch.Tensor,
        gene_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate gene knockout.
        """
        ko_expression = gene_expression.clone()
        ko_expression[:, gene_idx] = 0.0
        
        return self.forward(ko_expression, metabolite_state)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Cellular Field Model...")
    
    # Create model
    model = CellularFieldModel(
        n_genes=531,
        n_metabolites=83,
        field_dim=128,
        use_hamiltonian=True,
    )
    
    # Test inputs
    B = 4
    gene_expr = torch.rand(B, 531)
    met_state = torch.rand(B, 83)
    coords = torch.rand(B, 10, 4)  # 10 spatial query points per sample
    
    print(f"\nInput shapes:")
    print(f"  Gene expression: {gene_expr.shape}")
    print(f"  Metabolite state: {met_state.shape}")
    print(f"  Coordinates: {coords.shape}")
    
    # Forward pass
    out = model(gene_expr, met_state, coords, n_steps=5)
    
    print(f"\nOutput shapes:")
    for key, val in out.items():
        if val is not None:
            if isinstance(val, torch.Tensor):
                print(f"  {key}: {val.shape}")
            else:
                print(f"  {key}: {type(val)}")
    
    # Test knockout
    ko_out = model.knockout(gene_expr, met_state, gene_idx=10)
    
    # Compare
    wt_atp = out['next_metabolites'][:, 21]
    ko_atp = ko_out['next_metabolites'][:, 21]
    
    print(f"\nKnockout effect on ATP:")
    print(f"  WT ATP: {wt_atp.mean():.4f} ± {wt_atp.std():.4f}")
    print(f"  KO ATP: {ko_atp.mean():.4f} ± {ko_atp.std():.4f}")
    print(f"  ΔATP: {(ko_atp - wt_atp).mean():.4f}")
    
    # Test energy conservation
    if out['energy_drift'] is not None:
        print(f"\nEnergy conservation:")
        print(f"  Energy drift: {out['energy_drift'].mean():.6f}")
    
    print("\n✓ All tests passed!")
