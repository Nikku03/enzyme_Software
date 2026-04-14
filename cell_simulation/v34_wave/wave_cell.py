"""
Wave Cell Simulator V34
=======================

Integrates all wave mechanics components into a unified cell simulator:
- ModalDynamicsEngine: O(K) metabolic dynamics in mode space
- CellularSIRENField: Continuous concentration field
- GeneRegulatoryPropagator: Non-local gene regulation via Green's function

This is the main entry point for Dark Manifold V34.

Target capabilities:
- 2000-5000x speedup over V33 ODE integration
- Same biological accuracy (validated against V33)
- Gene knockouts via Green's function propagation
- Continuous spatial representation

Author: Naresh Chhillar, 2026
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
import warnings

from .modal_dynamics import ModalDynamicsEngine, MetabolicNetwork, ModalBasis
from .siren_field import CellularSIRENField, create_syn3a_field
from .greens_propagator import GeneRegulatoryPropagator, GeneNetwork, create_minimal_network


@dataclass
class SimulationState:
    """Current state of the simulation."""
    
    time: float
    concentrations: np.ndarray      # [n_metabolites]
    modal_amplitudes: np.ndarray    # [n_modes]
    gene_expression: np.ndarray     # [n_genes]
    enzyme_levels: np.ndarray       # [n_reactions]
    
    # Diagnostics
    energy: float = 0.0             # Total "metabolic energy"
    growth_rate: float = 0.0        # Current growth rate
    
    def copy(self) -> "SimulationState":
        return SimulationState(
            time=self.time,
            concentrations=self.concentrations.copy(),
            modal_amplitudes=self.modal_amplitudes.copy(),
            gene_expression=self.gene_expression.copy(),
            enzyme_levels=self.enzyme_levels.copy(),
            energy=self.energy,
            growth_rate=self.growth_rate,
        )


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    
    times: np.ndarray
    concentrations: np.ndarray      # [n_times, n_metabolites]
    gene_expression: np.ndarray     # [n_times, n_genes]
    growth_rates: np.ndarray        # [n_times]
    
    # Performance metrics
    wall_time: float
    n_steps: int
    speedup_estimate: float         # vs ODE baseline
    
    # Final state
    final_state: SimulationState


class WaveCellSimulator:
    """
    Wave-based whole-cell simulator.
    
    Combines:
    1. Modal dynamics for fast metabolic simulation
    2. Green's function for gene regulation
    3. SIREN field for spatial concentration (optional)
    
    Usage:
        sim = WaveCellSimulator.from_imb155_data(...)
        result = sim.simulate(duration=120, dt=0.1)
    """
    
    def __init__(
        self,
        metabolic_network: MetabolicNetwork,
        gene_network: Optional[GeneNetwork] = None,
        use_siren_field: bool = False,
        n_modes: Optional[int] = None,
        device: str = "cpu",
    ):
        """
        Initialize wave cell simulator.
        
        Args:
            metabolic_network: Metabolic network definition
            gene_network: Gene regulatory network (optional)
            use_siren_field: Use SIREN for spatial field (slower but richer)
            n_modes: Number of modes for modal dynamics (auto if None)
            device: Compute device for SIREN
        """
        self.metabolic_network = metabolic_network
        self.gene_network = gene_network
        self.use_siren_field = use_siren_field
        self.device = device
        
        # Initialize modal dynamics engine
        self.modal_engine = ModalDynamicsEngine(
            network=metabolic_network,
            n_modes=n_modes,
        )
        
        # Initialize gene regulatory propagator
        if gene_network is not None:
            self.gene_propagator = GeneRegulatoryPropagator(gene_network)
        else:
            self.gene_propagator = None
        
        # Initialize SIREN field (optional)
        if use_siren_field:
            self.siren_field = create_syn3a_field(
                n_metabolites=metabolic_network.n_met,
            )
        else:
            self.siren_field = None
        
        # Current state
        self.state: Optional[SimulationState] = None
        
        # External boundary conditions
        self.external_concentrations: Dict[int, float] = {}
        
        # Knockout genes (set to 0 expression)
        self.knockouts: List[int] = []
    
    def initialize(
        self,
        initial_concentrations: np.ndarray,
        initial_gene_expression: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize simulation state.
        
        Args:
            initial_concentrations: [n_metabolites] initial concentrations
            initial_gene_expression: [n_genes] initial expression (optional)
        """
        n_met = self.metabolic_network.n_met
        n_rxn = self.metabolic_network.n_rxn
        n_genes = self.gene_network.n_genes if self.gene_network else 0
        
        # Validate
        if len(initial_concentrations) != n_met:
            raise ValueError(f"Expected {n_met} concentrations, got {len(initial_concentrations)}")
        
        # Initialize modal engine
        self.modal_engine.initialize(initial_concentrations)
        
        # Gene expression
        if initial_gene_expression is not None:
            gene_expr = initial_gene_expression.copy()
        elif n_genes > 0:
            gene_expr = np.ones(n_genes)  # Default: all expressed
        else:
            gene_expr = np.array([])
        
        # Apply knockouts
        for ko in self.knockouts:
            if ko < len(gene_expr):
                gene_expr[ko] = 0.0
        
        # Enzyme levels from gene expression
        enzyme_levels = self._compute_enzyme_levels(gene_expr)
        
        # Create state
        self.state = SimulationState(
            time=0.0,
            concentrations=initial_concentrations.copy(),
            modal_amplitudes=self.modal_engine.get_mode_amplitudes(),
            gene_expression=gene_expr,
            enzyme_levels=enzyme_levels,
        )
        
        # Initialize SIREN field if used
        if self.siren_field is not None:
            self.siren_field.set_reference_concentrations(initial_concentrations)
    
    def _compute_enzyme_levels(self, gene_expression: np.ndarray) -> np.ndarray:
        """Compute enzyme levels from gene expression."""
        n_rxn = self.metabolic_network.n_rxn
        
        if self.gene_network is None or len(gene_expression) == 0:
            return np.ones(n_rxn)
        
        enzyme_levels = np.ones(n_rxn)
        
        # For each reaction, enzyme level = product of associated gene expressions
        # (AND logic for multi-gene enzymes)
        for gene_idx, (gene_name, rxns) in enumerate(self.gene_network.gene_reaction_map.items()):
            if gene_idx < len(gene_expression):
                expr = gene_expression[gene_idx]
                for rxn_idx in rxns:
                    if rxn_idx < n_rxn:
                        enzyme_levels[rxn_idx] *= expr
        
        return enzyme_levels
    
    def knockout(self, gene_indices: Union[int, List[int]]) -> None:
        """
        Knock out one or more genes.
        
        Args:
            gene_indices: Gene index or list of indices to knock out
        """
        if isinstance(gene_indices, int):
            gene_indices = [gene_indices]
        
        self.knockouts.extend(gene_indices)
        
        # If already initialized, update state
        if self.state is not None:
            for ko in gene_indices:
                if ko < len(self.state.gene_expression):
                    self.state.gene_expression[ko] = 0.0
            
            # Recompute enzyme levels
            self.state.enzyme_levels = self._compute_enzyme_levels(self.state.gene_expression)
            
            # If we have gene propagator, compute knockout effect
            if self.gene_propagator is not None:
                for ko in gene_indices:
                    effect = self.gene_propagator.knockout_effect(ko)
                    # Apply effect to gene expression (with damping)
                    self.state.gene_expression += 0.5 * effect
                    self.state.gene_expression = np.maximum(self.state.gene_expression, 0)
    
    def set_external(self, metabolite_idx: int, concentration: float) -> None:
        """Set external/boundary concentration for a metabolite."""
        self.external_concentrations[metabolite_idx] = concentration
    
    def step(self, dt: float) -> SimulationState:
        """
        Advance simulation by one time step.
        
        Args:
            dt: Time step
            
        Returns:
            Updated state
        """
        if self.state is None:
            raise RuntimeError("Call initialize() first")
        
        # Update gene expression (slow timescale)
        if self.gene_propagator is not None:
            # Gene regulation feedback
            # Expression evolves based on current metabolite levels
            metabolite_signal = self._metabolite_to_gene_signal()
            steady_expr = self.gene_propagator.steady_state_expression(metabolite_signal)
            
            # Exponential relaxation towards steady state
            tau_gene = 10.0  # Gene expression timescale
            self.state.gene_expression += dt / tau_gene * (steady_expr - self.state.gene_expression)
            self.state.gene_expression = np.maximum(self.state.gene_expression, 0)
            
            # Apply knockouts
            for ko in self.knockouts:
                if ko < len(self.state.gene_expression):
                    self.state.gene_expression[ko] = 0.0
        
        # Update enzyme levels
        self.state.enzyme_levels = self._compute_enzyme_levels(self.state.gene_expression)
        
        # Metabolic dynamics via modal engine
        new_conc = self.modal_engine.step(
            dt=dt,
            enzyme_levels=self.state.enzyme_levels,
            external_concentrations=self.external_concentrations,
        )
        
        # Update state
        self.state.time += dt
        self.state.concentrations = new_conc
        self.state.modal_amplitudes = self.modal_engine.get_mode_amplitudes()
        
        # Compute diagnostics
        self.state.energy = self._compute_energy()
        self.state.growth_rate = self._compute_growth_rate()
        
        # Update SIREN field if used
        if self.siren_field is not None:
            self.siren_field.update_time(dt)
        
        return self.state.copy()
    
    def _metabolite_to_gene_signal(self) -> np.ndarray:
        """Convert metabolite concentrations to gene regulatory signals."""
        n_genes = self.gene_network.n_genes if self.gene_network else 0
        if n_genes == 0:
            return np.array([])
        
        # Simple model: sum of associated metabolite concentrations
        signals = np.zeros(n_genes)
        
        # Use ATP, GTP as global signals
        # This is a simplification - real cells have complex signaling
        atp_idx = self._find_metabolite("atp")
        gtp_idx = self._find_metabolite("gtp")
        
        if atp_idx >= 0:
            signals += 0.1 * self.state.concentrations[atp_idx]
        if gtp_idx >= 0:
            signals += 0.1 * self.state.concentrations[gtp_idx]
        
        return signals
    
    def _find_metabolite(self, name: str) -> int:
        """Find metabolite index by name (partial match)."""
        name_lower = name.lower()
        for i, m in enumerate(self.metabolic_network.metabolites):
            if name_lower in m.lower():
                return i
        return -1
    
    def _compute_energy(self) -> float:
        """Compute total metabolic energy."""
        # Simple: ATP + GTP levels
        energy = 0.0
        for name in ["atp", "gtp"]:
            idx = self._find_metabolite(name)
            if idx >= 0:
                energy += self.state.concentrations[idx]
        return energy
    
    def _compute_growth_rate(self) -> float:
        """Compute current growth rate."""
        # Growth rate ~ biomass production rate
        # Use protein level as proxy
        protein_idx = self._find_metabolite("protein")
        if protein_idx >= 0:
            return 0.01 * self.state.concentrations[protein_idx]
        return 0.0
    
    def simulate(
        self,
        duration: float,
        dt: float = 0.1,
        record_interval: int = 1,
        verbose: bool = True,
    ) -> SimulationResult:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Total simulation time
            dt: Time step
            record_interval: Record state every N steps
            verbose: Print progress
            
        Returns:
            SimulationResult with trajectories
        """
        if self.state is None:
            raise RuntimeError("Call initialize() first")
        
        n_steps = int(duration / dt)
        n_records = n_steps // record_interval + 1
        
        # Allocate recording arrays
        times = np.zeros(n_records)
        concentrations = np.zeros((n_records, self.metabolic_network.n_met))
        n_genes = self.gene_network.n_genes if self.gene_network else 0
        gene_expression = np.zeros((n_records, max(n_genes, 1)))
        growth_rates = np.zeros(n_records)
        
        # Record initial state
        record_idx = 0
        times[0] = self.state.time
        concentrations[0] = self.state.concentrations
        if n_genes > 0:
            gene_expression[0] = self.state.gene_expression
        growth_rates[0] = self.state.growth_rate
        record_idx = 1
        
        # Run simulation
        start_time = time.time()
        
        for step in range(1, n_steps + 1):
            self.step(dt)
            
            if step % record_interval == 0 and record_idx < n_records:
                times[record_idx] = self.state.time
                concentrations[record_idx] = self.state.concentrations
                if n_genes > 0:
                    gene_expression[record_idx] = self.state.gene_expression
                growth_rates[record_idx] = self.state.growth_rate
                record_idx += 1
            
            if verbose and step % (n_steps // 10) == 0:
                print(f"  Step {step}/{n_steps} (t={self.state.time:.1f})")
        
        wall_time = time.time() - start_time
        
        # Estimate speedup vs ODE
        # ODE: ~0.1ms per step * n_met * n_rxn ops
        # Wave: ~0.01ms per step * n_modes ops
        ode_estimate = n_steps * 0.0001 * self.metabolic_network.n_met * self.metabolic_network.n_rxn
        speedup = ode_estimate / max(wall_time, 1e-6)
        
        if verbose:
            print(f"Completed in {wall_time:.2f}s (estimated {speedup:.0f}x speedup)")
        
        return SimulationResult(
            times=times[:record_idx],
            concentrations=concentrations[:record_idx],
            gene_expression=gene_expression[:record_idx],
            growth_rates=growth_rates[:record_idx],
            wall_time=wall_time,
            n_steps=n_steps,
            speedup_estimate=speedup,
            final_state=self.state.copy(),
        )
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about the simulator."""
        diag = {
            "n_metabolites": self.metabolic_network.n_met,
            "n_reactions": self.metabolic_network.n_rxn,
            "n_genes": self.gene_network.n_genes if self.gene_network else 0,
            "n_knockouts": len(self.knockouts),
            "use_siren_field": self.use_siren_field,
            "modal_engine": self.modal_engine.get_diagnostics(),
        }
        
        if self.gene_propagator is not None:
            diag["gene_propagator"] = self.gene_propagator.get_diagnostics()
        
        if self.siren_field is not None:
            diag["siren_field"] = self.siren_field.get_diagnostics()
        
        return diag
    
    @classmethod
    def from_v33_network(
        cls,
        stoichiometry: np.ndarray,
        metabolites: List[str],
        reactions: List[str],
        kcat: np.ndarray,
        km: np.ndarray,
        genes: Optional[List[str]] = None,
        gene_reaction_map: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ) -> "WaveCellSimulator":
        """
        Create simulator from V33-style network data.
        
        This is the bridge from V33 to V34.
        """
        # Build metabolic network
        metabolic_network = MetabolicNetwork(
            metabolites=metabolites,
            reactions=reactions,
            stoichiometry=stoichiometry,
            reversible=np.ones(len(reactions), dtype=bool),
            kcat=kcat,
            km=km,
            genes=genes or [],
        )
        
        # Build gene network if provided
        if genes and gene_reaction_map:
            interaction_matrix = np.zeros((len(genes), len(genes)))
            # Simple: genes sharing reactions interact
            for g1_idx, (g1, rxns1) in enumerate(gene_reaction_map.items()):
                for g2_idx, (g2, rxns2) in enumerate(gene_reaction_map.items()):
                    if g1 != g2:
                        shared = len(set(rxns1) & set(rxns2))
                        interaction_matrix[g1_idx, g2_idx] = 0.1 * shared
            
            gene_network = GeneNetwork(
                genes=genes,
                interaction_matrix=interaction_matrix,
                gene_reaction_map=gene_reaction_map,
            )
        else:
            gene_network = None
        
        return cls(
            metabolic_network=metabolic_network,
            gene_network=gene_network,
            **kwargs,
        )


# ============================================================================
# Convenience functions
# ============================================================================

def create_test_simulator(n_metabolites: int = 20, n_reactions: int = 30) -> WaveCellSimulator:
    """Create a minimal test simulator for smoke testing."""
    
    np.random.seed(42)
    
    # Random stoichiometry
    stoichiometry = np.random.randn(n_metabolites, n_reactions) * 0.5
    
    # Make sparse
    mask = np.abs(stoichiometry) > 0.3
    stoichiometry = stoichiometry * mask
    
    metabolic_network = MetabolicNetwork(
        metabolites=[f"met_{i}" for i in range(n_metabolites)],
        reactions=[f"rxn_{i}" for i in range(n_reactions)],
        stoichiometry=stoichiometry,
        reversible=np.ones(n_reactions, dtype=bool),
        kcat=np.ones(n_reactions) * 10,
        km=np.ones((n_reactions, 3)) * 0.1,
    )
    
    gene_network = create_minimal_network(n_genes=10)
    
    return WaveCellSimulator(
        metabolic_network=metabolic_network,
        gene_network=gene_network,
        use_siren_field=False,
    )
