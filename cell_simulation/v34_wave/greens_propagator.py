"""
Green's Function Propagator for Gene Regulatory Networks
=========================================================

Models non-local gene-gene interactions via the Green's function:

    G(ω) = (ω + iη - H)^(-1)

Where:
- ω is the frequency (time scale of regulation)
- η is the broadening parameter (damping/decay)
- H is the effective Hamiltonian (gene-gene interaction matrix)

The Green's function captures how regulatory signals propagate through
the gene network, including:
- Direct interactions (A regulates B)
- Indirect paths (A → X → B)
- Feedback loops
- Long-range correlations

Based on:
- dark_manifold_cellular_field/src/qft/greens_function.py
- Concepts from quantum field theory for non-local propagation

Author: Naresh Chhillar, 2026
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class GeneNetwork:
    """Gene regulatory network definition."""
    
    genes: List[str]
    
    # Interaction matrix: H[i,j] = strength of gene j regulating gene i
    # Positive = activation, Negative = repression
    interaction_matrix: np.ndarray  # [n_genes, n_genes]
    
    # Gene-reaction associations
    gene_reaction_map: Dict[str, List[int]]  # gene -> list of reaction indices
    
    @property
    def n_genes(self) -> int:
        return len(self.genes)
    
    def get_regulators(self, gene_idx: int) -> List[Tuple[int, float]]:
        """Get all regulators of a gene with their strengths."""
        regulators = []
        for j in range(self.n_genes):
            strength = self.interaction_matrix[gene_idx, j]
            if abs(strength) > 1e-6:
                regulators.append((j, strength))
        return regulators
    
    def get_targets(self, gene_idx: int) -> List[Tuple[int, float]]:
        """Get all targets of a gene with their strengths."""
        targets = []
        for i in range(self.n_genes):
            strength = self.interaction_matrix[i, gene_idx]
            if abs(strength) > 1e-6:
                targets.append((i, strength))
        return targets


class GeneRegulatoryPropagator:
    """
    Non-local gene regulatory propagator using Green's function.
    
    Core idea: Instead of simulating step-by-step signal propagation,
    we compute the Green's function which encodes ALL possible paths
    through the network at once.
    
    This allows us to answer questions like:
    - If gene A is knocked out, what's the effect on gene Z?
    - What's the steady-state response to a perturbation?
    - Which genes are most sensitive to network changes?
    
    Without explicitly simulating every intermediate step.
    """
    
    def __init__(
        self,
        network: GeneNetwork,
        eta: float = 0.05,
        use_low_rank: bool = True,
        rank: int = 32,
    ):
        """
        Initialize Green's function propagator.
        
        Args:
            network: Gene regulatory network
            eta: Broadening parameter (damping)
            use_low_rank: Use low-rank approximation for efficiency
            rank: Rank for low-rank approximation
        """
        self.network = network
        self.eta = eta
        self.use_low_rank = use_low_rank
        self.rank = min(rank, network.n_genes // 2)
        
        # Precompute Hamiltonian
        self.H = self._build_hamiltonian()
        
        # Precompute eigendecomposition for fast G(ω) evaluation
        self._precompute_eigen()
        
        # Cache for Green's functions at different frequencies
        self._greens_cache: Dict[float, np.ndarray] = {}
    
    def _build_hamiltonian(self) -> np.ndarray:
        """
        Build effective Hamiltonian from interaction matrix.
        
        The Hamiltonian encodes the gene-gene interaction strengths.
        We symmetrize it for proper spectral properties.
        """
        H = self.network.interaction_matrix.copy()
        
        # Add self-interaction (degradation) term on diagonal
        # This ensures the system is stable
        for i in range(self.network.n_genes):
            H[i, i] -= 0.1  # Basal degradation rate
        
        # Symmetrize for real eigenvalues
        H = 0.5 * (H + H.T)
        
        return H
    
    def _precompute_eigen(self) -> None:
        """Precompute eigendecomposition for fast propagator evaluation."""
        
        if self.use_low_rank and self.network.n_genes > 2 * self.rank:
            # Low-rank approximation via truncated SVD
            # H ≈ U @ S @ V.T
            try:
                from scipy.sparse.linalg import svds
                U, S, Vt = svds(self.H, k=self.rank)
                self.eigenvalues = S
                self.eigenvectors = U
                self.eigenvectors_inv = Vt.T  # For reconstruction
            except ImportError:
                # Fallback to full eigendecomposition
                self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.H)
                self.eigenvectors_inv = self.eigenvectors.T
        else:
            # Full eigendecomposition
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.H)
            self.eigenvectors_inv = self.eigenvectors.T
    
    def greens_function(self, omega: float = 0.0) -> np.ndarray:
        """
        Compute Green's function G(ω).
        
        G(ω) = (ω + iη - H)^(-1)
        
        Using spectral decomposition:
        G(ω) = V @ diag(1/(ω + iη - λ_k)) @ V^T
        
        Args:
            omega: Frequency/energy parameter
            
        Returns:
            G: [n_genes, n_genes] Green's function matrix
        """
        # Check cache
        cache_key = round(omega, 6)
        if cache_key in self._greens_cache:
            return self._greens_cache[cache_key]
        
        n = self.network.n_genes
        
        # Spectral representation
        # G_ij = sum_k V_ik * V_jk / (omega + i*eta - lambda_k)
        
        denominators = omega + 1j * self.eta - self.eigenvalues
        
        # Handle potential division by zero
        safe_denom = np.where(
            np.abs(denominators) < 1e-10,
            1e-10 * np.sign(denominators.real + 1e-10),
            denominators
        )
        
        # Build G using spectral decomposition
        n_modes = len(self.eigenvalues)
        G = np.zeros((n, n), dtype=complex)
        
        for k in range(n_modes):
            v_k = self.eigenvectors[:, k]
            G += np.outer(v_k, v_k) / safe_denom[k]
        
        # Take real part (imaginary part encodes dissipation)
        G_real = np.abs(G).astype(float)
        
        # Clamp for numerical stability
        G_real = np.clip(G_real, 0, 10.0)
        
        # Cache result
        self._greens_cache[cache_key] = G_real
        
        return G_real
    
    def propagate(
        self,
        perturbation: np.ndarray,
        omega: float = 0.0,
    ) -> np.ndarray:
        """
        Propagate a perturbation through the gene network.
        
        Args:
            perturbation: [n_genes] perturbation vector (e.g., knockout = -1)
            omega: Frequency parameter (0 = static response)
            
        Returns:
            response: [n_genes] response to perturbation
        """
        G = self.greens_function(omega)
        response = G @ perturbation
        return response
    
    def knockout_effect(
        self,
        gene_idx: int,
        omega: float = 0.0,
    ) -> np.ndarray:
        """
        Compute effect of knocking out a single gene.
        
        Args:
            gene_idx: Index of gene to knock out
            omega: Frequency parameter
            
        Returns:
            effect: [n_genes] change in all gene expression levels
        """
        perturbation = np.zeros(self.network.n_genes)
        perturbation[gene_idx] = -1.0  # Knockout = -1
        
        return self.propagate(perturbation, omega)
    
    def sensitivity_matrix(self, omega: float = 0.0) -> np.ndarray:
        """
        Compute sensitivity matrix: how each gene affects all others.
        
        S[i,j] = effect on gene i from perturbation in gene j
        
        This is essentially |G(ω)|.
        """
        return self.greens_function(omega)
    
    def find_critical_genes(
        self,
        threshold: float = 0.5,
        omega: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Find genes whose knockout has the largest network effect.
        
        Args:
            threshold: Minimum effect magnitude to report
            omega: Frequency parameter
            
        Returns:
            List of (gene_idx, total_effect) tuples, sorted by effect
        """
        effects = []
        
        for g in range(self.network.n_genes):
            effect = self.knockout_effect(g, omega)
            total_effect = np.sum(np.abs(effect))
            if total_effect > threshold:
                effects.append((g, total_effect))
        
        return sorted(effects, key=lambda x: -x[1])
    
    def spectral_density(self, omega_range: np.ndarray) -> np.ndarray:
        """
        Compute spectral density A(ω) = -Im[Tr G(ω)] / π
        
        This reveals the characteristic frequencies of the network.
        """
        densities = []
        
        for omega in omega_range:
            G = self.greens_function(omega)
            # Trace of imaginary part (we stored real, so approximate)
            density = np.trace(G)
            densities.append(density)
        
        return np.array(densities)
    
    def steady_state_expression(
        self,
        external_inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute steady-state gene expression given external inputs.
        
        Args:
            external_inputs: [n_genes] external regulatory signals
            
        Returns:
            expression: [n_genes] steady-state expression levels
        """
        # Static response (omega = 0)
        G = self.greens_function(0.0)
        
        # Steady state = G @ inputs
        expression = G @ external_inputs
        
        # Apply non-negativity (expression can't be negative)
        return np.maximum(expression, 0)
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information about the propagator."""
        return {
            "n_genes": self.network.n_genes,
            "eta": self.eta,
            "use_low_rank": self.use_low_rank,
            "rank": self.rank,
            "n_eigenvalues": len(self.eigenvalues),
            "eigenvalue_range": (float(self.eigenvalues.min()), float(self.eigenvalues.max())),
            "h_norm": float(np.linalg.norm(self.H)),
            "cache_size": len(self._greens_cache),
        }


def create_from_gpr_rules(
    genes: List[str],
    gpr_rules: Dict[str, str],
    reaction_genes: Dict[int, List[str]],
) -> GeneNetwork:
    """
    Create GeneNetwork from GPR (Gene-Protein-Reaction) rules.
    
    This is how iMB155 encodes gene-reaction associations.
    
    Args:
        genes: List of gene names
        gpr_rules: Dict of reaction_id -> GPR string (e.g., "geneA and geneB")
        reaction_genes: Dict of reaction_idx -> list of associated genes
        
    Returns:
        GeneNetwork instance
    """
    n_genes = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    
    # Build interaction matrix
    # Simple model: genes that catalyze the same reaction interact
    interaction_matrix = np.zeros((n_genes, n_genes))
    
    for rxn_idx, rxn_genes in reaction_genes.items():
        for g1 in rxn_genes:
            for g2 in rxn_genes:
                if g1 != g2 and g1 in gene_to_idx and g2 in gene_to_idx:
                    i, j = gene_to_idx[g1], gene_to_idx[g2]
                    # Co-regulation implies interaction
                    interaction_matrix[i, j] += 0.1
    
    # Build gene-reaction map
    gene_reaction_map = {g: [] for g in genes}
    for rxn_idx, rxn_genes in reaction_genes.items():
        for g in rxn_genes:
            if g in gene_reaction_map:
                gene_reaction_map[g].append(rxn_idx)
    
    return GeneNetwork(
        genes=genes,
        interaction_matrix=interaction_matrix,
        gene_reaction_map=gene_reaction_map,
    )


def create_minimal_network(n_genes: int = 38) -> GeneNetwork:
    """
    Create a minimal test network.
    
    For smoke testing and validation.
    """
    genes = [f"gene_{i}" for i in range(n_genes)]
    
    # Random sparse interaction matrix
    np.random.seed(42)
    interaction_matrix = np.random.randn(n_genes, n_genes) * 0.1
    
    # Make sparse (only ~20% non-zero)
    mask = np.random.rand(n_genes, n_genes) > 0.8
    interaction_matrix = interaction_matrix * mask
    
    # Simple gene-reaction map (each gene -> one reaction)
    gene_reaction_map = {g: [i] for i, g in enumerate(genes)}
    
    return GeneNetwork(
        genes=genes,
        interaction_matrix=interaction_matrix,
        gene_reaction_map=gene_reaction_map,
    )
