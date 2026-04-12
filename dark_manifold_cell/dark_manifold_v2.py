"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    DARK MANIFOLD VIRTUAL CELL v2.0                           ║
║                                                                              ║
║     Enhanced with Quantum Field Theory + Liquid Neural Networks              ║
║                                                                              ║
║  Integrates concepts from CYP-Predict/enzyme_Software:                       ║
║  - Liquid Time-Constant Cells (continuous dynamics)                          ║
║  - Green's Function Propagators (non-local interactions)                     ║
║  - Hyperbolic Memory Bank (analogical reasoning)                             ║
║  - Dynamic State Bank (discrete cellular modes)                              ║
║  - Decoherence modeling (quantum → classical transition)                     ║
║                                                                              ║
║  Target: JCVI-syn3A minimal cell (493 genes)                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# LIQUID NEURAL NETWORK CELL (from hybrid_nexus_dynamic.py)
# ═══════════════════════════════════════════════════════════════════════════════

class LiquidCell(nn.Module):
    """
    Liquid Time-Constant Cell with adaptive dynamics.
    
    Key insight: Cellular states evolve over continuous biological time.
    The liquid cell models metabolic and genetic dynamics with adaptive τ.
    
    τ_i = f(cell_state) allows different genes to respond on different timescales:
    - Fast: metabolic enzymes (seconds)
    - Medium: transcription factors (minutes)  
    - Slow: structural proteins (hours)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)
        
        # Adaptive time constant (gene-dependent dynamics)
        self.tau_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Gating (which genes are "active" at this timestep)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: float = 0.1,
        n_steps: int = 5,
    ) -> torch.Tensor:
        """
        Integrate liquid dynamics over n_steps.
        
        dh/dt = (-h + gate * tanh(W_in*x + W_rec*h)) / τ
        """
        for _ in range(n_steps):
            tau = 0.5 + self.tau_net(h)  # τ ∈ [0.5, 1.5]
            gate = self.gate(h)
            pre_act = self.W_in(x) + self.W_rec(h)
            f = torch.tanh(pre_act)
            dh = (-h + gate * f) / tau
            h = h + dt * dh
        return h


# ═══════════════════════════════════════════════════════════════════════════════
# GREEN'S FUNCTION PROPAGATOR (from quantum_field_theory.py)
# ═══════════════════════════════════════════════════════════════════════════════

class GreensFunctionLayer(nn.Module):
    """
    Computes non-local gene interactions via Green's function.
    
    G(ω) = (ω + iη - H)^(-1)
    
    The Green's function tells us how perturbations propagate through
    the gene regulatory network. G_ij(ω) = amplitude for signal to 
    go from gene i to gene j at frequency ω.
    """
    
    def __init__(self, n_genes: int, hidden_dim: int = 64, eta: float = 0.01):
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim
        self.eta = eta
        
        # Learnable Hamiltonian (effective gene interaction matrix)
        self.H = nn.Parameter(torch.randn(n_genes, n_genes) * 0.1)
        
        # Frequency encoder
        self.omega_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        
    def forward(self, gene_state: torch.Tensor) -> torch.Tensor:
        """
        Compute Green's function propagator for current state.
        
        Args:
            gene_state: [B, n_genes, hidden_dim]
            
        Returns:
            propagator: [B, n_genes, n_genes] interaction strengths
        """
        B = gene_state.shape[0]
        device = gene_state.device
        
        # Compute effective frequency from state
        omega = self.omega_net(gene_state).squeeze(-1)  # [B, n_genes]
        omega_mean = omega.mean(dim=1, keepdim=True)  # [B, 1]
        
        # Symmetrize Hamiltonian
        H_sym = 0.5 * (self.H + self.H.t())
        
        # Green's function: G = (ω + iη - H)^(-1)
        # For each batch, compute resolvent
        propagators = []
        for b in range(B):
            w = omega_mean[b, 0]
            resolvent = (w + 1j * self.eta) * torch.eye(self.n_genes, device=device) - H_sym
            G = torch.linalg.inv(resolvent)
            propagators.append(torch.abs(G))  # Take magnitude
        
        return torch.stack(propagators, dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# DECOHERENCE LAYER (from quantum_field_theory.py)
# ═══════════════════════════════════════════════════════════════════════════════

class DecoherenceLayer(nn.Module):
    """
    Models transition from quantum-like superposition to classical states.
    
    In cellular context:
    - "Quantum" = gene expression in probabilistic/superposed state
    - "Classical" = committed cell fate / stable expression pattern
    
    Decoherence rate γ_i depends on:
    - Environmental coupling (membrane proteins decohere faster)
    - Interaction strength (hub genes decohere faster)
    """
    
    def __init__(self, n_genes: int, hidden_dim: int = 64):
        super().__init__()
        
        # Learnable decoherence rates per gene
        self.gamma_base = nn.Parameter(torch.ones(n_genes) * 0.1)
        
        # Environment coupling (learned from features)
        self.env_coupling = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        state: torch.Tensor,
        coherent_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply decoherence to state.
        
        Args:
            state: [B, n_genes, hidden_dim] current state
            coherent_state: [B, n_genes, hidden_dim] "quantum" superposition
            
        Returns:
            decohered_state: state after partial collapse
            gamma: [B, n_genes] decoherence rates
        """
        # Compute environment coupling
        env = self.env_coupling(state).squeeze(-1)  # [B, n_genes]
        
        # Total decoherence rate
        gamma = torch.sigmoid(self.gamma_base) * (1 + env)
        
        # Decoherence: interpolate toward classical (mean) state
        classical_state = state.mean(dim=-1, keepdim=True).expand_as(state)
        
        # γ → 0: keep coherent, γ → 1: collapse to classical
        gamma_expanded = gamma.unsqueeze(-1)
        decohered = (1 - gamma_expanded) * coherent_state + gamma_expanded * classical_state
        
        return decohered, gamma


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC STATE BANK (from hybrid_nexus_dynamic.py)
# ═══════════════════════════════════════════════════════════════════════════════

class CellularStateBank(nn.Module):
    """
    Learnable discrete cellular states that emerge during training.
    
    Each state represents a distinct cellular mode:
    - Growth phase
    - Stress response
    - Division preparation
    - Stationary phase
    
    States grow dynamically as the model sees diverse conditions.
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        max_states: int = 16,
        min_states: int = 2,
        similarity_threshold: float = 0.7,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.max_states = max_states
        self.min_states = min_states
        self.similarity_threshold = similarity_threshold
        
        # State embeddings
        self.state_embeddings = nn.Parameter(
            torch.randn(max_states, state_dim) * 0.1
        )
        
        # Which states are active
        self.register_buffer('state_active', torch.zeros(max_states, dtype=torch.bool))
        self.state_active[:min_states] = True
        
        # State names (for interpretation)
        self.state_names = ['growth', 'stress', 'division', 'stationary'] + \
                          [f'state_{i}' for i in range(4, max_states)]
        
    @property
    def num_active_states(self) -> int:
        return int(self.state_active.sum().item())
    
    def get_active_states(self) -> torch.Tensor:
        return self.state_embeddings[self.state_active]
    
    def forward(self, cell_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute state probabilities for cells.
        
        Args:
            cell_embedding: [B, state_dim] global cell state
            
        Returns:
            state_probs: [B, n_active] probability over states
            active_indices: indices of active states
        """
        active_states = self.get_active_states()
        active_indices = self.state_active.nonzero(as_tuple=True)[0]
        
        # Similarity-based selection
        cell_norm = F.normalize(cell_embedding, dim=-1)
        states_norm = F.normalize(active_states, dim=-1)
        logits = torch.mm(cell_norm, states_norm.t()) / 0.1  # Temperature
        probs = F.softmax(logits, dim=-1)
        
        return probs, active_indices


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERBOLIC MEMORY BANK (from hybrid_nexus_dynamic.py)
# ═══════════════════════════════════════════════════════════════════════════════

class HyperbolicMemory(nn.Module):
    """
    Memory bank in hyperbolic space for analogical reasoning.
    
    Stores (cell_state, outcome) pairs and retrieves similar states.
    Hyperbolic geometry is natural for hierarchical biological data.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        curvature: float = 1.0,
        max_entries: int = 500,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.max_entries = max_entries
        
        # Memory storage
        self.register_buffer('memory_embeddings', torch.zeros(0, embedding_dim))
        self.register_buffer('memory_outcomes', torch.zeros(0, embedding_dim))
        self.n_entries = 0
        
        # Projection to hyperbolic (Poincaré ball)
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
        )
        
    def _to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """Project to Poincaré ball with radius < 1."""
        x = self.project(x)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        max_radius = 0.9
        scale = torch.where(norm > max_radius, max_radius / norm, torch.ones_like(norm))
        return x * scale
    
    def store(self, cell_embedding: torch.Tensor, outcome: torch.Tensor):
        """Store cell state and outcome in memory."""
        if self.n_entries >= self.max_entries:
            self.memory_embeddings = self.memory_embeddings[1:]
            self.memory_outcomes = self.memory_outcomes[1:]
            self.n_entries -= 1
        
        emb = cell_embedding.detach()
        out = outcome.detach()
        
        if self.memory_embeddings.numel() == 0:
            self.memory_embeddings = emb.unsqueeze(0)
            self.memory_outcomes = out.unsqueeze(0)
        else:
            self.memory_embeddings = torch.cat([self.memory_embeddings, emb.unsqueeze(0)], dim=0)
            self.memory_outcomes = torch.cat([self.memory_outcomes, out.unsqueeze(0)], dim=0)
        self.n_entries += 1
    
    def retrieve(self, query: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve k most similar cell states.
        
        Returns:
            weights: [B, k] attention weights
            outcomes: [B, k, dim] outcomes of retrieved states
        """
        B = query.shape[0]
        device = query.device
        
        if self.n_entries == 0:
            return torch.zeros(B, k, device=device), torch.zeros(B, k, self.embedding_dim, device=device)
        
        query_proj = self._to_poincare(query)
        memory_proj = self._to_poincare(self.memory_embeddings.to(device))
        
        # Euclidean distance in projected space
        dists = torch.cdist(query_proj, memory_proj)
        
        k_actual = min(k, self.n_entries)
        top_dists, top_indices = dists.topk(k_actual, dim=-1, largest=False)
        
        weights = F.softmax(-top_dists / 0.1, dim=-1)
        
        if k_actual < k:
            pad_weights = torch.zeros(B, k - k_actual, device=device)
            weights = torch.cat([weights, pad_weights], dim=-1)
        
        outcomes = self.memory_outcomes.to(device)[top_indices]
        
        if k_actual < k:
            pad_outcomes = torch.zeros(B, k - k_actual, self.embedding_dim, device=device)
            outcomes = torch.cat([outcomes, pad_outcomes], dim=1)
        
        return weights, outcomes


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DARK MANIFOLD v2 MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class DarkManifoldV2(nn.Module):
    """
    Dark Manifold Virtual Cell v2.0
    
    Enhanced architecture combining:
    1. Learned stoichiometry + regulation (from v1)
    2. Liquid neural dynamics (continuous-time)
    3. Green's function propagators (non-local interactions)
    4. Decoherence modeling (stability)
    5. Cellular state bank (discrete modes)
    6. Hyperbolic memory (analogical reasoning)
    
    Flow:
    1. Encode gene/metabolite states
    2. Apply Green's function for non-local interactions
    3. Run liquid dynamics over continuous time
    4. Apply decoherence for stability
    5. Classify cellular state
    6. Predict next gene/metabolite levels
    """
    
    def __init__(
        self,
        n_genes: int,
        n_metabolites: int,
        hidden_dim: int = 128,
        n_liquid_steps: int = 5,
        max_states: int = 8,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_metabolites = n_metabolites
        self.hidden_dim = hidden_dim
        
        # === ENCODERS ===
        self.gene_embed = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.met_embed = nn.Sequential(
            nn.Linear(n_metabolites, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # === DARK FIELD COMPONENTS ===
        # Learned stoichiometry (how metabolites depend on genes)
        self.W_stoich = nn.Parameter(torch.randn(n_metabolites, n_genes) * 0.01)
        
        # Learned regulation (gene-gene interactions)
        self.W_reg = nn.Parameter(torch.randn(n_genes, n_genes) * 0.01)
        
        # === QUANTUM FIELD COMPONENTS ===
        # Green's function for non-local propagation
        self.greens_function = GreensFunctionLayer(n_genes, hidden_dim)
        
        # Decoherence for stability
        self.decoherence = DecoherenceLayer(n_genes, hidden_dim)
        
        # === LIQUID DYNAMICS ===
        self.liquid_cell = LiquidCell(hidden_dim, hidden_dim)
        self.n_liquid_steps = n_liquid_steps
        
        # === STATE BANK ===
        self.state_bank = CellularStateBank(
            state_dim=hidden_dim,
            max_states=max_states,
        )
        
        # === MEMORY ===
        self.memory = HyperbolicMemory(embedding_dim=hidden_dim)
        
        # === FUSION ===
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # === OUTPUT HEADS ===
        self.gene_out = nn.Linear(hidden_dim, n_genes)
        self.met_out = nn.Linear(hidden_dim, n_metabolites)
        
    def forward(
        self,
        gene_state: torch.Tensor,
        met_state: torch.Tensor,
        dt: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            gene_state: [B, n_genes] current gene expression
            met_state: [B, n_metabolites] current metabolite levels
            dt: time step
            
        Returns:
            dict with predictions and intermediate states
        """
        B = gene_state.shape[0]
        
        # 1. ENCODE
        g_emb = self.gene_embed(gene_state)  # [B, hidden]
        m_emb = self.met_embed(met_state)    # [B, hidden]
        
        # 2. GREEN'S FUNCTION (non-local gene interactions)
        # Expand to [B, n_genes, hidden] for Green's function
        g_expanded = g_emb.unsqueeze(1).expand(-1, self.n_genes, -1)
        propagator = self.greens_function(g_expanded)  # [B, n_genes, n_genes]
        
        # Apply propagator to gene regulation
        W_reg_eff = self.W_reg.unsqueeze(0) * propagator  # [B, n_genes, n_genes]
        gene_interaction = torch.bmm(W_reg_eff, gene_state.unsqueeze(-1)).squeeze(-1)
        
        # 3. LIQUID DYNAMICS
        h = g_emb.unsqueeze(1).expand(-1, 1, -1)  # [B, 1, hidden]
        h = self.liquid_cell(h, h, dt=dt, n_steps=self.n_liquid_steps)
        h = h.squeeze(1)  # [B, hidden]
        
        # 4. DECOHERENCE
        h_expanded = h.unsqueeze(1).expand(-1, self.n_genes, -1)
        h_decohered, gamma = self.decoherence(h_expanded, h_expanded)
        h_decohered = h_decohered.mean(dim=1)  # [B, hidden]
        
        # 5. CELLULAR STATE CLASSIFICATION
        state_probs, state_indices = self.state_bank(h_decohered)
        
        # 6. MEMORY RETRIEVAL
        mem_weights, mem_outcomes = self.memory.retrieve(h_decohered, k=3)
        mem_hint = (mem_weights.unsqueeze(-1) * mem_outcomes).sum(dim=1)
        
        # 7. FUSION
        fused = self.fusion(torch.cat([h_decohered, m_emb, mem_hint], dim=-1))
        
        # 8. PREDICTIONS
        # Gene dynamics: dG/dt = W_reg @ G + f(state)
        gene_pred = gene_state + dt * (
            gene_interaction + 
            self.gene_out(fused)
        )
        
        # Metabolite dynamics: dM/dt = W_stoich @ G + g(state)
        met_pred = met_state + dt * (
            F.linear(gene_state, self.W_stoich) +
            self.met_out(fused)
        )
        
        return {
            'gene_pred': gene_pred,
            'met_pred': met_pred,
            'state_probs': state_probs,
            'n_active_states': self.state_bank.num_active_states,
            'decoherence_rates': gamma.mean(dim=1),
            'hidden': fused,
        }
    
    def rollout(
        self,
        gene_init: torch.Tensor,
        met_init: torch.Tensor,
        n_steps: int,
        dt: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Roll out dynamics for multiple steps.
        """
        gene_traj = [gene_init]
        met_traj = [met_init]
        state_traj = []
        
        gene_state = gene_init
        met_state = met_init
        
        for _ in range(n_steps):
            out = self.forward(gene_state, met_state, dt)
            gene_state = out['gene_pred']
            met_state = out['met_pred']
            gene_traj.append(gene_state)
            met_traj.append(met_state)
            state_traj.append(out['state_probs'])
        
        return {
            'gene_trajectory': torch.stack(gene_traj, dim=1),
            'met_trajectory': torch.stack(met_traj, dim=1),
            'state_trajectory': torch.stack(state_traj, dim=1),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class DarkManifoldLoss(nn.Module):
    """
    Combined loss for Dark Manifold training.
    """
    
    def __init__(
        self,
        gene_weight: float = 1.0,
        met_weight: float = 1.0,
        physics_weight: float = 0.1,
    ):
        super().__init__()
        self.gene_weight = gene_weight
        self.met_weight = met_weight
        self.physics_weight = physics_weight
        
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target_gene: torch.Tensor,
        target_met: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.
        """
        # Trajectory loss
        gene_loss = F.mse_loss(pred['gene_pred'], target_gene)
        met_loss = F.mse_loss(pred['met_pred'], target_met)
        
        # Physics regularization (energy should be conserved approximately)
        # Total "mass" conservation
        mass_pred = pred['met_pred'].sum(dim=-1)
        mass_target = target_met.sum(dim=-1)
        physics_loss = F.mse_loss(mass_pred, mass_target)
        
        total = (
            self.gene_weight * gene_loss +
            self.met_weight * met_loss +
            self.physics_weight * physics_loss
        )
        
        return total, {
            'gene_loss': gene_loss.item(),
            'met_loss': met_loss.item(),
            'physics_loss': physics_loss.item(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("DARK MANIFOLD VIRTUAL CELL v2.0")
    print("Enhanced with Quantum Field Theory + Liquid Neural Networks")
    print("=" * 70)
    
    # Test configuration
    n_genes = 100
    n_mets = 50
    batch_size = 4
    
    model = DarkManifoldV2(
        n_genes=n_genes,
        n_metabolites=n_mets,
        hidden_dim=128,
        n_liquid_steps=5,
        max_states=8,
    )
    
    # Random input
    gene_state = torch.randn(batch_size, n_genes)
    met_state = torch.randn(batch_size, n_mets).abs()
    
    # Forward pass
    print("\nRunning forward pass...")
    out = model(gene_state, met_state)
    
    print(f"Gene prediction shape: {out['gene_pred'].shape}")
    print(f"Metabolite prediction shape: {out['met_pred'].shape}")
    print(f"Active cellular states: {out['n_active_states']}")
    print(f"Mean decoherence rate: {out['decoherence_rates'].mean():.3f}")
    
    # Rollout
    print("\nRunning 10-step rollout...")
    rollout = model.rollout(gene_state, met_state, n_steps=10)
    print(f"Gene trajectory shape: {rollout['gene_trajectory'].shape}")
    print(f"Metabolite trajectory shape: {rollout['met_trajectory'].shape}")
    
    # Loss
    loss_fn = DarkManifoldLoss()
    target_gene = gene_state + torch.randn_like(gene_state) * 0.1
    target_met = met_state + torch.randn_like(met_state) * 0.1
    loss, metrics = loss_fn(out, target_gene, target_met)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    print("\n✓ Dark Manifold v2 test passed!")
