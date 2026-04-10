"""
Unified Cognitive Engine (UCE)
==============================

A TRULY unified architecture where learning, reasoning, and electron dynamics
are ONE integrated system, not three separate modules.

Core Principle: Everything is a continuous field that evolves together.

For each atom i, we maintain a unified state:
    S_i(t) = [h_i(t), ρ_i(t), m_i(t)]
    
    h_i = hidden representation (learning)
    ρ_i = electron density state (wave dynamics) 
    m_i = reasoning/memory state (analogical reasoning)

All three evolve via coupled ODEs with adaptive time constants:
    τ_i · dS_i/dt = F(S_i, S_neighbors, Memory, Perturbation)

Author: CYP-Predict Team
Version: 2.0 - True Unification
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass

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
    from rdkit.Chem import AllChem, rdFMCS
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass  
class UCEConfig:
    """Configuration for the Unified Cognitive Engine."""
    
    # Core dimensions
    hidden_dim: int = 96          # h dimension (representation)
    wave_dim: int = 32            # ρ dimension (electron density)
    memory_dim: int = 32          # m dimension (reasoning)
    
    @property
    def state_dim(self) -> int:
        """Total unified state dimension."""
        return self.hidden_dim + self.wave_dim + self.memory_dim
    
    # Atom encoding
    atom_feature_dim: int = 64
    edge_feature_dim: int = 16
    
    # Dynamics
    num_layers: int = 4           # Depth of dynamics
    max_ode_steps: int = 20       # Maximum ODE steps per layer
    dt: float = 0.1               # ODE step size
    convergence_threshold: float = 0.005
    
    # Adaptive time constants
    tau_min: float = 0.1
    tau_max: float = 3.0
    
    # Memory/Reasoning
    memory_capacity: int = 2048
    memory_topk: int = 8
    hyperbolic_curvature: float = 1.0
    
    # Wave dynamics
    wave_perturbation_strength: float = 1.0
    
    # Training
    dropout: float = 0.1


# =============================================================================
# UNIFIED STATE REPRESENTATION
# =============================================================================

if TORCH_AVAILABLE:
    
    class UnifiedState(NamedTuple):
        """
        The unified state for all atoms.
        
        Contains three coupled components that evolve together:
        - h: Hidden representations (learning)
        - rho: Electron density states (wave dynamics)
        - m: Memory/reasoning states (analogical reasoning)
        """
        h: Tensor      # [N, hidden_dim]
        rho: Tensor    # [N, wave_dim]
        m: Tensor      # [N, memory_dim]
        
        def flatten(self) -> Tensor:
            """Flatten to single tensor [N, state_dim]."""
            return torch.cat([self.h, self.rho, self.m], dim=-1)
        
        @staticmethod
        def unflatten(S: Tensor, hidden_dim: int, wave_dim: int, memory_dim: int) -> "UnifiedState":
            """Unflatten from single tensor."""
            h = S[..., :hidden_dim]
            rho = S[..., hidden_dim:hidden_dim + wave_dim]
            m = S[..., hidden_dim + wave_dim:]
            return UnifiedState(h, rho, m)
        
        def norm(self) -> Tensor:
            """Compute norm for convergence check."""
            return self.flatten().norm(dim=-1)


# =============================================================================
# COUPLED DYNAMICS - The Heart of the System
# =============================================================================

if TORCH_AVAILABLE:
    
    class MessageAggregator(nn.Module):
        """Aggregate messages from neighbors."""
        
        def __init__(self, state_dim: int, edge_dim: int, hidden: int):
            super().__init__()
            self.edge_encoder = nn.Linear(edge_dim, hidden) if edge_dim > 0 else None
            self.msg_net = nn.Sequential(
                nn.Linear(state_dim * 2 + hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
            )
            self.hidden = hidden
            
        def forward(self, S: Tensor, edge_index: Tensor, 
                    edge_attr: Optional[Tensor] = None) -> Tensor:
            """
            Compute aggregated neighbor messages.
            
            Args:
                S: [N, state_dim] - unified states
                edge_index: [2, E] - edges
                edge_attr: [E, edge_dim] - edge features
                
            Returns:
                messages: [N, hidden] - aggregated messages per atom
            """
            if edge_index.numel() == 0:
                return torch.zeros(S.shape[0], self.hidden, device=S.device)
            
            src, dst = edge_index[0], edge_index[1]
            
            # Edge encoding
            if edge_attr is not None and self.edge_encoder is not None:
                e = self.edge_encoder(edge_attr)
            else:
                e = torch.zeros(src.shape[0], self.hidden, device=S.device)
            
            # Compute messages
            msg_input = torch.cat([S[src], S[dst], e], dim=-1)
            messages = self.msg_net(msg_input)
            
            # Aggregate by destination
            agg = torch.zeros(S.shape[0], self.hidden, device=S.device)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
            
            # Normalize
            degree = torch.bincount(dst, minlength=S.shape[0]).clamp(min=1).float()
            return agg / degree.unsqueeze(-1)
    
    
    class WaveDynamicsOperator(nn.Module):
        """
        Computes dρ/dt - how electron density evolves.
        
        Models Schrödinger-like dynamics:
        dρ/dt = WaveOp(ρ_i, ρ_neighbors, h_i, perturbation)
        
        Key insight: Wave dynamics are COUPLED to hidden states h.
        """
        
        def __init__(self, wave_dim: int, hidden_dim: int, neighbor_dim: int):
            super().__init__()
            
            # Input: own ρ, hidden h, neighbor info, perturbation
            input_dim = wave_dim + hidden_dim + neighbor_dim + 1
            
            self.dynamics = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, wave_dim),
                nn.Tanh(),  # Bounded for stability
            )
            
        def forward(self, rho: Tensor, h: Tensor, neighbor_msg: Tensor,
                    perturbation: Tensor) -> Tensor:
            """
            Compute dρ/dt.
            
            Args:
                rho: [N, wave_dim] - current electron density states
                h: [N, hidden_dim] - hidden representations (coupling!)
                neighbor_msg: [N, neighbor_dim] - neighbor information
                perturbation: [N] - attack perturbation strength
                
            Returns:
                d_rho: [N, wave_dim] - rate of change
            """
            x = torch.cat([rho, h, neighbor_msg, perturbation.unsqueeze(-1)], dim=-1)
            return self.dynamics(x)
    
    
    class ReasoningDynamicsOperator(nn.Module):
        """
        Computes dm/dt - how reasoning state evolves.
        
        This is where analogical reasoning happens DYNAMICALLY:
        - Memory retrieval influences dm/dt
        - Reasoning accumulates over time (not instant!)
        - Coupled to hidden states h
        
        dm/dt = ReasonOp(m_i, h_i, memory_retrieval)
        """
        
        def __init__(self, memory_dim: int, hidden_dim: int, retrieval_dim: int):
            super().__init__()
            
            # Input: own m, hidden h, retrieved memory info
            input_dim = memory_dim + hidden_dim + retrieval_dim
            
            self.dynamics = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, memory_dim),
                nn.Tanh(),
            )
            
            # Gating - controls how much to update reasoning
            self.gate = nn.Sequential(
                nn.Linear(input_dim, memory_dim),
                nn.Sigmoid(),
            )
            
        def forward(self, m: Tensor, h: Tensor, retrieval: Tensor) -> Tensor:
            """
            Compute dm/dt.
            
            Args:
                m: [N, memory_dim] - current reasoning states
                h: [N, hidden_dim] - hidden representations
                retrieval: [N, retrieval_dim] - retrieved memory context
                
            Returns:
                d_m: [N, memory_dim] - rate of change
            """
            x = torch.cat([m, h, retrieval], dim=-1)
            
            # Gated update - reasoning accumulates gradually
            delta = self.dynamics(x)
            gate = self.gate(x)
            
            return gate * delta
    
    
    class HiddenDynamicsOperator(nn.Module):
        """
        Computes dh/dt - how hidden representations evolve.
        
        This is the liquid neural network dynamics:
        dh/dt = (-h + f(h, msg, ρ, m)) / τ
        
        Key: h is influenced by BOTH electron density ρ AND reasoning m.
        """
        
        def __init__(self, hidden_dim: int, wave_dim: int, memory_dim: int, msg_dim: int):
            super().__init__()
            
            # Input: own h, messages, ρ (wave), m (reasoning)
            input_dim = hidden_dim + msg_dim + wave_dim + memory_dim
            
            self.target_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )
            
        def forward(self, h: Tensor, msg: Tensor, rho: Tensor, m: Tensor,
                    tau: Tensor) -> Tensor:
            """
            Compute dh/dt.
            
            Args:
                h: [N, hidden_dim] - current hidden states
                msg: [N, msg_dim] - neighbor messages
                rho: [N, wave_dim] - electron density (coupling!)
                m: [N, memory_dim] - reasoning state (coupling!)
                tau: [N] - adaptive time constants
                
            Returns:
                d_h: [N, hidden_dim] - rate of change
            """
            x = torch.cat([h, msg, rho, m], dim=-1)
            target = self.target_net(x)
            
            # Liquid dynamics: dh/dt = (-h + target) / τ
            return (-h + target) / tau.unsqueeze(-1)
    
    
    class AdaptiveTauNetwork(nn.Module):
        """
        Predicts per-atom time constants τ.
        
        Complex atoms (conjugated systems, near heteroatoms, electronically
        active) get LARGER τ → more ODE steps → "think longer".
        """
        
        def __init__(self, state_dim: int, hidden_dim: int, 
                     tau_min: float = 0.1, tau_max: float = 3.0):
            super().__init__()
            self.tau_min = tau_min
            self.tau_max = tau_max
            
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            
        def forward(self, S: Tensor) -> Tensor:
            """
            Predict τ for each atom.
            
            Args:
                S: [N, state_dim] - unified states
                
            Returns:
                tau: [N] - time constants in [tau_min, tau_max]
            """
            raw = self.net(S).squeeze(-1)
            # Sigmoid scaling
            return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(raw)
    
    
    class CoupledDynamics(nn.Module):
        """
        The unified coupled dynamics system.
        
        Computes dS/dt for the entire unified state:
            dS/dt = [dh/dt, dρ/dt, dm/dt]
            
        All three components are coupled and influence each other.
        """
        
        def __init__(self, config: UCEConfig):
            super().__init__()
            self.config = config
            
            # Message aggregation (operates on full unified state)
            self.message_agg = MessageAggregator(
                state_dim=config.state_dim,
                edge_dim=config.edge_feature_dim,
                hidden=config.hidden_dim,
            )
            
            # Individual dynamics operators (but coupled!)
            self.wave_op = WaveDynamicsOperator(
                wave_dim=config.wave_dim,
                hidden_dim=config.hidden_dim,
                neighbor_dim=config.hidden_dim,
            )
            
            self.reason_op = ReasoningDynamicsOperator(
                memory_dim=config.memory_dim,
                hidden_dim=config.hidden_dim,
                retrieval_dim=config.memory_dim,  # Retrieved memory
            )
            
            self.hidden_op = HiddenDynamicsOperator(
                hidden_dim=config.hidden_dim,
                wave_dim=config.wave_dim,
                memory_dim=config.memory_dim,
                msg_dim=config.hidden_dim,
            )
            
            # Adaptive tau
            self.tau_net = AdaptiveTauNetwork(
                state_dim=config.state_dim,
                hidden_dim=config.hidden_dim,
                tau_min=config.tau_min,
                tau_max=config.tau_max,
            )
            
        def forward(self, state: UnifiedState, edge_index: Tensor,
                    edge_attr: Optional[Tensor], perturbation: Tensor,
                    memory_retrieval: Tensor) -> Tuple[UnifiedState, Tensor]:
            """
            Compute dS/dt for the unified state.
            
            Args:
                state: UnifiedState (h, rho, m)
                edge_index: [2, E] - molecular graph
                edge_attr: [E, edge_dim] - edge features
                perturbation: [N] - attack perturbation per atom
                memory_retrieval: [N, memory_dim] - retrieved memory context
                
            Returns:
                d_state: UnifiedState - derivatives
                tau: [N] - time constants used
            """
            # Get full state
            S = state.flatten()
            
            # Adaptive time constants
            tau = self.tau_net(S)
            
            # Neighbor messages (computed once, shared)
            msg = self.message_agg(S, edge_index, edge_attr)
            
            # Wave dynamics: dρ/dt
            # Coupled to h (hidden states influence electron flow!)
            d_rho = self.wave_op(state.rho, state.h, msg, perturbation)
            
            # Reasoning dynamics: dm/dt
            # Coupled to h (representations guide memory retrieval!)
            d_m = self.reason_op(state.m, state.h, memory_retrieval)
            
            # Hidden dynamics: dh/dt
            # Coupled to BOTH ρ and m
            d_h = self.hidden_op(state.h, msg, state.rho, state.m, tau)
            
            return UnifiedState(d_h, d_rho, d_m), tau


# =============================================================================
# MEMORY SYSTEM - Continuous Reasoning
# =============================================================================

if TORCH_AVAILABLE and RDKIT_AVAILABLE:
    
    class ContinuousMemory(nn.Module):
        """
        Memory bank for analogical reasoning.
        
        Unlike traditional retrieval (instant lookup), this provides
        CONTINUOUS retrieval that evolves with the computation.
        
        At each ODE step, memory retrieval is recomputed based on
        current hidden states, allowing reasoning to refine over time.
        """
        
        def __init__(self, config: UCEConfig):
            super().__init__()
            self.config = config
            
            # Query projection (from hidden state to query)
            self.query_proj = nn.Linear(config.hidden_dim, config.memory_dim)
            
            # Key/Value projections
            self.key_proj = nn.Linear(config.memory_dim, config.memory_dim)
            self.value_proj = nn.Linear(config.memory_dim, config.memory_dim)
            
            # Molecule encoder (SMILES -> embedding)
            self.mol_encoder = nn.Sequential(
                nn.Linear(2048, 256),  # Morgan FP
                nn.SiLU(),
                nn.Linear(256, config.memory_dim),
            )
            
            # Storage
            self.memory_keys: Optional[Tensor] = None
            self.memory_values: Optional[Tensor] = None
            self.memory_mols: List[Chem.Mol] = []
            self.memory_sites: List[List[int]] = []
            
        def encode_molecule(self, mol: Chem.Mol) -> Tensor:
            """Encode molecule to memory embedding."""
            fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_tensor = torch.tensor(
                [fp.GetBit(i) for i in range(2048)], 
                dtype=torch.float32
            )
            return self.mol_encoder(fp_tensor)
        
        def add(self, smiles: str, som_sites: List[int], 
                site_features: Optional[Tensor] = None):
            """Add molecule to memory."""
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return
            
            with torch.no_grad():
                emb = self.encode_molecule(mol)
                key = self.key_proj(emb)
                
                # Value encodes SoM site information
                if site_features is not None:
                    value = self.value_proj(site_features.mean(dim=0))
                else:
                    value = self.value_proj(emb)
            
            # Store
            if self.memory_keys is None:
                self.memory_keys = key.unsqueeze(0)
                self.memory_values = value.unsqueeze(0)
            else:
                self.memory_keys = torch.cat([self.memory_keys, key.unsqueeze(0)])
                self.memory_values = torch.cat([self.memory_values, value.unsqueeze(0)])
            
            self.memory_mols.append(mol)
            self.memory_sites.append(som_sites)
            
            # Capacity limit
            if len(self.memory_mols) > self.config.memory_capacity:
                self.memory_keys = self.memory_keys[1:]
                self.memory_values = self.memory_values[1:]
                self.memory_mols = self.memory_mols[1:]
                self.memory_sites = self.memory_sites[1:]
        
        def retrieve(self, h: Tensor, topk: Optional[int] = None) -> Tensor:
            """
            Continuous memory retrieval based on hidden states.
            
            This is called at EACH ODE step, allowing retrieval to
            evolve as the network "thinks".
            
            Args:
                h: [N, hidden_dim] - current hidden states
                topk: Number of memories to attend to
                
            Returns:
                retrieval: [N, memory_dim] - retrieved context per atom
            """
            if self.memory_keys is None or len(self.memory_mols) == 0:
                return torch.zeros(h.shape[0], self.config.memory_dim, device=h.device)
            
            topk = topk or self.config.memory_topk
            topk = min(topk, len(self.memory_mols))
            
            # Project hidden states to queries
            queries = self.query_proj(h)  # [N, memory_dim]
            
            # Move memory to same device
            keys = self.memory_keys.to(h.device)  # [M, memory_dim]
            values = self.memory_values.to(h.device)  # [M, memory_dim]
            
            # Compute attention scores
            # Use molecular-level pooled query for efficiency
            mol_query = queries.mean(dim=0, keepdim=True)  # [1, memory_dim]
            scores = torch.matmul(mol_query, keys.T).squeeze(0)  # [M]
            
            # Softmax over top-k
            topk_scores, topk_idx = torch.topk(scores, topk)
            attn_weights = F.softmax(topk_scores, dim=0)  # [topk]
            
            # Weighted sum of values
            topk_values = values[topk_idx]  # [topk, memory_dim]
            retrieved = (attn_weights.unsqueeze(-1) * topk_values).sum(dim=0)  # [memory_dim]
            
            # Broadcast to all atoms
            return retrieved.unsqueeze(0).expand(h.shape[0], -1)


# =============================================================================
# UNIFIED COGNITIVE ENGINE - Main Model
# =============================================================================

if TORCH_AVAILABLE:
    
    class UnifiedCognitiveEngine(nn.Module):
        """
        The Unified Cognitive Engine (UCE).
        
        A single integrated system where learning, reasoning, and wave
        dynamics are coupled and evolve together.
        
        Core equation:
            τ_i · dS_i/dt = F(S_i, S_neighbors, Memory, Perturbation)
            
        Where S_i = [h_i, ρ_i, m_i] is the unified state.
        """
        
        def __init__(self, config: Optional[UCEConfig] = None):
            super().__init__()
            self.config = config or UCEConfig()
            
            # Atom encoding
            self.atom_embedding = nn.Embedding(100, self.config.atom_feature_dim)
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.config.atom_feature_dim + 16, self.config.atom_feature_dim),
                nn.LayerNorm(self.config.atom_feature_dim),
                nn.SiLU(),
            )
            
            # Project to initial unified state
            self.to_hidden = nn.Linear(self.config.atom_feature_dim, self.config.hidden_dim)
            self.to_wave = nn.Linear(self.config.atom_feature_dim, self.config.wave_dim)
            self.to_memory = nn.Linear(self.config.atom_feature_dim, self.config.memory_dim)
            
            # Coupled dynamics
            self.dynamics = CoupledDynamics(self.config)
            
            # Memory system
            if RDKIT_AVAILABLE:
                self.memory = ContinuousMemory(self.config)
            else:
                self.memory = None
            
            # Perturbation model (learns attack patterns)
            self.perturbation_net = nn.Sequential(
                nn.Linear(self.config.atom_feature_dim, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            
            # Output head
            self.output_net = nn.Sequential(
                nn.Linear(self.config.state_dim, self.config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(self.config.hidden_dim // 2, 1),
            )
            
            self.dropout = nn.Dropout(self.config.dropout)
            
        def encode_molecule(self, mol: Chem.Mol) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
            """
            Encode molecule to initial features.
            
            Returns:
                atom_features: [N, atom_feature_dim]
                edge_index: [2, E]
                edge_attr: [E, edge_dim] or None
            """
            # Atomic embeddings
            atomic_nums = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
            embeddings = self.atom_embedding(atomic_nums)
            
            # Additional features
            extra = []
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
                # Electronegativity proxy
                en = {6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 16: 2.58, 17: 3.16}.get(
                    atom.GetAtomicNum(), 2.5
                )
                feat.append(en / 4.0)
                
                # Pad
                feat = feat + [0.0] * (16 - len(feat))
                extra.append(feat)
            
            extra_tensor = torch.tensor(extra, dtype=torch.float32)
            atom_features = self.feature_encoder(torch.cat([embeddings, extra_tensor], dim=-1))
            
            # Build edges
            src, dst = [], []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                src.extend([i, j])
                dst.extend([j, i])
            
            if src:
                edge_index = torch.tensor([src, dst], dtype=torch.long)
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
            
            return atom_features, edge_index, None
        
        def init_state(self, atom_features: Tensor) -> UnifiedState:
            """Initialize unified state from atom features."""
            h = self.to_hidden(atom_features)
            rho = self.to_wave(atom_features)
            m = self.to_memory(atom_features)
            return UnifiedState(h, rho, m)
        
        def ode_step_rk4(self, state: UnifiedState, edge_index: Tensor,
                         edge_attr: Optional[Tensor], perturbation: Tensor,
                         memory_retrieval: Tensor, dt: float) -> Tuple[UnifiedState, Tensor]:
            """Single RK4 ODE step."""
            
            def compute_derivative(s: UnifiedState) -> Tuple[UnifiedState, Tensor]:
                return self.dynamics(s, edge_index, edge_attr, perturbation, memory_retrieval)
            
            # RK4 stages
            k1, tau = compute_derivative(state)
            
            s2 = UnifiedState(
                state.h + 0.5 * dt * k1.h,
                state.rho + 0.5 * dt * k1.rho,
                state.m + 0.5 * dt * k1.m,
            )
            k2, _ = compute_derivative(s2)
            
            s3 = UnifiedState(
                state.h + 0.5 * dt * k2.h,
                state.rho + 0.5 * dt * k2.rho,
                state.m + 0.5 * dt * k2.m,
            )
            k3, _ = compute_derivative(s3)
            
            s4 = UnifiedState(
                state.h + dt * k3.h,
                state.rho + dt * k3.rho,
                state.m + dt * k3.m,
            )
            k4, _ = compute_derivative(s4)
            
            # Combine
            new_state = UnifiedState(
                state.h + (dt / 6.0) * (k1.h + 2*k2.h + 2*k3.h + k4.h),
                state.rho + (dt / 6.0) * (k1.rho + 2*k2.rho + 2*k3.rho + k4.rho),
                state.m + (dt / 6.0) * (k1.m + 2*k2.m + 2*k3.m + k4.m),
            )
            
            return new_state, tau
        
        def forward(self, smiles: str, labels: Optional[Tensor] = None,
                   return_trajectory: bool = False) -> Dict[str, Any]:
            """
            Forward pass through the unified system.
            
            Args:
                smiles: Molecule SMILES
                labels: [N] ground truth SoM labels (optional)
                return_trajectory: Whether to return state evolution
                
            Returns:
                Dict with scores, states, and diagnostics
            """
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            # Encode molecule
            atom_features, edge_index, edge_attr = self.encode_molecule(mol)
            N = atom_features.shape[0]
            
            # Initialize unified state
            state = self.init_state(atom_features)
            
            # Perturbation (learned attack susceptibility)
            perturbation = self.perturbation_net(atom_features).squeeze(-1)
            perturbation = perturbation * self.config.wave_perturbation_strength
            
            # Evolution tracking
            trajectory = [state] if return_trajectory else None
            tau_history = []
            total_steps = 0
            
            # Evolve until convergence or max steps
            prev_norm = state.norm()
            
            for step in range(self.config.max_ode_steps):
                # Continuous memory retrieval (evolves with computation!)
                if self.memory is not None:
                    memory_retrieval = self.memory.retrieve(state.h)
                else:
                    memory_retrieval = torch.zeros(N, self.config.memory_dim)
                
                # ODE step
                state, tau = self.ode_step_rk4(
                    state, edge_index, edge_attr, 
                    perturbation, memory_retrieval, 
                    self.config.dt
                )
                
                tau_history.append(tau)
                total_steps += 1
                
                if return_trajectory:
                    trajectory.append(state)
                
                # Check convergence
                curr_norm = state.norm()
                delta = (curr_norm - prev_norm).abs().max()
                if delta < self.config.convergence_threshold:
                    break
                prev_norm = curr_norm
            
            # Final unified state
            final_S = state.flatten()
            
            # Output scores
            scores = self.output_net(final_S).squeeze(-1)
            
            # Compute loss if labels provided
            loss = None
            if labels is not None:
                # Candidate mask (carbons with H)
                candidate_mask = torch.tensor([
                    a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0
                    for a in mol.GetAtoms()
                ], dtype=torch.bool)
                
                loss = self._ranking_loss(scores, labels, candidate_mask)
            
            return {
                'scores': scores,
                'final_state': state,
                'loss': loss,
                'num_steps': total_steps,
                'tau_history': tau_history,
                'trajectory': trajectory,
            }
        
        def _ranking_loss(self, scores: Tensor, labels: Tensor, 
                          mask: Tensor) -> Tensor:
            """ListMLE ranking loss."""
            if not mask.any():
                return scores.new_zeros(1)
            
            valid_scores = scores[mask]
            valid_labels = labels[mask]
            
            if valid_labels.sum() == 0:
                return scores.new_zeros(1)
            
            # Sort by ground truth
            sorted_idx = torch.argsort(valid_labels, descending=True)
            sorted_scores = valid_scores[sorted_idx]
            
            # ListMLE
            n = len(sorted_scores)
            loss = 0.0
            for i in range(n):
                remaining = sorted_scores[i:]
                loss = loss - sorted_scores[i] + torch.logsumexp(remaining, dim=0)
            
            return loss / n
        
        def predict(self, smiles: str, top_k: int = 3) -> List[Tuple[int, float]]:
            """
            Predict top-k SoM sites.
            
            Returns:
                List of (atom_idx, score) tuples
            """
            self.eval()
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            with torch.no_grad():
                output = self.forward(smiles)
            
            scores = output['scores']
            
            # Filter to candidates
            candidate_mask = torch.tensor([
                a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0
                for a in mol.GetAtoms()
            ], dtype=torch.bool)
            
            candidate_scores = scores.clone()
            candidate_scores[~candidate_mask] = float('-inf')
            
            top_k = min(top_k, candidate_mask.sum().item())
            if top_k == 0:
                return []
            
            top_idx = torch.topk(candidate_scores, top_k).indices
            
            return [(idx.item(), scores[idx].item()) for idx in top_idx]
        
        def add_to_memory(self, smiles: str, som_sites: List[int]):
            """Add molecule to memory for future reasoning."""
            if self.memory is not None:
                self.memory.add(smiles, som_sites)
        
        def get_diagnostics(self, smiles: str) -> Dict[str, Any]:
            """
            Get detailed diagnostics for a prediction.
            
            Returns state evolution, tau values, etc.
            """
            self.eval()
            with torch.no_grad():
                output = self.forward(smiles, return_trajectory=True)
            
            trajectory = output['trajectory']
            
            return {
                'num_steps': output['num_steps'],
                'tau_mean_per_step': [t.mean().item() for t in output['tau_history']],
                'h_evolution': [s.h.norm(dim=-1).mean().item() for s in trajectory],
                'rho_evolution': [s.rho.norm(dim=-1).mean().item() for s in trajectory],
                'm_evolution': [s.m.norm(dim=-1).mean().item() for s in trajectory],
                'final_scores': output['scores'].tolist(),
            }


# =============================================================================
# FACTORY
# =============================================================================

def create_uce(hidden_dim: int = 96, wave_dim: int = 32, memory_dim: int = 32,
               **kwargs) -> "UnifiedCognitiveEngine":
    """
    Create a Unified Cognitive Engine.
    
    Args:
        hidden_dim: Hidden representation dimension
        wave_dim: Electron density state dimension
        memory_dim: Reasoning/memory state dimension
        **kwargs: Additional config parameters
        
    Returns:
        UCE model
    """
    config = UCEConfig(
        hidden_dim=hidden_dim,
        wave_dim=wave_dim,
        memory_dim=memory_dim,
        **kwargs
    )
    return UnifiedCognitiveEngine(config)
