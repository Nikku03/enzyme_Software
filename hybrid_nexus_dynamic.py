"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     HYBRID NEXUS-DYNAMIC STATE ARCHITECTURE                                  ║
║                                                                              ║
║     Combines:                                                                ║
║     1. Dynamic State Discovery (enzyme conformational states)               ║
║     2. Liquid Neural Network (continuous-time dynamics)                      ║
║     3. Hyperbolic Memory + Analogical Reasoning                             ║
║                                                                              ║
║     Key insight: Dynamic states handle enzyme plasticity,                    ║
║                  LNN handles molecular dynamics,                             ║
║                  Memory provides transfer from similar molecules             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# LIQUID NEURAL NETWORK CELL
# ═══════════════════════════════════════════════════════════════════════════════

class LiquidCell(nn.Module):
    """
    Liquid Time-Constant Cell with adaptive dynamics.
    
    The key insight: molecular reactivity evolves over continuous "reaction time"
    as the molecule explores the enzyme pocket. The liquid cell models this.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)
        
        # Adaptive time constant (molecule-dependent reaction dynamics)
        self.tau_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Gating (which atoms are "active" at this timestep)
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
        
        Args:
            x: Input features [B, N, input_dim]
            h: Hidden state [B, N, hidden_dim]
            dt: Integration timestep
            n_steps: Number of integration steps
            
        Returns:
            Updated hidden state [B, N, hidden_dim]
        """
        for _ in range(n_steps):
            # Adaptive time constant
            tau = 0.5 + self.tau_net(h)  # τ ∈ [0.5, 1.5]
            
            # Gate
            gate = self.gate(h)
            
            # Dynamics
            pre_act = self.W_in(x) + self.W_rec(h)
            f = torch.tanh(pre_act)
            
            # Update with Euler integration
            dh = (-h + gate * f) / tau
            h = h + dt * dh
        
        return h


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC STATE BANK (simplified from dynamic_states.py)
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicStateBank(nn.Module):
    """
    Learnable enzyme conformational states that grow during training.
    Each state represents a distinct binding mode of CYP3A4.
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        max_states: int = 32,
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
        self.n_active_states = min_states
        
        # Context encoder for state matching
        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim + 3, state_dim),  # mol_emb + som_pos
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )
        
        # Pocket deformation per state
        self.pocket_deformations = nn.Parameter(
            torch.randn(max_states, 100, 3) * 0.1  # 100 pocket atoms, 3D offset
        )
    
    @property
    def num_active_states(self) -> int:
        return int(self.state_active.sum().item())
    
    def get_active_states(self) -> torch.Tensor:
        return self.state_embeddings[self.state_active]
    
    def find_or_create_state(
        self,
        context_emb: torch.Tensor,
        training: bool = True,
    ) -> int:
        """Find matching state or create new one."""
        active_states = self.get_active_states()
        
        # Compute similarities
        context_norm = F.normalize(context_emb.unsqueeze(0), dim=-1)
        states_norm = F.normalize(active_states, dim=-1)
        sims = torch.mm(context_norm, states_norm.t()).squeeze(0)
        
        max_sim, best_idx = sims.max(dim=0)
        
        # If similar enough, return existing
        if max_sim >= self.similarity_threshold:
            return best_idx.item()
        
        # Otherwise create new state if possible
        if training and self.num_active_states < self.max_states:
            inactive = (~self.state_active).nonzero(as_tuple=True)[0]
            if len(inactive) > 0:
                new_idx = inactive[0].item()
                with torch.no_grad():
                    self.state_embeddings[new_idx] = context_emb.detach()
                self.state_active[new_idx] = True
                self.n_active_states += 1
                return new_idx
        
        return best_idx.item()
    
    def forward(
        self,
        mol_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute state probabilities for molecules.
        
        Args:
            mol_embedding: [B, state_dim]
            
        Returns:
            state_probs: [B, n_active]
            active_indices: indices of active states
        """
        active_states = self.get_active_states()
        active_indices = self.state_active.nonzero(as_tuple=True)[0]
        
        # Similarity-based selection
        mol_norm = F.normalize(mol_embedding, dim=-1)
        states_norm = F.normalize(active_states, dim=-1)
        
        logits = torch.mm(mol_norm, states_norm.t()) / 0.1  # Temperature
        probs = F.softmax(logits, dim=-1)
        
        return probs, active_indices


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERBOLIC MEMORY (simplified)
# ═══════════════════════════════════════════════════════════════════════════════

class HyperbolicMemory(nn.Module):
    """
    Memory bank in hyperbolic space for analogical reasoning.
    Stores (molecule, SoM) pairs and retrieves similar molecules.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        curvature: float = 1.0,
        max_entries: int = 1000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.max_entries = max_entries
        
        # Memory storage
        self.register_buffer('memory_embeddings', torch.zeros(0, embedding_dim))
        self.register_buffer('memory_som_masks', torch.zeros(0, 200))  # Max 200 atoms
        self.n_entries = 0
        
        # Projection to hyperbolic
        self.project = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),  # Keep in [-1, 1] for Poincaré ball
        )
    
    def _to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """Project to Poincaré ball with radius < 1."""
        x = self.project(x)
        norm = x.norm(dim=-1, keepdim=True)
        # Clamp to radius 0.95
        scale = torch.where(norm > 0.95, 0.95 / norm, torch.ones_like(norm))
        return x * scale
    
    def _hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute hyperbolic distance in Poincaré ball."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [B, M, D]
        diff_norm_sq = (diff ** 2).sum(-1)
        
        x_norm_sq = (x ** 2).sum(-1, keepdim=True)
        y_norm_sq = (y ** 2).sum(-1).unsqueeze(0)
        
        # Poincaré distance formula
        num = diff_norm_sq
        denom = (1 - x_norm_sq) * (1 - y_norm_sq)
        
        arg = 1 + 2 * num / (denom + 1e-8)
        return torch.acosh(arg.clamp(min=1.0 + 1e-6))
    
    def store(
        self,
        mol_embedding: torch.Tensor,
        som_mask: torch.Tensor,
    ):
        """Store molecule in memory."""
        if self.n_entries >= self.max_entries:
            # FIFO eviction
            self.memory_embeddings = self.memory_embeddings[1:]
            self.memory_som_masks = self.memory_som_masks[1:]
            self.n_entries -= 1
        
        # Add new entry
        emb = mol_embedding.detach().cpu()
        mask = som_mask.detach().cpu()
        
        if self.memory_embeddings.numel() == 0:
            self.memory_embeddings = emb.unsqueeze(0)
            self.memory_som_masks = mask.unsqueeze(0)
        else:
            self.memory_embeddings = torch.cat([
                self.memory_embeddings,
                emb.unsqueeze(0)
            ], dim=0)
            self.memory_som_masks = torch.cat([
                self.memory_som_masks,
                mask.unsqueeze(0)
            ], dim=0)
        
        self.n_entries += 1
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        k: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve k most similar molecules.
        
        Returns:
            weights: [B, k] attention weights
            som_hints: [B, k, max_atoms] SoM masks of retrieved molecules
        """
        if self.n_entries == 0:
            B = query_embedding.shape[0]
            return (
                torch.zeros(B, k, device=query_embedding.device),
                torch.zeros(B, k, 200, device=query_embedding.device)
            )
        
        # Project to hyperbolic
        query_hyp = self._to_poincare(query_embedding)
        memory_hyp = self._to_poincare(
            self.memory_embeddings.to(query_embedding.device)
        )
        
        # Compute distances
        dists = self._hyperbolic_distance(query_hyp, memory_hyp)  # [B, M]
        
        # Get top-k closest
        k = min(k, self.n_entries)
        top_dists, top_indices = dists.topk(k, dim=-1, largest=False)
        
        # Convert distances to weights
        weights = F.softmax(-top_dists, dim=-1)
        
        # Get SoM hints
        som_hints = self.memory_som_masks.to(query_embedding.device)[top_indices]
        
        return weights, som_hints


# ═══════════════════════════════════════════════════════════════════════════════
# ANALOGICAL FUSION HEAD
# ═══════════════════════════════════════════════════════════════════════════════

class AnalogicalFusion(nn.Module):
    """
    Fuses first-principles predictions with analogical hints.
    
    Key insight: For novel molecules, trust physics.
                 For similar molecules, trust memory.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        # Physics path
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Analogy path
        self.analogy_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for memory hint
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Learnable gate between paths
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        atom_features: torch.Tensor,
        memory_hints: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            atom_features: [B, N, hidden_dim]
            memory_hints: [B, N] soft SoM hints from memory
            
        Returns:
            scores: [B, N] final SoM scores
            physics_scores: [B, N] physics-only scores
            analogy_scores: [B, N] analogy-only scores
        """
        # Physics path
        physics_scores = self.physics_head(atom_features).squeeze(-1)
        
        # Analogy path (with memory hint)
        hints_expanded = memory_hints.unsqueeze(-1)
        analogy_input = torch.cat([atom_features, hints_expanded], dim=-1)
        analogy_scores = self.analogy_head(analogy_input).squeeze(-1)
        
        # Adaptive gate
        gate_input = torch.cat([atom_features, hints_expanded], dim=-1)
        gate = self.gate(gate_input).squeeze(-1)  # [B, N]
        
        # Fuse
        scores = gate * analogy_scores + (1 - gate) * physics_scores
        
        return scores, physics_scores, analogy_scores


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HYBRID MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class HybridNexusDynamic(nn.Module):
    """
    Complete hybrid architecture combining:
    1. Dynamic enzyme states
    2. Liquid neural dynamics
    3. Hyperbolic memory + analogical fusion
    
    Flow:
    1. Encode molecule → mol_embedding
    2. Select enzyme state(s) based on mol_embedding
    3. Run liquid dynamics to evolve atom representations
    4. Retrieve similar molecules from memory
    5. Fuse physics + analogy predictions
    """
    
    def __init__(
        self,
        mol_dim: int = 128,
        hidden_dim: int = 64,
        max_states: int = 32,
        similarity_threshold: float = 0.7,
        n_liquid_steps: int = 5,
        memory_k: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_liquid_steps = n_liquid_steps
        self.memory_k = memory_k
        
        # Molecule encoder
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Dynamic state bank
        self.state_bank = DynamicStateBank(
            state_dim=hidden_dim,
            max_states=max_states,
            similarity_threshold=similarity_threshold,
        )
        
        # Liquid cell for temporal dynamics
        self.liquid_cell = LiquidCell(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )
        
        # Hyperbolic memory
        self.memory = HyperbolicMemory(
            embedding_dim=hidden_dim,
        )
        
        # Analogical fusion head
        self.fusion = AnalogicalFusion(hidden_dim=hidden_dim)
        
        # Pocket encoder
        self.pocket_encoder = nn.Linear(14, hidden_dim)  # 14 = pocket feature dim
        
        # Cross-attention mol ↔ pocket
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        
        # Chemistry prior (alpha-het, benzylic, etc.)
        self.chem_prior = nn.Sequential(
            nn.Linear(mol_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        
        # Fe distance scorer
        self.fe_coords = nn.Parameter(torch.tensor([54.95, 77.69, 10.64]))
    
    def compute_fe_scores(
        self,
        mol_coords: torch.Tensor,
        state_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Fe-distance based scores.
        
        Closer to Fe = higher reactivity (for oxidation).
        Different states may have different optimal distances.
        """
        B, N, _ = mol_coords.shape
        
        # Distance to Fe
        fe = self.fe_coords.to(mol_coords.device)
        dists = torch.norm(mol_coords - fe.unsqueeze(0).unsqueeze(0), dim=-1)
        
        # Gaussian activation around optimal distance
        optimal_dist = 4.0  # Typical Fe-C distance for oxidation
        sigma = 2.0
        scores = torch.exp(-(dists - optimal_dist) ** 2 / (2 * sigma ** 2))
        
        return scores
    
    def forward(
        self,
        mol_features: torch.Tensor,
        mol_coords: torch.Tensor,
        pocket_features: torch.Tensor,
        som_mask: torch.Tensor = None,
        valid_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            mol_features: [B, N, mol_dim] atom features
            mol_coords: [B, N, 3] atom coordinates
            pocket_features: [N_pocket, 14] pocket atom features
            som_mask: [B, N] ground truth SoM (for training memory)
            valid_mask: [B, N] which atoms are valid
            
        Returns:
            dict with scores, state_probs, etc.
        """
        B, N, mol_dim = mol_features.shape
        device = mol_features.device
        
        # 1. Encode molecule
        mol_encoded = self.mol_encoder(mol_features)  # [B, N, hidden]
        mol_global = mol_encoded.mean(dim=1)  # [B, hidden]
        
        # 2. Dynamic state selection
        if self.training and som_mask is not None:
            for b in range(B):
                som_indices = torch.where(som_mask[b] > 0)[0]
                if len(som_indices) > 0:
                    som_pos = mol_coords[b, som_indices].mean(dim=0)
                    context = torch.cat([mol_global[b], som_pos], dim=-1)
                    context_emb = self.state_bank.context_encoder(context)
                    self.state_bank.find_or_create_state(context_emb, training=True)
        
        state_probs, active_indices = self.state_bank(mol_global)
        
        # 3. Liquid dynamics
        h = mol_encoded
        h = self.liquid_cell(mol_encoded, h, n_steps=self.n_liquid_steps)
        
        # 4. Cross-attention with pocket
        pocket_encoded = self.pocket_encoder(pocket_features)  # [N_pocket, hidden]
        pocket_batch = pocket_encoded.unsqueeze(0).expand(B, -1, -1)
        
        h_attended, _ = self.cross_attn(h, pocket_batch, pocket_batch)
        h = h + h_attended  # Residual
        
        # 5. Memory retrieval
        weights, som_hints = self.memory.retrieve(mol_global, k=self.memory_k)
        
        # Aggregate hints
        memory_hint = (weights.unsqueeze(-1) * som_hints[:, :, :N]).sum(dim=1)  # [B, N]
        
        # 6. Store in memory (training only)
        if self.training and som_mask is not None:
            for b in range(B):
                padded_mask = F.pad(som_mask[b], (0, 200 - N))
                self.memory.store(mol_global[b], padded_mask)
        
        # 7. Analogical fusion
        scores, physics_scores, analogy_scores = self.fusion(h, memory_hint)
        
        # 8. Add chemistry prior and Fe distance
        chem_scores = self.chem_prior(mol_features).squeeze(-1)
        fe_scores = self.compute_fe_scores(mol_coords, state_probs)
        
        final_scores = scores + 0.3 * chem_scores + 0.5 * fe_scores
        
        return {
            'scores': final_scores,
            'final_scores': final_scores,
            'physics_scores': physics_scores,
            'analogy_scores': analogy_scores,
            'state_probs': state_probs,
            'n_active_states': self.state_bank.num_active_states,
            'active_indices': active_indices,
            'memory_hint': memory_hint,
            'fe_scores': fe_scores,
            'chem_scores': chem_scores,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class HybridLoss(nn.Module):
    """
    Loss combining:
    1. SoM ranking loss (main task)
    2. Physics-analogy consistency
    3. Memory utilization
    """
    
    def __init__(
        self,
        physics_weight: float = 0.3,
        consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.physics_weight = physics_weight
        self.consistency_weight = consistency_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        som_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.
        
        Args:
            outputs: model outputs
            som_mask: [B, N] ground truth SoM
            valid_mask: [B, N] valid atoms
            
        Returns:
            loss: scalar
            metrics: dict of loss components
        """
        scores = outputs['final_scores']
        physics_scores = outputs['physics_scores']
        analogy_scores = outputs['analogy_scores']
        
        B, N = scores.shape
        device = scores.device
        
        # Mask invalid atoms
        scores = scores.masked_fill(~valid_mask, float('-inf'))
        physics_scores = physics_scores.masked_fill(~valid_mask, float('-inf'))
        
        # Main ranking loss (cross-entropy over atoms)
        log_probs = F.log_softmax(scores, dim=-1)
        som_probs = som_mask / (som_mask.sum(dim=-1, keepdim=True) + 1e-8)
        main_loss = -(som_probs * log_probs).sum(dim=-1).mean()
        
        # Physics path loss
        physics_log_probs = F.log_softmax(physics_scores, dim=-1)
        physics_loss = -(som_probs * physics_log_probs).sum(dim=-1).mean()
        
        # Consistency loss (physics and analogy should agree on ranking)
        physics_ranking = F.softmax(physics_scores, dim=-1)
        analogy_ranking = F.softmax(analogy_scores, dim=-1)
        consistency_loss = F.kl_div(
            analogy_ranking.log(),
            physics_ranking.detach(),
            reduction='batchmean'
        )
        
        # Total
        loss = (
            main_loss
            + self.physics_weight * physics_loss
            + self.consistency_weight * consistency_loss
        )
        
        metrics = {
            'main_loss': main_loss.item(),
            'physics_loss': physics_loss.item(),
            'consistency_loss': consistency_loss.item(),
        }
        
        return loss, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing HybridNexusDynamic...")
    
    model = HybridNexusDynamic(
        mol_dim=128,
        hidden_dim=64,
        max_states=32,
        similarity_threshold=0.7,
    )
    
    B, N = 4, 30
    mol_features = torch.randn(B, N, 128)
    mol_coords = torch.randn(B, N, 3) * 10
    pocket_features = torch.randn(100, 14)
    som_mask = torch.zeros(B, N)
    for b in range(B):
        som_mask[b, b + 5] = 1
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    model.train()
    
    # Run a few batches to populate memory
    for i in range(5):
        outputs = model(mol_features, mol_coords, pocket_features, som_mask, valid_mask)
    
    print(f"Active states: {outputs['n_active_states']}")
    print(f"Memory entries: {model.memory.n_entries}")
    print(f"Scores shape: {outputs['scores'].shape}")
    print(f"Physics scores: {outputs['physics_scores'][:2, :5]}")
    print(f"Analogy scores: {outputs['analogy_scores'][:2, :5]}")
    print(f"Memory hint: {outputs['memory_hint'][:2, :5]}")
    
    # Test loss
    loss_fn = HybridLoss()
    loss, metrics = loss_fn(outputs, som_mask, valid_mask)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")
    
    print("\n✓ HybridNexusDynamic test passed!")
