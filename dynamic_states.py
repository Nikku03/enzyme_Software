#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DYNAMIC STATE DISCOVERY                                                      ║
║  ════════════════════════════════════════════════════════════════════════════ ║
║                                                                               ║
║  Instead of fixed n_states, discover pocket conformations dynamically:        ║
║  1. Start with empty state bank                                               ║
║  2. For each (molecule, SoM) pair, compute what state explains it             ║
║  3. If similar state exists: reinforce it                                     ║
║     Else: create new state                                                    ║
║  4. Periodically prune/merge similar states                                   ║
║                                                                               ║
║  This lets the data tell us how many states exist.                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict


class DynamicStateBank(nn.Module):
    """
    A memory bank that discovers pocket states during training.
    
    States are stored as embeddings. New states are added when
    a molecule-SoM pair doesn't match existing states well enough.
    """
    
    def __init__(
        self,
        state_dim: int = 64,
        pocket_dim: int = 14,
        max_states: int = 64,
        similarity_threshold: float = 0.7,
        min_states: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.pocket_dim = pocket_dim
        self.max_states = max_states
        self.similarity_threshold = similarity_threshold
        self.min_states = min_states
        
        # State bank - starts with min_states, grows dynamically
        # Using a buffer so it's saved with model but not a parameter
        self.register_buffer('state_embeddings', torch.randn(max_states, state_dim))
        self.register_buffer('state_active', torch.zeros(max_states, dtype=torch.bool))
        self.register_buffer('state_usage_count', torch.zeros(max_states))
        self.register_buffer('n_active_states', torch.tensor(0))
        
        # Initialize with min_states
        self.state_active[:min_states] = True
        self.n_active_states.fill_(min_states)
        
        # Reactive centers for each state (learnable)
        self.reactive_centers = nn.Parameter(torch.zeros(max_states, 3))
        self.reactive_radii = nn.Parameter(torch.ones(max_states, 1) * 4.0)
        
        # State deformation networks (one per possible state)
        self.state_deformations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pocket_dim, state_dim),
                nn.SiLU(),
                nn.Linear(state_dim, pocket_dim),
            )
            for _ in range(max_states)
        ])
        
        # Network to encode molecule-SoM context into state space
        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim + 3, state_dim),  # mol_embedding + som_position
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        
        # State selector (mol → state logits)
        self.state_selector = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, max_states),
        )
        
        self._init_states()
    
    def _init_states(self):
        """Initialize starting states with diverse conformations."""
        # Spread initial states around Fe
        for i in range(self.min_states):
            angle = 2 * np.pi * i / self.min_states
            self.reactive_centers.data[i] = torch.tensor([
                4.0 * np.cos(angle),
                4.0 * np.sin(angle),
                0.0,
            ])
        
        # Initialize state embeddings to be distinct
        nn.init.orthogonal_(self.state_embeddings[:self.min_states])
    
    @property
    def num_active_states(self) -> int:
        return int(self.n_active_states.item())
    
    def get_active_states(self) -> torch.Tensor:
        """Get embeddings of active states only."""
        return self.state_embeddings[self.state_active]
    
    def get_active_indices(self) -> torch.Tensor:
        """Get indices of active states."""
        return torch.where(self.state_active)[0]
    
    def compute_state_similarity(
        self,
        query: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute similarity between query and all active states.
        
        Args:
            query: (B, state_dim) query embeddings
            
        Returns:
            similarities: (B, n_active) cosine similarities
            indices: (n_active,) indices of active states
        """
        active_states = self.get_active_states()  # (n_active, state_dim)
        active_indices = self.get_active_indices()
        
        # Normalize for cosine similarity
        query_norm = F.normalize(query, dim=-1)
        states_norm = F.normalize(active_states, dim=-1)
        
        # (B, state_dim) @ (state_dim, n_active) -> (B, n_active)
        similarities = torch.mm(query_norm, states_norm.t())
        
        return similarities, active_indices
    
    def find_or_create_state(
        self,
        context_embedding: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Find matching state or create new one.
        
        Args:
            context_embedding: (state_dim,) embedding from (mol, SoM) context
            training: whether we're in training mode
            
        Returns:
            state_idx: index of matched/created state
            similarity: how well it matched
            is_new: whether a new state was created
        """
        context_embedding = context_embedding.unsqueeze(0)  # (1, state_dim)
        
        similarities, active_indices = self.compute_state_similarity(context_embedding)
        similarities = similarities.squeeze(0)  # (n_active,)
        
        best_sim, best_local_idx = similarities.max(dim=0)
        best_idx = active_indices[best_local_idx]
        
        # Check if we should create a new state
        if training and best_sim < self.similarity_threshold and self.num_active_states < self.max_states:
            # Create new state
            new_idx = self._create_new_state(context_embedding.squeeze(0))
            return new_idx, torch.tensor(1.0, device=context_embedding.device), True
        
        # Use existing state
        self.state_usage_count[best_idx] += 1
        return best_idx, best_sim, False
    
    def _create_new_state(self, embedding: torch.Tensor) -> int:
        """Create a new state from the given embedding."""
        # Find first inactive slot
        inactive_indices = torch.where(~self.state_active)[0]
        if len(inactive_indices) == 0:
            return -1  # No room
        
        new_idx = inactive_indices[0].item()
        
        # Initialize new state
        self.state_embeddings[new_idx] = embedding.detach()
        self.state_active[new_idx] = True
        self.state_usage_count[new_idx] = 1
        self.n_active_states += 1
        
        # Initialize reactive center (random direction from Fe)
        angle = torch.rand(1).item() * 2 * np.pi
        phi = torch.rand(1).item() * np.pi - np.pi/2
        self.reactive_centers.data[new_idx] = torch.tensor([
            4.0 * np.cos(angle) * np.cos(phi),
            4.0 * np.sin(angle) * np.cos(phi),
            4.0 * np.sin(phi),
        ])
        
        return new_idx
    
    def select_states(
        self,
        mol_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select states based on molecular embedding.
        
        Args:
            mol_embedding: (B, state_dim)
            
        Returns:
            probs: (B, n_active) probabilities over active states
            indices: (n_active,) indices of active states
        """
        # Get logits for all states
        all_logits = self.state_selector(mol_embedding)  # (B, max_states)
        
        # Mask inactive states
        active_mask = self.state_active.unsqueeze(0)  # (1, max_states)
        masked_logits = all_logits.masked_fill(~active_mask, float('-inf'))
        
        # Softmax over active states only
        probs = F.softmax(masked_logits, dim=-1)
        
        # Get active indices
        active_indices = self.get_active_indices()
        
        # Extract only active probabilities
        active_probs = probs[:, self.state_active]  # (B, n_active)
        
        return active_probs, active_indices
    
    def get_state_pocket(
        self,
        state_idx: int,
        base_pocket: torch.Tensor,
    ) -> torch.Tensor:
        """Get deformed pocket for a specific state."""
        deformation = self.state_deformations[state_idx](base_pocket)
        return base_pocket + 0.1 * deformation
    
    def forward(
        self,
        mol_embedding: torch.Tensor,
        base_pocket: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute state-weighted pocket representation.
        
        Args:
            mol_embedding: (B, state_dim)
            base_pocket: (N_pocket, pocket_dim)
            
        Returns:
            weighted_pocket: (B, N_pocket, pocket_dim)
            state_probs: (B, max_states) - full size, inactive are 0
            active_indices: indices of active states
        """
        B = mol_embedding.shape[0]
        N_pocket = base_pocket.shape[0]
        device = mol_embedding.device
        
        # Get state probabilities
        active_probs, active_indices = self.select_states(mol_embedding)
        n_active = len(active_indices)
        
        # Compute deformed pocket for each active state
        state_pockets = []
        for idx in active_indices:
            deformed = self.get_state_pocket(idx.item(), base_pocket)
            state_pockets.append(deformed)
        
        state_pockets = torch.stack(state_pockets)  # (n_active, N_pocket, pocket_dim)
        
        # Weight by state probabilities
        # (B, n_active, 1, 1) * (1, n_active, N_pocket, pocket_dim)
        weighted = active_probs.unsqueeze(-1).unsqueeze(-1) * state_pockets.unsqueeze(0)
        weighted_pocket = weighted.sum(dim=1)  # (B, N_pocket, pocket_dim)
        
        # Create full-size probs tensor (for compatibility)
        full_probs = torch.zeros(B, self.max_states, device=device)
        full_probs[:, self.state_active] = active_probs
        
        return weighted_pocket, full_probs, active_indices
    
    def prune_unused_states(self, min_usage: int = 5):
        """Remove states that haven't been used enough."""
        for i in range(self.max_states):
            if self.state_active[i] and self.state_usage_count[i] < min_usage:
                if self.num_active_states > self.min_states:
                    self.state_active[i] = False
                    self.n_active_states -= 1
    
    def merge_similar_states(self, threshold: float = 0.95):
        """Merge states that are too similar."""
        active_indices = self.get_active_indices()
        if len(active_indices) <= self.min_states:
            return
        
        active_states = self.get_active_states()
        states_norm = F.normalize(active_states, dim=-1)
        
        # Compute pairwise similarities
        sims = torch.mm(states_norm, states_norm.t())
        
        # Find pairs above threshold (excluding diagonal)
        sims.fill_diagonal_(0)
        
        for i in range(len(active_indices)):
            for j in range(i + 1, len(active_indices)):
                if sims[i, j] > threshold and self.num_active_states > self.min_states:
                    # Merge j into i
                    idx_i = active_indices[i]
                    idx_j = active_indices[j]
                    
                    # Average embeddings
                    self.state_embeddings[idx_i] = (
                        self.state_embeddings[idx_i] + self.state_embeddings[idx_j]
                    ) / 2
                    
                    # Deactivate j
                    self.state_active[idx_j] = False
                    self.state_usage_count[idx_i] += self.state_usage_count[idx_j]
                    self.n_active_states -= 1


class DynamicReverseGeometryEngine(nn.Module):
    """
    Reverse geometry engine with dynamic state discovery.
    """
    
    def __init__(
        self,
        mol_dim: int = 128,
        state_dim: int = 64,
        pocket_dim: int = 14,
        max_states: int = 64,
        similarity_threshold: float = 0.7,
        fe_position: torch.Tensor = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.pocket_dim = pocket_dim
        
        # Fe position (default from 1W0E.pdb)
        if fe_position is None:
            fe_position = torch.tensor([54.949, 77.690, 10.642])
        self.register_buffer('fe_position', fe_position)
        
        # Dynamic state bank
        self.state_bank = DynamicStateBank(
            state_dim=state_dim,
            pocket_dim=pocket_dim,
            max_states=max_states,
            similarity_threshold=similarity_threshold,
        )
        
        # Molecule encoder
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
        )
        
        # Pocket projection
        self.pocket_proj = nn.Linear(pocket_dim, state_dim)
        
        # Cross-attention: molecule attends to pocket
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        
        # Atom scorer
        self.atom_scorer = nn.Sequential(
            nn.Linear(state_dim * 2 + 3, state_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(state_dim, 1),
        )
        
        # Chemistry-based scoring (parallel path)
        self.chem_scorer = nn.Sequential(
            nn.Linear(mol_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, 1),
        )
    
    def compute_fe_distance_scores(
        self,
        mol_coords: torch.Tensor,
        state_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score atoms by distance to predicted reactive center.
        
        For each state, atoms closer to the reactive center get higher scores.
        Weight by state probabilities.
        """
        B, N, _ = mol_coords.shape
        device = mol_coords.device
        
        active_indices = self.state_bank.get_active_indices()
        n_active = len(active_indices)
        
        if n_active == 0:
            return torch.zeros(B, N, device=device)
        
        # Get reactive centers for active states
        reactive_centers = self.state_bank.reactive_centers[active_indices]  # (n_active, 3)
        reactive_radii = self.state_bank.reactive_radii[active_indices]  # (n_active, 1)
        
        # Compute Fe position + reactive center offset
        target_positions = self.fe_position.unsqueeze(0) + reactive_centers  # (n_active, 3)
        
        # Distance from each atom to each state's target
        # mol_coords: (B, N, 3), target: (n_active, 3)
        distances = torch.cdist(mol_coords, target_positions.unsqueeze(0).expand(B, -1, -1))
        # distances: (B, N, n_active)
        
        # Convert to scores (closer = higher)
        scores_per_state = torch.exp(-distances / reactive_radii.squeeze(-1))  # (B, N, n_active)
        
        # Get active state probs
        active_probs = state_probs[:, self.state_bank.state_active]  # (B, n_active)
        
        # Weight by state probabilities
        weighted_scores = (scores_per_state * active_probs.unsqueeze(1)).sum(dim=-1)  # (B, N)
        
        return weighted_scores
    
    def forward(
        self,
        mol_features: torch.Tensor,
        mol_coords: torch.Tensor,
        pocket_features: torch.Tensor,
        som_mask: torch.Tensor = None,
        valid_mask: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dynamic state discovery.
        """
        B, N, mol_dim = mol_features.shape
        device = mol_features.device
        
        # Encode molecules
        mol_encoded = self.mol_encoder(mol_features)  # (B, N, state_dim)
        mol_global = mol_encoded.mean(dim=1)  # (B, state_dim)
        
        # Get state-weighted pocket
        weighted_pocket, state_probs, active_indices = self.state_bank(
            mol_global, pocket_features
        )
        
        # Project pocket
        pocket_proj = self.pocket_proj(weighted_pocket)  # (B, N_pocket, state_dim)
        
        # Cross-attention
        attended, _ = self.cross_attention(
            mol_encoded,
            pocket_proj,
            pocket_proj,
        )
        
        # Fe distance scores
        fe_scores = self.compute_fe_distance_scores(mol_coords, state_probs)
        
        # Combine for atom scoring
        combined = torch.cat([
            mol_encoded,
            attended,
            mol_coords,
        ], dim=-1)
        
        atom_scores = self.atom_scorer(combined).squeeze(-1)  # (B, N)
        
        # Chemistry scores
        chem_scores = self.chem_scorer(mol_features).squeeze(-1)  # (B, N)
        
        # Final scores
        scores = atom_scores + 0.5 * fe_scores + 0.3 * chem_scores
        
        output = {
            'scores': scores,
            'final_scores': scores,
            'state_probs': state_probs,
            'n_active_states': self.state_bank.num_active_states,
            'active_indices': active_indices,
            'fe_scores': fe_scores,
            'chem_scores': chem_scores,
        }
        
        return output


class DynamicStateLoss(nn.Module):
    """Loss function for dynamic state model."""
    
    def __init__(self, diversity_weight: float = 0.01):
        super().__init__()
        self.diversity_weight = diversity_weight
    
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        som_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss."""
        scores = output['scores']
        B, N = scores.shape
        device = scores.device
        
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=device)
        
        # Ranking loss (cross-entropy softmax)
        ranking_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for b in range(B):
            mask = valid_mask[b]
            labels = som_mask[b]
            s = scores[b]
            
            som_indices = torch.where((labels > 0) & mask)[0]
            if len(som_indices) == 0:
                continue
            
            all_valid_scores = s[mask]
            
            for som_idx in som_indices:
                som_score = s[som_idx]
                log_softmax = som_score - torch.logsumexp(all_valid_scores, dim=0)
                ranking_loss = ranking_loss - log_softmax
                count += 1
        
        if count > 0:
            ranking_loss = ranking_loss / count
        
        total_loss = ranking_loss
        loss_dict['ranking'] = ranking_loss.item()
        loss_dict['total'] = total_loss.item()
        loss_dict['n_active_states'] = output['n_active_states']
        
        return total_loss, loss_dict
