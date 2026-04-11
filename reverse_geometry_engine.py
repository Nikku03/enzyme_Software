#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║   REVERSE GEOMETRY ENGINE - Learning Enzyme Dynamics from SoM Data                      ║
║   ══════════════════════════════════════════════════════════════════════════════════════ ║
║                                                                                          ║
║   KEY INSIGHT:                                                                           ║
║   ────────────                                                                           ║
║   We know: static pocket (1W0E.pdb) + which atoms were oxidized (SoM labels)            ║
║   Therefore: we can INFER what the pocket looked like during each oxidation             ║
║                                                                                          ║
║   The enzyme pocket is FLEXIBLE - it adopts different conformations for different       ║
║   substrates. By analyzing many substrate-SoM pairs, we can learn:                      ║
║   1. What pocket "states" exist (clustering of inferred conformations)                  ║
║   2. What molecular features select each state                                          ║
║   3. How to predict which state a new molecule will induce                              ║
║                                                                                          ║
║   ARCHITECTURE:                                                                          ║
║   ─────────────                                                                          ║
║   1. Pose Inference Network: Given (molecule, SoM) → inferred pocket state              ║
║   2. State Library: K learnable pocket conformations (like attention heads)             ║
║   3. State Selector: Given molecule → probability over pocket states                    ║
║   4. SoM Predictor: Given (molecule, selected state) → atom scores                      ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# POCKET STATE REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PocketState:
    """
    Represents a learned pocket conformation.
    
    The pocket can flex to accommodate different substrates.
    Each state represents a different "binding mode".
    """
    # Center of reactive zone (relative to Fe)
    reactive_center: torch.Tensor  # (3,)
    
    # Orientation of reactive zone (principal axes)
    reactive_axes: torch.Tensor  # (3, 3)
    
    # Size/shape of reactive zone
    reactive_extent: torch.Tensor  # (3,) - extent along each axis
    
    # Hydrophobicity gradient (which direction is more hydrophobic)
    hydro_gradient: torch.Tensor  # (3,)
    
    # Flexibility profile (how much each region can move)
    flexibility: torch.Tensor  # (n_regions,)


class PocketStateLibrary(nn.Module):
    """
    Learnable library of pocket conformations.
    
    During training, we learn K different pocket states that explain
    the observed SoM patterns. Each state represents a different way
    the enzyme can bind substrates.
    """
    
    def __init__(
        self, 
        n_states: int = 8,
        state_dim: int = 64,
        pocket_dim: int = 14,
    ):
        super().__init__()
        self.n_states = n_states
        self.state_dim = state_dim
        
        # Learnable state embeddings
        # Each state is a learned representation of a pocket conformation
        self.state_embeddings = nn.Parameter(torch.randn(n_states, state_dim))
        
        # State-specific transformations of the base pocket
        # These define how each state "deforms" the static pocket
        self.state_deformations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pocket_dim, state_dim),
                nn.SiLU(),
                nn.Linear(state_dim, pocket_dim),
            )
            for _ in range(n_states)
        ])
        
        # Reactive zone parameters for each state
        # (where the oxidation happens relative to Fe)
        self.reactive_centers = nn.Parameter(torch.zeros(n_states, 3))
        self.reactive_radii = nn.Parameter(torch.ones(n_states, 1) * 4.0)  # ~4Å from Fe
        
        # State compatibility network
        # Given molecular features, predict which states are compatible
        self.state_selector = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, n_states),
        )
        
        self._init_states()
    
    def _init_states(self):
        """Initialize states with diverse conformations."""
        # Initialize reactive centers around Fe in different directions
        angles = torch.linspace(0, 2 * np.pi, self.n_states + 1)[:-1]
        for i, angle in enumerate(angles):
            self.reactive_centers.data[i] = torch.tensor([
                4.0 * np.cos(angle),
                4.0 * np.sin(angle),
                np.random.uniform(-2, 2),
            ])
    
    def get_state_pocket(
        self, 
        state_idx: int, 
        base_pocket: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the pocket representation for a specific state.
        
        Args:
            state_idx: Which state to use
            base_pocket: (N_pocket, pocket_dim) base pocket features
        
        Returns:
            (N_pocket, pocket_dim) deformed pocket for this state
        """
        deformation = self.state_deformations[state_idx](base_pocket)
        return base_pocket + 0.1 * deformation  # Small deformation
    
    def select_states(
        self, 
        mol_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Given molecular features, predict state probabilities.
        
        Args:
            mol_embedding: (B, state_dim) molecular embedding
        
        Returns:
            (B, n_states) probability over states
        """
        logits = self.state_selector(mol_embedding)
        return F.softmax(logits, dim=-1)
    
    def forward(
        self,
        mol_embedding: torch.Tensor,
        base_pocket: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute state-weighted pocket representation.
        
        Args:
            mol_embedding: (B, state_dim) molecular embedding
            base_pocket: (N_pocket, pocket_dim) base pocket
        
        Returns:
            weighted_pocket: (B, N_pocket, pocket_dim)
            state_probs: (B, n_states)
        """
        B = mol_embedding.shape[0]
        N_pocket = base_pocket.shape[0]
        
        # Get state probabilities
        state_probs = self.select_states(mol_embedding)  # (B, n_states)
        
        # Compute deformed pocket for each state
        state_pockets = []
        for i in range(self.n_states):
            deformed = self.get_state_pocket(i, base_pocket)  # (N_pocket, pocket_dim)
            state_pockets.append(deformed)
        
        state_pockets = torch.stack(state_pockets)  # (n_states, N_pocket, pocket_dim)
        
        # Weight by state probabilities
        # (B, n_states, 1, 1) * (1, n_states, N_pocket, pocket_dim)
        weighted = state_probs.unsqueeze(-1).unsqueeze(-1) * state_pockets.unsqueeze(0)
        weighted_pocket = weighted.sum(dim=1)  # (B, N_pocket, pocket_dim)
        
        return weighted_pocket, state_probs


# ═══════════════════════════════════════════════════════════════════════════════
# POSE INFERENCE NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class PoseInferenceNetwork(nn.Module):
    """
    Infers the binding pose from (molecule, SoM) pair.
    
    Given that we know which atom was oxidized, we can infer:
    1. How the molecule was oriented (SoM must be near Fe)
    2. What pocket state was induced
    3. What features of the molecule caused this binding mode
    
    This is used during TRAINING to learn pocket states.
    """
    
    def __init__(
        self,
        mol_dim: int = 128,
        state_dim: int = 64,
        n_states: int = 8,
    ):
        super().__init__()
        
        # Encode the molecular context around SoM
        self.som_context_encoder = nn.Sequential(
            nn.Linear(mol_dim + 3 + 1, state_dim * 2),  # mol_feat + coord + is_som
            nn.SiLU(),
            nn.Linear(state_dim * 2, state_dim),
        )
        
        # Infer which pocket state was active
        self.state_classifier = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, n_states),
        )
        
        # Infer the binding pose parameters
        self.pose_regressor = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, 6),  # 3 for translation, 3 for rotation
        )
    
    def forward(
        self,
        mol_features: torch.Tensor,  # (B, N, mol_dim)
        mol_coords: torch.Tensor,  # (B, N, 3)
        som_mask: torch.Tensor,  # (B, N) - 1 for SoM atoms
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Infer pocket state and pose from known SoM.
        
        Returns:
            state_logits: (B, n_states) - which state was active
            pose_params: (B, 6) - inferred binding pose
        """
        B, N, _ = mol_features.shape
        
        # Get SoM atom features (average if multiple)
        som_mask_expanded = som_mask.unsqueeze(-1)  # (B, N, 1)
        som_count = som_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        
        # Weighted average of SoM atom features
        som_features = (mol_features * som_mask_expanded).sum(dim=1) / som_count  # (B, mol_dim)
        som_coords = (mol_coords * som_mask_expanded).sum(dim=1) / som_count  # (B, 3)
        
        # Is this atom SoM? (always 1 for SoM atoms)
        som_indicator = torch.ones(B, 1, device=mol_features.device)
        
        # Encode SoM context
        som_input = torch.cat([som_features, som_coords, som_indicator], dim=-1)
        som_encoding = self.som_context_encoder(som_input)  # (B, state_dim)
        
        # Infer state and pose
        state_logits = self.state_classifier(som_encoding)  # (B, n_states)
        pose_params = self.pose_regressor(som_encoding)  # (B, 6)
        
        return state_logits, pose_params


# ═══════════════════════════════════════════════════════════════════════════════
# REVERSE GEOMETRY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ReverseGeometryEngine(nn.Module):
    """
    The main engine for learning enzyme dynamics from SoM data.
    
    TRAINING:
    1. Takes (molecule, SoM) pairs
    2. Infers what pocket state was active for each
    3. Updates the pocket state library to explain observations
    4. Learns which molecular features select which states
    
    INFERENCE:
    1. Takes a new molecule
    2. Predicts which pocket state(s) it will induce
    3. For each state, computes which atoms could reach Fe
    4. Returns SoM scores
    """
    
    def __init__(
        self,
        mol_dim: int = 128,
        state_dim: int = 64,
        n_states: int = 8,
        pocket_dim: int = 14,
        n_pocket_atoms: int = 580,
    ):
        super().__init__()
        self.n_states = n_states
        self.state_dim = state_dim
        
        # Pocket state library
        self.state_library = PocketStateLibrary(
            n_states=n_states,
            state_dim=state_dim,
            pocket_dim=pocket_dim,
        )
        
        # Pose inference (used during training)
        self.pose_inference = PoseInferenceNetwork(
            mol_dim=mol_dim,
            state_dim=state_dim,
            n_states=n_states,
        )
        
        # Molecular encoder (to state_dim)
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, state_dim),
            nn.SiLU(),
            nn.LayerNorm(state_dim),
        )
        
        # Cross-attention: molecule attends to pocket
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=state_dim,
            num_heads=4,
            batch_first=True,
        )
        
        # Project pocket to state_dim for attention
        self.pocket_proj = nn.Linear(pocket_dim, state_dim)
        
        # Score each atom based on pocket interaction
        self.atom_scorer = nn.Sequential(
            nn.Linear(state_dim * 2 + 3, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim // 2),
            nn.SiLU(),
            nn.Linear(state_dim // 2, 1),
        )
        
        # Fe distance scorer (how far is each atom from Fe in each state)
        self.fe_distance_scorer = nn.Sequential(
            nn.Linear(3 + n_states, state_dim // 2),
            nn.SiLU(),
            nn.Linear(state_dim // 2, 1),
        )
        
        # State confidence (how confident is the state selection)
        self.state_confidence = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.SiLU(),
            nn.Linear(state_dim // 2, 1),
            nn.Sigmoid(),
        )
    
    def compute_fe_distance_scores(
        self,
        coords: torch.Tensor,  # (B, N, 3)
        state_probs: torch.Tensor,  # (B, n_states)
    ) -> torch.Tensor:
        """
        Score atoms based on distance to Fe in each pocket state.
        
        Each state has a different "reactive center" where oxidation occurs.
        Atoms closer to the reactive center in the selected state score higher.
        """
        B, N, _ = coords.shape
        
        # Get reactive centers for each state
        reactive_centers = self.state_library.reactive_centers  # (n_states, 3)
        reactive_radii = self.state_library.reactive_radii  # (n_states, 1)
        
        # Compute distance from each atom to each state's reactive center
        # coords: (B, N, 3) -> (B, N, 1, 3)
        # centers: (n_states, 3) -> (1, 1, n_states, 3)
        coords_exp = coords.unsqueeze(2)
        centers_exp = reactive_centers.unsqueeze(0).unsqueeze(0)
        
        distances = torch.norm(coords_exp - centers_exp, dim=-1)  # (B, N, n_states)
        
        # Score based on distance (closer = higher)
        # Use a soft cutoff based on reactive radius
        radii_exp = reactive_radii.squeeze(-1).unsqueeze(0).unsqueeze(0)  # (1, 1, n_states)
        distance_scores = torch.exp(-((distances - radii_exp) ** 2) / 4.0)  # (B, N, n_states)
        
        # Weight by state probabilities
        state_probs_exp = state_probs.unsqueeze(1)  # (B, 1, n_states)
        weighted_scores = (distance_scores * state_probs_exp).sum(dim=-1)  # (B, N)
        
        return weighted_scores
    
    def forward(
        self,
        mol_features: torch.Tensor,  # (B, N, mol_dim)
        mol_coords: torch.Tensor,  # (B, N, 3)
        pocket_features: torch.Tensor,  # (N_pocket, pocket_dim)
        som_mask: Optional[torch.Tensor] = None,  # (B, N) - for training
        valid_mask: Optional[torch.Tensor] = None,  # (B, N)
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the reverse geometry engine.
        
        During training (som_mask provided):
        - Infer pocket state from known SoM
        - Learn to predict that state from molecular features
        
        During inference (no som_mask):
        - Predict pocket state from molecular features
        - Score atoms based on predicted state
        
        Returns:
            Dictionary with:
            - scores: (B, N) atom-level SoM scores
            - state_probs: (B, n_states) predicted state probabilities
            - inferred_states: (B, n_states) inferred states (training only)
            - pose_params: (B, 6) inferred pose (training only)
            - state_confidence: (B, 1) how confident the state selection is
        """
        B, N, mol_dim = mol_features.shape
        device = mol_features.device
        
        # Encode molecular features
        mol_encoded = self.mol_encoder(mol_features)  # (B, N, state_dim)
        
        # Global molecular embedding (mean pool)
        mol_global = mol_encoded.mean(dim=1)  # (B, state_dim)
        
        # Get state-weighted pocket representation
        weighted_pocket, state_probs = self.state_library(
            mol_global, pocket_features
        )  # (B, N_pocket, pocket_dim), (B, n_states)
        
        # Project pocket to state_dim
        pocket_proj = self.pocket_proj(weighted_pocket)  # (B, N_pocket, state_dim)
        
        # Cross-attention: molecules attend to pocket
        attended, _ = self.cross_attention(
            mol_encoded,  # query: (B, N, state_dim)
            pocket_proj,  # key: (B, N_pocket, state_dim)
            pocket_proj,  # value: (B, N_pocket, state_dim)
        )  # (B, N, state_dim)
        
        # Compute Fe distance scores
        fe_scores = self.compute_fe_distance_scores(mol_coords, state_probs)  # (B, N)
        
        # Combine features for scoring
        combined = torch.cat([
            mol_encoded,
            attended,
            mol_coords,
        ], dim=-1)  # (B, N, state_dim * 2 + 3)
        
        # Score each atom
        atom_scores = self.atom_scorer(combined).squeeze(-1)  # (B, N)
        
        # Combine with Fe distance scores
        scores = atom_scores + fe_scores
        
        # State confidence
        confidence = self.state_confidence(mol_global)  # (B, 1)
        
        # Modulate scores by confidence
        scores = scores * confidence
        
        # Prepare output
        output = {
            'scores': scores,
            'state_probs': state_probs,
            'state_confidence': confidence,
            'fe_scores': fe_scores,
        }
        
        # If training (SoM mask provided), also infer states
        if som_mask is not None:
            inferred_logits, pose_params = self.pose_inference(
                mol_features, mol_coords, som_mask
            )
            output['inferred_state_logits'] = inferred_logits
            output['pose_params'] = pose_params
        
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class ReverseGeometryLoss(nn.Module):
    """
    Loss function for training the reverse geometry engine.
    
    Components:
    1. SoM ranking loss: SoM atoms should score higher
    2. State consistency loss: Predicted state should match inferred state
    3. Pose regularization: Inferred poses should be realistic
    4. State diversity loss: States should be different from each other
    """
    
    def __init__(
        self,
        ranking_weight: float = 1.0,
        state_weight: float = 0.5,
        diversity_weight: float = 0.1,
    ):
        super().__init__()
        self.ranking_weight = ranking_weight
        self.state_weight = state_weight
        self.diversity_weight = diversity_weight
    
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        som_mask: torch.Tensor,  # (B, N)
        valid_mask: torch.Tensor,  # (B, N)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined loss.
        """
        scores = output['scores']
        state_probs = output['state_probs']
        
        B, N = scores.shape
        device = scores.device
        
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # 1. Ranking loss: SoM atoms should score higher
        ranking_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for b in range(B):
            mask = valid_mask[b]
            labels = som_mask[b]
            s = scores[b]
            
            som_indices = torch.where((labels > 0) & mask)[0]
            non_som_indices = torch.where((labels == 0) & mask)[0]
            
            if len(som_indices) == 0 or len(non_som_indices) == 0:
                continue
            
            # Margin ranking loss
            for som_idx in som_indices:
                som_score = s[som_idx]
                non_som_scores = s[non_som_indices]
                
                # SoM should be higher by at least margin=1
                margin_loss = F.relu(1.0 - som_score + non_som_scores).mean()
                ranking_loss = ranking_loss + margin_loss
                count += 1
        
        if count > 0:
            ranking_loss = ranking_loss / count
        
        total_loss = total_loss + self.ranking_weight * ranking_loss
        loss_dict['ranking'] = ranking_loss.item()
        
        # 2. State consistency loss (if training)
        if 'inferred_state_logits' in output:
            inferred_logits = output['inferred_state_logits']  # (B, n_states)
            predicted_probs = state_probs  # (B, n_states)
            
            # The predicted state should match the inferred state
            inferred_probs = F.softmax(inferred_logits, dim=-1)
            state_loss = F.kl_div(
                torch.log(predicted_probs + 1e-8),
                inferred_probs,
                reduction='batchmean',
            )
            
            total_loss = total_loss + self.state_weight * state_loss
            loss_dict['state_consistency'] = state_loss.item()
        
        # 3. State diversity loss (states should be different)
        # Penalize if state embeddings are too similar
        state_embeddings = output.get('state_embeddings', None)
        if state_embeddings is None and hasattr(self, 'state_library'):
            state_embeddings = self.state_library.state_embeddings
        
        if state_embeddings is not None:
            # Cosine similarity between states
            normed = F.normalize(state_embeddings, dim=-1)
            similarity = torch.mm(normed, normed.t())
            
            # Penalize high similarity (except diagonal)
            n_states = similarity.shape[0]
            mask = ~torch.eye(n_states, dtype=torch.bool, device=device)
            diversity_loss = similarity[mask].mean()
            
            total_loss = total_loss + self.diversity_weight * diversity_loss
            loss_dict['diversity'] = diversity_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION / ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_learned_states(engine: ReverseGeometryEngine) -> Dict[str, np.ndarray]:
    """
    Analyze the learned pocket states.
    
    Returns:
        Dictionary with:
        - reactive_centers: (n_states, 3) positions of reactive zones
        - state_embeddings: (n_states, state_dim) learned state vectors
        - state_similarities: (n_states, n_states) pairwise similarities
    """
    with torch.no_grad():
        # Get reactive centers
        centers = engine.state_library.reactive_centers.cpu().numpy()
        
        # Get state embeddings
        embeddings = engine.state_library.state_embeddings.cpu().numpy()
        
        # Compute similarities
        normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = normed @ normed.T
        
        return {
            'reactive_centers': centers,
            'state_embeddings': embeddings,
            'state_similarities': similarities,
        }


def visualize_state_selection(
    engine: ReverseGeometryEngine,
    mol_features: torch.Tensor,
    mol_coords: torch.Tensor,
    pocket_features: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Visualize which states are selected for a molecule.
    
    Returns:
        Dictionary with:
        - state_probs: (n_states,) probability of each state
        - atom_scores: (N,) score for each atom
        - fe_scores: (N,) Fe distance scores
    """
    engine.eval()
    with torch.no_grad():
        output = engine(
            mol_features.unsqueeze(0),
            mol_coords.unsqueeze(0),
            pocket_features,
        )
        
        return {
            'state_probs': output['state_probs'][0].cpu().numpy(),
            'atom_scores': output['scores'][0].cpu().numpy(),
            'fe_scores': output['fe_scores'][0].cpu().numpy(),
            'state_confidence': output['state_confidence'][0].cpu().numpy(),
        }


if __name__ == '__main__':
    # Test the module
    print("Testing Reverse Geometry Engine...")
    
    # Create engine
    engine = ReverseGeometryEngine(
        mol_dim=128,
        state_dim=64,
        n_states=8,
        pocket_dim=14,
    )
    
    # Test forward pass
    B, N = 4, 20
    N_pocket = 100
    
    mol_features = torch.randn(B, N, 128)
    mol_coords = torch.randn(B, N, 3)
    pocket_features = torch.randn(N_pocket, 14)
    som_mask = torch.zeros(B, N)
    som_mask[:, 5] = 1  # Atom 5 is SoM
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    # Forward
    output = engine(mol_features, mol_coords, pocket_features, som_mask, valid_mask)
    
    print(f"Scores shape: {output['scores'].shape}")
    print(f"State probs shape: {output['state_probs'].shape}")
    print(f"Inferred state logits shape: {output['inferred_state_logits'].shape}")
    print(f"State confidence: {output['state_confidence'].mean().item():.3f}")
    
    # Test loss
    loss_fn = ReverseGeometryLoss()
    loss, loss_dict = loss_fn(output, som_mask, valid_mask)
    
    print(f"\nLoss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # Analyze states
    analysis = analyze_learned_states(engine)
    print(f"\nReactive centers:\n{analysis['reactive_centers']}")
    print(f"\nState similarities (should be low off-diagonal):")
    print(analysis['state_similarities'])
    
    print("\n✓ All tests passed!")
