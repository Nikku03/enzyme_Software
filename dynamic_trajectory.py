"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     DYNAMIC TRAJECTORY MODEL FOR SoM PREDICTION                              ║
║                                                                              ║
║     Key insight: CYP450 catalysis is a DYNAMIC PROCESS, not a static state  ║
║                                                                              ║
║     The model simulates:                                                     ║
║     1. Substrate entry and initial binding                                   ║
║     2. Conformational sampling (multiple poses)                              ║
║     3. Approach to reactive center (trajectory)                              ║
║     4. Compound I formation (reactive state)                                 ║
║     5. HAT/rebound (final SoM selection)                                     ║
║                                                                              ║
║     Architecture:                                                            ║
║     - Neural ODE for continuous-time dynamics                                ║
║     - Attention over enzyme conformational states                            ║
║     - Multi-pose prediction with pose weighting                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# CONFORMATIONAL ENSEMBLE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ConformationalEnsemble(nn.Module):
    """
    Generate multiple conformational states of the substrate.
    
    Instead of predicting SoM from ONE pose, we sample multiple poses
    and predict from the ENSEMBLE.
    
    This captures:
    - Substrate flexibility (rotatable bonds)
    - Multiple binding modes
    - Time-averaged positions
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        n_conformers: int = 8,
        noise_scale: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_conformers = n_conformers
        self.noise_scale = noise_scale
        
        # Learnable conformational perturbations
        self.conformer_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * n_conformers),
        )
        
        # Conformer weighting (Boltzmann-like)
        self.conformer_energy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, atom_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate conformational ensemble.
        
        Args:
            atom_features: [B, N, hidden_dim]
            
        Returns:
            conformers: [B, n_conf, N, hidden_dim] - features for each conformer
            weights: [B, n_conf] - Boltzmann weights for each conformer
        """
        B, N, D = atom_features.shape
        
        # Global molecule representation
        mol_global = atom_features.mean(dim=1)  # [B, D]
        
        # Generate conformer perturbations
        perturbations = self.conformer_generator(mol_global)  # [B, D * n_conf]
        perturbations = perturbations.view(B, self.n_conformers, D)  # [B, n_conf, D]
        
        # Add stochastic noise (conformational sampling)
        if self.training:
            noise = torch.randn_like(perturbations) * self.noise_scale
            perturbations = perturbations + noise
        
        # Apply perturbations to each atom
        # [B, N, D] + [B, n_conf, 1, D] -> [B, n_conf, N, D]
        conformers = atom_features.unsqueeze(1) + perturbations.unsqueeze(2) * 0.1
        
        # Compute energies (lower = more stable = higher weight)
        # [B, n_conf, D]
        conformer_global = conformers.mean(dim=2)
        energies = self.conformer_energy(conformer_global).squeeze(-1)  # [B, n_conf]
        
        # Boltzmann weights (softmax with temperature)
        temperature = 1.0
        weights = F.softmax(-energies / temperature, dim=-1)  # [B, n_conf]
        
        return conformers, weights


# ═══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY DYNAMICS (Neural ODE-inspired)
# ═══════════════════════════════════════════════════════════════════════════════

class TrajectoryDynamics(nn.Module):
    """
    Model the TRAJECTORY of substrate atoms approaching the reactive center.
    
    Key insight: The SoM is not just the atom closest to Fe at t=0,
    but the atom that ENDS UP closest to Fe at t=reaction.
    
    We use a simplified Neural ODE approach:
    - State: atom positions relative to Fe
    - Dynamics: learned drift towards/away from reactive center
    - Integrate over "time" steps
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        n_steps: int = 10,
        dt: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.dt = dt
        
        # Dynamics network: predicts velocity field
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for position relative to Fe
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 + hidden_dim),  # position delta + feature delta
        )
        
        # Enzyme influence (how enzyme state affects dynamics)
        self.enzyme_influence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Fe attraction field (learned)
        self.fe_attraction = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))
        
    def forward(
        self,
        atom_features: torch.Tensor,
        atom_coords: torch.Tensor,
        fe_position: torch.Tensor,
        enzyme_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate trajectory dynamics.
        
        Args:
            atom_features: [B, N, hidden_dim]
            atom_coords: [B, N, 3]
            fe_position: [B, 3] or [3]
            enzyme_state: [B, hidden_dim] optional enzyme representation
            
        Returns:
            final_features: [B, N, hidden_dim] - features after dynamics
            final_coords: [B, N, 3] - positions after dynamics
            trajectory_scores: [B, N] - how much time each atom spent near Fe
        """
        B, N, D = atom_features.shape
        
        # Expand Fe position
        if fe_position.dim() == 1:
            fe_position = fe_position.unsqueeze(0).expand(B, -1)  # [B, 3]
        
        # Initialize state
        h = atom_features.clone()
        pos = atom_coords.clone()
        
        # Track time spent near Fe (trajectory score)
        proximity_integral = torch.zeros(B, N, device=atom_features.device)
        
        # Apply enzyme influence
        if enzyme_state is not None:
            enzyme_effect = self.enzyme_influence(enzyme_state)  # [B, D]
            h = h + enzyme_effect.unsqueeze(1) * 0.1
        
        # Euler integration
        for step in range(self.n_steps):
            # Relative position to Fe
            rel_pos = pos - fe_position.unsqueeze(1)  # [B, N, 3]
            dist_to_fe = rel_pos.norm(dim=-1, keepdim=True).clamp(min=0.1)  # [B, N, 1]
            
            # Combine features with position
            combined = torch.cat([h, rel_pos], dim=-1)  # [B, N, D+3]
            
            # Compute dynamics (velocity field)
            delta = self.dynamics(combined)  # [B, N, D+3]
            d_pos = delta[:, :, :3]  # Position change
            d_h = delta[:, :, 3:]    # Feature change
            
            # Add Fe attraction (atoms are drawn towards Fe)
            attraction = -rel_pos / dist_to_fe.pow(2) * self.fe_attraction
            d_pos = d_pos + attraction * 0.1
            
            # Update state
            pos = pos + d_pos * self.dt
            h = h + d_h * self.dt
            
            # Accumulate proximity score (Gaussian proximity)
            new_dist = (pos - fe_position.unsqueeze(1)).norm(dim=-1)
            proximity = torch.exp(-new_dist.pow(2) / (2 * 3.0**2))  # σ = 3 Å
            proximity_integral = proximity_integral + proximity * self.dt
        
        return h, pos, proximity_integral


# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME STATE EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

class EnzymeStateEvolution(nn.Module):
    """
    Model the enzyme conformational changes during catalysis.
    
    CYP3A4 has multiple conformational states:
    - Open (substrate entry)
    - Closed (reactive complex)
    - Intermediate states
    
    The enzyme state affects which atoms can access the reactive center.
    """
    
    def __init__(
        self,
        pocket_dim: int = 14,
        hidden_dim: int = 64,
        n_enzyme_states: int = 4,
    ):
        super().__init__()
        self.n_enzyme_states = n_enzyme_states
        
        # Encode pocket features
        self.pocket_encoder = nn.Sequential(
            nn.Linear(pocket_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Enzyme conformational states (learned)
        self.enzyme_states = nn.Parameter(torch.randn(n_enzyme_states, hidden_dim) * 0.1)
        
        # State transition matrix (which state transitions to which)
        self.transition_logits = nn.Parameter(torch.zeros(n_enzyme_states, n_enzyme_states))
        
        # Substrate-dependent state selection
        self.state_selector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_enzyme_states),
        )
        
        # Active site "gating" per state (which regions are accessible)
        self.accessibility_gate = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        pocket_features: torch.Tensor,
        substrate_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute enzyme state distribution and accessibility.
        
        Args:
            pocket_features: [n_pocket, pocket_dim]
            substrate_features: [B, N, hidden_dim]
            
        Returns:
            enzyme_state: [B, hidden_dim] - weighted enzyme state
            state_probs: [B, n_states] - probability of each state
            accessibility: [B, N] - how accessible each atom is
        """
        B, N, D = substrate_features.shape
        
        # Encode pocket
        pocket_encoded = self.pocket_encoder(pocket_features)  # [n_pocket, D]
        pocket_global = pocket_encoded.mean(dim=0)  # [D]
        
        # Substrate global
        substrate_global = substrate_features.mean(dim=1)  # [B, D]
        
        # Select enzyme state based on substrate
        combined = torch.cat([
            substrate_global,
            pocket_global.unsqueeze(0).expand(B, -1)
        ], dim=-1)  # [B, 2D]
        
        state_logits = self.state_selector(combined)  # [B, n_states]
        state_probs = F.softmax(state_logits, dim=-1)
        
        # Weighted enzyme state
        enzyme_state = torch.einsum('bs,sd->bd', state_probs, self.enzyme_states)  # [B, D]
        
        # Compute accessibility for each atom
        # (how much can this atom approach Fe given enzyme state)
        atom_enzyme = torch.cat([
            substrate_features,
            enzyme_state.unsqueeze(1).expand(-1, N, -1)
        ], dim=-1)  # [B, N, 2D]
        
        accessibility = self.accessibility_gate(atom_enzyme).squeeze(-1)  # [B, N]
        
        return enzyme_state, state_probs, accessibility


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-POSE SoM PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MultiPoseSoMPredictor(nn.Module):
    """
    Predict SoM distribution by considering MULTIPLE POSES.
    
    Instead of picking ONE best atom, we predict a DISTRIBUTION
    that matches the distribution of metabolites observed experimentally.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        n_poses: int = 5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_poses = n_poses
        
        # Pose generator (different ways substrate can bind)
        self.pose_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * n_poses),
        )
        
        # Pose-specific scorer
        self.pose_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Pose probability (how likely is each pose)
        self.pose_prob = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
    def forward(
        self,
        atom_features: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict SoM distribution from multiple poses.
        
        Args:
            atom_features: [B, N, hidden_dim]
            valid_mask: [B, N]
            
        Returns:
            som_distribution: [B, N] - probability distribution over atoms
            pose_scores: [B, n_poses, N] - scores per pose
            pose_weights: [B, n_poses] - weight of each pose
        """
        B, N, D = atom_features.shape
        
        # Generate pose-specific features
        mol_global = atom_features.mean(dim=1)  # [B, D]
        pose_deltas = self.pose_generator(mol_global)  # [B, D * n_poses]
        pose_deltas = pose_deltas.view(B, self.n_poses, D)  # [B, n_poses, D]
        
        # Apply pose-specific transformations
        # [B, N, D] + [B, n_poses, 1, D] -> [B, n_poses, N, D]
        pose_features = atom_features.unsqueeze(1) + pose_deltas.unsqueeze(2) * 0.2
        
        # Score each atom in each pose
        pose_scores = self.pose_scorer(pose_features).squeeze(-1)  # [B, n_poses, N]
        
        # Mask invalid atoms
        pose_scores = pose_scores.masked_fill(~valid_mask.unsqueeze(1), -1e9)
        
        # Compute pose probabilities
        pose_global = pose_features.mean(dim=2)  # [B, n_poses, D]
        pose_logits = self.pose_prob(pose_global).squeeze(-1)  # [B, n_poses]
        pose_weights = F.softmax(pose_logits, dim=-1)
        
        # Weighted combination of pose scores
        # [B, n_poses, N] weighted by [B, n_poses, 1]
        weighted_scores = pose_scores * pose_weights.unsqueeze(-1)
        combined_scores = weighted_scores.sum(dim=1)  # [B, N]
        
        # Final SoM distribution
        som_distribution = F.softmax(combined_scores, dim=-1)
        
        return som_distribution, pose_scores, pose_weights


# ═══════════════════════════════════════════════════════════════════════════════
# FULL DYNAMIC TRAJECTORY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicTrajectoryModel(nn.Module):
    """
    Complete model combining all dynamic components.
    
    Pipeline:
    1. Encode molecule features
    2. Generate conformational ensemble
    3. For each conformer:
       a. Compute enzyme state
       b. Simulate trajectory dynamics
       c. Score atoms based on final positions
    4. Aggregate across conformers (Boltzmann weighting)
    5. Predict SoM distribution
    """
    
    def __init__(
        self,
        mol_dim: int = 128,
        hidden_dim: int = 64,
        pocket_dim: int = 14,
        n_conformers: int = 4,
        n_trajectory_steps: int = 8,
        n_poses: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Molecule encoder
        self.mol_encoder = nn.Sequential(
            nn.Linear(mol_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Dynamic components
        self.conformational_ensemble = ConformationalEnsemble(
            hidden_dim, n_conformers
        )
        self.trajectory_dynamics = TrajectoryDynamics(
            hidden_dim, n_trajectory_steps
        )
        self.enzyme_evolution = EnzymeStateEvolution(
            pocket_dim, hidden_dim
        )
        self.multi_pose_predictor = MultiPoseSoMPredictor(
            hidden_dim, n_poses
        )
        
        # Final aggregation
        self.final_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 1 + 1, hidden_dim),  # features + trajectory + accessibility
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Fe position (learnable or from structure)
        self.register_buffer('fe_position', torch.tensor([54.95, 77.69, 10.64]))
        
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        pocket_features: torch.Tensor,
        som_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass with dynamics."""
        B, N, _ = features.shape
        
        if valid_mask is None:
            valid_mask = torch.ones(B, N, dtype=torch.bool, device=features.device)
        
        # 1. Encode molecules
        atom_encoded = self.mol_encoder(features)  # [B, N, D]
        
        # 2. Generate conformational ensemble
        conformers, conf_weights = self.conformational_ensemble(atom_encoded)
        # conformers: [B, n_conf, N, D], conf_weights: [B, n_conf]
        
        n_conf = conformers.shape[1]
        
        # 3. Process each conformer
        all_scores = []
        all_trajectory_scores = []
        
        for c in range(n_conf):
            conf_features = conformers[:, c]  # [B, N, D]
            
            # 3a. Enzyme state evolution
            enzyme_state, state_probs, accessibility = self.enzyme_evolution(
                pocket_features, conf_features
            )
            
            # 3b. Trajectory dynamics
            final_features, final_coords, trajectory_scores = self.trajectory_dynamics(
                conf_features, coords, self.fe_position, enzyme_state
            )
            
            # 3c. Score atoms
            combined = torch.cat([
                final_features,
                trajectory_scores.unsqueeze(-1),
                accessibility.unsqueeze(-1),
            ], dim=-1)  # [B, N, D+2]
            
            scores = self.final_scorer(combined).squeeze(-1)  # [B, N]
            scores = scores.masked_fill(~valid_mask, -1e9)
            
            all_scores.append(scores)
            all_trajectory_scores.append(trajectory_scores)
        
        # 4. Aggregate across conformers (Boltzmann weighting)
        all_scores = torch.stack(all_scores, dim=1)  # [B, n_conf, N]
        weighted_scores = all_scores * conf_weights.unsqueeze(-1)
        final_scores = weighted_scores.sum(dim=1)  # [B, N]
        
        # 5. Multi-pose prediction
        som_dist, pose_scores, pose_weights = self.multi_pose_predictor(
            atom_encoded, valid_mask
        )
        
        # Combine trajectory-based and pose-based predictions
        combined_scores = 0.5 * final_scores + 0.5 * (som_dist * 10)  # Scale som_dist
        
        return {
            'final_scores': combined_scores,
            'trajectory_scores': torch.stack(all_trajectory_scores, dim=1).mean(dim=1),
            'conformer_weights': conf_weights,
            'pose_weights': pose_weights,
            'som_distribution': som_dist,
            'physics_scores': final_scores,  # For compatibility
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicLoss(nn.Module):
    """
    Loss function for dynamic model.
    
    Key insight: For molecules with MULTIPLE SoMs, we should match
    the DISTRIBUTION, not just get one right.
    """
    
    def __init__(self, distribution_weight: float = 0.3):
        super().__init__()
        self.distribution_weight = distribution_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        som_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        final_scores = outputs['final_scores']
        som_dist = outputs['som_distribution']
        
        # Normalize SoM mask to distribution
        som_sum = som_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        som_normalized = som_mask / som_sum
        
        # Main ranking loss
        log_probs = F.log_softmax(final_scores, dim=-1).clamp(min=-100)
        main_loss = -(som_normalized * log_probs).sum(dim=-1).mean()
        
        # Distribution matching loss (KL divergence)
        dist_loss = F.kl_div(
            som_dist.log().clamp(min=-100),
            som_normalized,
            reduction='batchmean',
        ).clamp(max=10.0)
        
        total_loss = main_loss + self.distribution_weight * dist_loss
        
        metrics = {
            'main_loss': main_loss.item(),
            'dist_loss': dist_loss.item(),
        }
        
        return total_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Dynamic Trajectory Model...")
    
    model = DynamicTrajectoryModel()
    loss_fn = DynamicLoss()
    
    # Test input
    B, N = 4, 50
    features = torch.randn(B, N, 128)
    coords = torch.randn(B, N, 3) * 10
    pocket = torch.randn(100, 14)
    som_mask = torch.zeros(B, N)
    for b in range(B):
        som_mask[b, b * 5] = 1.0
        som_mask[b, b * 5 + 1] = 0.5  # Multi-SoM
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    # Forward pass
    outputs = model(features, coords, pocket, som_mask, valid_mask)
    
    print(f"Final scores shape: {outputs['final_scores'].shape}")
    print(f"Trajectory scores shape: {outputs['trajectory_scores'].shape}")
    print(f"Conformer weights: {outputs['conformer_weights'][0]}")
    print(f"Pose weights: {outputs['pose_weights'][0]}")
    print(f"SoM distribution sum: {outputs['som_distribution'].sum(dim=-1)}")
    
    # Loss
    loss, metrics = loss_fn(outputs, som_mask, valid_mask)
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Backward
    loss.backward()
    print("\n✓ Dynamic Trajectory Model test passed!")
