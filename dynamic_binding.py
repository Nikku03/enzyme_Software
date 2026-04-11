"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     DYNAMIC BINDING POSE MODEL FOR CYP3A4 SoM PREDICTION                     ║
║                                                                              ║
║     Key insight: Metabolism is NOT determined by a single static pose.       ║
║     Instead, the substrate samples MULTIPLE orientations within the          ║
║     active site. The observed SoMs reflect:                                  ║
║                                                                              ║
║     P(SoM at atom i) = Σ_pose P(pose) × P(oxidation at i | pose)             ║
║                                                                              ║
║     This module models:                                                      ║
║     1. Multiple binding poses (via conformer ensemble)                       ║
║     2. Pose probabilities (via interaction scoring)                          ║
║     3. Per-pose reactivity (distance + intrinsic reactivity)                 ║
║     4. Aggregation to final SoM probability                                  ║
║                                                                              ║
║     Also models ENZYME DYNAMICS:                                             ║
║     - Active site flexibility (F-G loop movement)                            ║
║     - Induced fit upon substrate binding                                     ║
║     - Breathing motions during catalysis                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ═══════════════════════════════════════════════════════════════════════════════
# BINDING POSE REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BindingPose:
    """Represents one possible binding orientation."""
    coords: np.ndarray  # [n_atoms, 3] - atom coordinates in this pose
    fe_distances: np.ndarray  # [n_atoms] - distance to Fe for each atom
    interaction_score: float  # How favorable is this pose?
    accessible_atoms: List[int]  # Which atoms can reach Fe in this pose
    pose_probability: float = 0.0  # P(this pose) - set by ensemble


class PoseEnsemble:
    """
    Generates and manages ensemble of binding poses.
    
    Models the dynamic sampling of substrate orientations in the active site.
    """
    
    # CYP3A4 active site reference points (from 1W0E.pdb)
    FE_POSITION = np.array([54.946, 77.694, 10.644])
    
    # Key residues for interaction scoring
    HYDROPHOBIC_CENTERS = [
        np.array([51.2, 79.1, 8.3]),   # Phe304 approximate center
        np.array([56.8, 74.2, 12.1]),  # Phe213
        np.array([48.5, 76.3, 14.2]),  # Phe241
        np.array([52.1, 82.4, 6.7]),   # Ile369
    ]
    
    HBOND_ACCEPTORS = [
        np.array([49.3, 80.2, 11.5]),  # Ser119 OH
        np.array([57.2, 79.8, 7.9]),   # Thr309 OH
    ]
    
    # Reactive distance range (Fe=O to H for abstraction)
    REACTIVE_MIN = 2.5  # Å
    REACTIVE_MAX = 5.0  # Å
    
    def __init__(
        self,
        n_poses: int = 20,
        temperature: float = 1.0,
    ):
        self.n_poses = n_poses
        self.temperature = temperature
    
    def generate_poses(self, mol) -> List[BindingPose]:
        """
        Generate ensemble of binding poses for a molecule.
        
        Uses conformer generation + random orientations to sample
        the space of possible binding modes.
        """
        if not HAS_RDKIT:
            return []
        
        mol = Chem.AddHs(mol)
        n_atoms = mol.GetNumAtoms()
        
        # Generate conformers
        try:
            n_confs = min(self.n_poses * 2, 100)
            AllChem.EmbedMultipleConfs(
                mol, numConfs=n_confs, randomSeed=42,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                numThreads=0,
            )
            AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=100)
        except:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            except:
                return []
        
        if mol.GetNumConformers() == 0:
            return []
        
        poses = []
        
        for conf_id in range(mol.GetNumConformers()):
            conf = mol.GetConformer(conf_id)
            base_coords = np.array([
                [conf.GetAtomPosition(i).x,
                 conf.GetAtomPosition(i).y,
                 conf.GetAtomPosition(i).z]
                for i in range(n_atoms)
            ])
            
            # Generate multiple orientations of this conformer
            for _ in range(max(1, self.n_poses // mol.GetNumConformers())):
                # Random rotation
                rotation = self._random_rotation()
                
                # Center on molecule centroid
                centroid = base_coords.mean(axis=0)
                centered = base_coords - centroid
                rotated = centered @ rotation.T
                
                # Position near Fe with some randomness
                # The substrate "finds" its way to the active site
                offset = np.random.randn(3) * 2.0  # 2 Å random displacement
                final_coords = rotated + self.FE_POSITION + offset
                
                # Compute pose properties
                fe_distances = np.linalg.norm(
                    final_coords - self.FE_POSITION, axis=1
                )
                
                # Which atoms are within reactive distance?
                accessible = np.where(
                    (fe_distances >= self.REACTIVE_MIN) &
                    (fe_distances <= self.REACTIVE_MAX)
                )[0].tolist()
                
                # Score this pose
                interaction_score = self._score_interactions(
                    final_coords, mol
                )
                
                poses.append(BindingPose(
                    coords=final_coords,
                    fe_distances=fe_distances,
                    interaction_score=interaction_score,
                    accessible_atoms=accessible,
                ))
                
                if len(poses) >= self.n_poses:
                    break
            
            if len(poses) >= self.n_poses:
                break
        
        # Compute pose probabilities (Boltzmann-weighted)
        self._compute_pose_probabilities(poses)
        
        return poses
    
    def _random_rotation(self) -> np.ndarray:
        """Generate random 3D rotation matrix."""
        # Uniform random rotation using quaternion method
        u1, u2, u3 = np.random.random(3)
        
        q = np.array([
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ])
        
        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
            [2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[0]*q[1])],
            [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)],
        ])
        
        return R
    
    def _score_interactions(self, coords: np.ndarray, mol) -> float:
        """
        Score pose based on favorable interactions with active site.
        
        Higher score = more stable pose.
        """
        score = 0.0
        
        n_atoms = len(coords)
        
        # 1. Hydrophobic interactions
        for center in self.HYDROPHOBIC_CENTERS:
            for i in range(n_atoms):
                atom = mol.GetAtomWithIdx(i)
                # Check if atom is hydrophobic (C not bonded to heteroatoms)
                if atom.GetSymbol() == 'C' and atom.GetAtomicNum() == 6:
                    is_hydrophobic = all(
                        n.GetSymbol() in ['C', 'H']
                        for n in atom.GetNeighbors()
                    )
                    if is_hydrophobic:
                        dist = np.linalg.norm(coords[i] - center)
                        if dist < 5.0:  # Within interaction range
                            score += np.exp(-dist / 2.0)
        
        # 2. H-bonding (simplified)
        for acceptor_pos in self.HBOND_ACCEPTORS:
            for i in range(n_atoms):
                atom = mol.GetAtomWithIdx(i)
                # Donor atoms: N-H, O-H
                if atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0:
                    dist = np.linalg.norm(coords[i] - acceptor_pos)
                    if dist < 3.5:  # H-bond distance
                        score += 2.0 * np.exp(-dist / 1.5)
        
        # 3. Penalty for steric clashes with Fe-heme
        min_fe_dist = np.min(np.linalg.norm(coords - self.FE_POSITION, axis=1))
        if min_fe_dist < 2.0:  # Too close to Fe
            score -= 5.0 * (2.0 - min_fe_dist)
        
        # 4. Bonus for aromatic stacking (π-π interactions)
        # Simplified: check if aromatic atoms are near Phe304
        phe304_center = self.HYDROPHOBIC_CENTERS[0]
        for i in range(n_atoms):
            atom = mol.GetAtomWithIdx(i)
            if atom.GetIsAromatic():
                dist = np.linalg.norm(coords[i] - phe304_center)
                if 3.5 < dist < 5.0:  # Optimal stacking distance
                    score += 1.5
        
        return score
    
    def _compute_pose_probabilities(self, poses: List[BindingPose]):
        """Assign Boltzmann-weighted probabilities to poses."""
        if not poses:
            return
        
        scores = np.array([p.interaction_score for p in poses])
        
        # Boltzmann weights
        exp_scores = np.exp((scores - scores.max()) / self.temperature)
        probabilities = exp_scores / exp_scores.sum()
        
        for pose, prob in zip(poses, probabilities):
            pose.pose_probability = prob


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC REACTIVITY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicReactivityModel(nn.Module):
    """
    Neural network that models SoM prediction accounting for dynamics.
    
    Architecture:
    1. Per-atom features (chemical properties)
    2. Per-pose accessibility features (from pose ensemble)
    3. Intrinsic reactivity head
    4. Pose-weighted aggregation
    """
    
    def __init__(
        self,
        atom_dim: int = 128,
        hidden_dim: int = 64,
        n_poses: int = 10,
    ):
        super().__init__()
        
        self.n_poses = n_poses
        self.hidden_dim = hidden_dim
        
        # Atom encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Pose-specific features encoder - larger for better discrimination
        # Input: [fe_distance, is_accessible, pose_probability]
        self.pose_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        
        # Combined reactivity predictor
        self.reactivity_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Intrinsic reactivity (pose-independent)
        self.intrinsic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Pose weighting network - ONLY depends on pose features
        # This allows poses to differentiate based on their properties
        self.pose_weight_net = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
    
    def forward(
        self,
        atom_features: torch.Tensor,      # [B, N, atom_dim]
        pose_features: torch.Tensor,       # [B, N, n_poses, 3]
        valid_mask: torch.Tensor,          # [B, N]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-pose aggregation.
        
        For each atom, we compute:
        - Intrinsic reactivity (chemical properties only)
        - Per-pose reactivity (depends on pose)
        - Weighted sum over poses
        
        Final score = intrinsic + Σ_pose (pose_prob × pose_reactivity)
        """
        B, N, _ = atom_features.shape
        n_poses = pose_features.shape[2]
        
        # Encode atom features
        atom_encoded = self.atom_encoder(atom_features)  # [B, N, hidden]
        
        # Intrinsic reactivity (pose-independent)
        intrinsic = self.intrinsic_head(atom_encoded).squeeze(-1)  # [B, N]
        
        # Per-pose reactivity
        pose_reactivities = []
        pose_weights = []
        
        for p in range(n_poses):
            pose_feat = pose_features[:, :, p, :]  # [B, N, 3]
            pose_encoded = self.pose_encoder(pose_feat)  # [B, N, hidden//2]
            
            # Combined features for this pose
            combined = torch.cat([atom_encoded, pose_encoded], dim=-1)
            
            # Reactivity in this pose
            reactivity = self.reactivity_head(combined).squeeze(-1)  # [B, N]
            pose_reactivities.append(reactivity)
            
            # Pose weight - ONLY depends on pose features, not atom features
            # This allows poses to differentiate based on their properties
            weight = self.pose_weight_net(pose_encoded).squeeze(-1)  # [B, N]
            pose_weights.append(weight)
        
        # Stack and normalize weights
        pose_reactivities = torch.stack(pose_reactivities, dim=-1)  # [B, N, n_poses]
        pose_weights = torch.stack(pose_weights, dim=-1)  # [B, N, n_poses]
        
        # Normalize weights (softmax over poses)
        pose_weights = F.softmax(pose_weights, dim=-1)
        
        # Weighted sum of pose reactivities
        dynamic_reactivity = (pose_reactivities * pose_weights).sum(dim=-1)  # [B, N]
        
        # Final score: intrinsic + dynamic
        final_scores = intrinsic + dynamic_reactivity
        
        # Mask invalid atoms
        final_scores = final_scores.masked_fill(~valid_mask, -1e4)
        
        return {
            'final_scores': final_scores,
            'intrinsic_scores': intrinsic,
            'dynamic_scores': dynamic_reactivity,
            'pose_weights': pose_weights,
            'pose_reactivities': pose_reactivities,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME DYNAMICS MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class EnzymeDynamicsModel(nn.Module):
    """
    Models enzyme conformational changes during catalysis.
    
    CYP3A4 is known for:
    1. F-G loop flexibility (opens/closes access channel)
    2. Induced fit upon substrate binding
    3. Active site volume changes
    
    This module learns to predict:
    - How the enzyme adapts to different substrates
    - Which enzyme conformations favor which SoMs
    """
    
    def __init__(
        self,
        mol_dim: int = 64,
        enzyme_dim: int = 32,
        n_conformations: int = 4,
    ):
        super().__init__()
        
        self.n_conformations = n_conformations
        
        # Enzyme conformation prototypes (learned)
        self.conformation_embeddings = nn.Parameter(
            torch.randn(n_conformations, enzyme_dim)
        )
        
        # Predict which enzyme conformation for given substrate
        self.conformation_predictor = nn.Sequential(
            nn.Linear(mol_dim, enzyme_dim),
            nn.SiLU(),
            nn.Linear(enzyme_dim, n_conformations),
        )
        
        # How each enzyme conformation affects atom accessibility
        self.accessibility_modifier = nn.Sequential(
            nn.Linear(enzyme_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        
        # Enzyme-substrate compatibility
        self.compatibility_net = nn.Sequential(
            nn.Linear(mol_dim + enzyme_dim, enzyme_dim),
            nn.SiLU(),
            nn.Linear(enzyme_dim, 1),
        )
    
    def forward(
        self,
        mol_features: torch.Tensor,  # [B, mol_dim] - global molecule features
        atom_features: torch.Tensor,  # [B, N, hidden_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict enzyme conformation and its effect on accessibility.
        """
        B, N, _ = atom_features.shape
        
        # Predict enzyme conformation distribution
        conf_logits = self.conformation_predictor(mol_features)  # [B, n_conf]
        conf_probs = F.softmax(conf_logits, dim=-1)  # [B, n_conf]
        
        # Weighted enzyme state
        enzyme_state = torch.einsum(
            'bc,cd->bd',
            conf_probs,
            self.conformation_embeddings
        )  # [B, enzyme_dim]
        
        # Accessibility modification per atom
        # Different enzyme conformations expose different parts of active site
        enzyme_expanded = enzyme_state.unsqueeze(1).expand(-1, N, -1)  # [B, N, enzyme_dim]
        accessibility_mod = self.accessibility_modifier(enzyme_expanded).squeeze(-1)  # [B, N]
        
        # Enzyme-substrate compatibility score
        compatibility = self.compatibility_net(
            torch.cat([mol_features, enzyme_state], dim=-1)
        ).squeeze(-1)  # [B]
        
        return {
            'enzyme_state': enzyme_state,
            'conformation_probs': conf_probs,
            'accessibility_modifier': accessibility_mod,
            'compatibility': compatibility,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED DYNAMIC SOM PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicSoMPredictor(nn.Module):
    """
    Full model combining:
    1. Substrate dynamics (multiple binding poses)
    2. Enzyme dynamics (conformational flexibility)
    3. Chemical reactivity
    
    Prediction: P(SoM at i) = Σ_pose Σ_enz_conf P(pose) × P(enz_conf|mol) × R(i|pose,conf)
    """
    
    def __init__(
        self,
        atom_dim: int = 128,
        hidden_dim: int = 64,
        n_poses: int = 10,
        n_enzyme_confs: int = 4,
    ):
        super().__init__()
        
        self.n_poses = n_poses
        
        # Atom encoder
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Global molecule encoder
        self.mol_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Dynamic reactivity model
        self.dynamic_model = DynamicReactivityModel(
            atom_dim=atom_dim,
            hidden_dim=hidden_dim,
            n_poses=n_poses,
        )
        
        # Enzyme dynamics
        self.enzyme_model = EnzymeDynamicsModel(
            mol_dim=hidden_dim,
            enzyme_dim=hidden_dim // 2,
            n_conformations=n_enzyme_confs,
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        atom_features: torch.Tensor,  # [B, N, atom_dim]
        pose_features: torch.Tensor,  # [B, N, n_poses, 3]
        valid_mask: torch.Tensor,     # [B, N]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with full dynamics."""
        B, N, _ = atom_features.shape
        
        # Encode atoms
        atom_encoded = self.atom_encoder(atom_features)  # [B, N, hidden]
        
        # Global molecule representation
        atom_masked = atom_encoded.masked_fill(~valid_mask.unsqueeze(-1), 0)
        mol_global = atom_masked.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mol_encoded = self.mol_encoder(mol_global)  # [B, hidden]
        
        # Dynamic reactivity (pose-dependent)
        dynamic_out = self.dynamic_model(atom_features, pose_features, valid_mask)
        
        # Enzyme dynamics
        enzyme_out = self.enzyme_model(mol_encoded, atom_encoded)
        
        # Combine everything
        mol_expanded = mol_encoded.unsqueeze(1).expand(-1, N, -1)
        enzyme_mod = enzyme_out['accessibility_modifier'].unsqueeze(-1)
        
        combined = torch.cat([
            atom_encoded,
            mol_expanded,
            enzyme_mod,
        ], dim=-1)
        
        base_scores = self.fusion(combined).squeeze(-1)  # [B, N]
        
        # Final scores = base + dynamic (pose-weighted)
        final_scores = base_scores + dynamic_out['dynamic_scores']
        
        # Apply enzyme accessibility modifier
        final_scores = final_scores + enzyme_out['accessibility_modifier']
        
        # Mask
        final_scores = final_scores.masked_fill(~valid_mask, -1e4)
        
        return {
            'final_scores': final_scores,
            'dynamic_scores': dynamic_out['dynamic_scores'],
            'intrinsic_scores': dynamic_out['intrinsic_scores'],
            'pose_weights': dynamic_out['pose_weights'],
            'enzyme_conformation': enzyme_out['conformation_probs'],
            'accessibility_modifier': enzyme_out['accessibility_modifier'],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: EXTRACT POSE FEATURES FROM ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pose_features(
    mol,
    n_poses: int = 10,
    max_atoms: int = 200,
) -> torch.Tensor:
    """
    Generate pose features for a molecule.
    
    Returns: [max_atoms, n_poses, 3] tensor with:
        - fe_distance (normalized)
        - is_accessible (0/1)
        - pose_probability
    """
    ensemble = PoseEnsemble(n_poses=n_poses)
    poses = ensemble.generate_poses(mol)
    
    features = torch.zeros(max_atoms, n_poses, 3)
    
    if not poses:
        return features
    
    n_atoms = mol.GetNumAtoms()
    
    for p_idx, pose in enumerate(poses[:n_poses]):
        if p_idx >= n_poses:
            break
        
        for a_idx in range(min(n_atoms, max_atoms)):
            # Normalized Fe distance
            fe_dist = pose.fe_distances[a_idx] if a_idx < len(pose.fe_distances) else 10.0
            features[a_idx, p_idx, 0] = fe_dist / 10.0  # Normalize
            
            # Is accessible
            features[a_idx, p_idx, 1] = 1.0 if a_idx in pose.accessible_atoms else 0.0
            
            # Pose probability
            features[a_idx, p_idx, 2] = pose.pose_probability
    
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Dynamic Binding Pose Model...")
    
    if HAS_RDKIT:
        # Test pose generation
        mol = Chem.MolFromSmiles("c1ccc(CC(=O)O)cc1")  # Phenylacetic acid
        ensemble = PoseEnsemble(n_poses=10)
        poses = ensemble.generate_poses(mol)
        
        print(f"\nGenerated {len(poses)} poses for phenylacetic acid")
        for i, pose in enumerate(poses[:3]):
            print(f"  Pose {i}: prob={pose.pose_probability:.3f}, "
                  f"accessible={pose.accessible_atoms}, "
                  f"score={pose.interaction_score:.2f}")
        
        # Test feature extraction
        pose_features = extract_pose_features(mol, n_poses=10, max_atoms=50)
        print(f"\nPose features shape: {pose_features.shape}")
    
    # Test neural network
    print("\nTesting DynamicSoMPredictor...")
    
    model = DynamicSoMPredictor(
        atom_dim=128,
        hidden_dim=64,
        n_poses=10,
        n_enzyme_confs=4,
    )
    
    B, N = 4, 50
    atom_features = torch.randn(B, N, 128)
    pose_features = torch.rand(B, N, 10, 3)
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    
    outputs = model(atom_features, pose_features, valid_mask)
    
    print(f"Final scores shape: {outputs['final_scores'].shape}")
    print(f"Pose weights shape: {outputs['pose_weights'].shape}")
    print(f"Enzyme conformations: {outputs['enzyme_conformation'][0]}")
    
    print("\n✓ Dynamic Binding Model test passed!")
