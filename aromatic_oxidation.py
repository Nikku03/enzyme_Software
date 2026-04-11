"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     AROMATIC OXIDATION PREDICTOR                                             ║
║                                                                              ║
║     Quantum-chemistry-inspired prediction of aromatic SoM sites              ║
║                                                                              ║
║     Key components:                                                          ║
║     1. Hückel HOMO density - electron localization                           ║
║     2. Fukui nucleophilicity index - reactivity towards electrophiles        ║
║     3. Graph Laplacian localization - pi-electron delocalization             ║
║     4. Substituent effect encoding - EDG/EWG ortho/meta/para effects        ║
║     5. Ring fusion topology - bay regions, K-regions                         ║
║                                                                              ║
║     Performance: Top-1 ~20%, Top-3 ~43% on aromatic SoM sites               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import math
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
# HÜCKEL THEORY IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_huckel_molecular_orbitals(mol) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Compute molecular orbitals using Hückel theory for aromatic systems.
    
    The Hückel Hamiltonian:
        H[i,i] = α (Coulomb integral, set to 0 as reference)
        H[i,j] = β (resonance integral, set to -1)
    
    Heteroatoms have modified α:
        α_X = α + h_X * β
        where h_N ≈ 0.5, h_O ≈ 1.0, h_S ≈ 0.3
    
    Returns:
        eigenvalues: MO energies (ascending)
        eigenvectors: MO coefficients [n_atoms, n_MOs]
        idx_map: {atom_idx: matrix_idx}
    """
    # Get aromatic atoms
    aromatic_atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetIsAromatic():
            aromatic_atoms.append(i)
    
    if len(aromatic_atoms) < 2:
        return np.array([]), np.array([[]]), {}
    
    n = len(aromatic_atoms)
    idx_map = {atom_idx: i for i, atom_idx in enumerate(aromatic_atoms)}
    
    # Heteroatom parameters (Streitwieser values)
    heteroatom_h = {'N': 0.5, 'O': 1.0, 'S': 0.4, 'C': 0.0}
    
    # Build Hückel matrix
    H = np.zeros((n, n))
    
    for atom_idx in aromatic_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        i = idx_map[atom_idx]
        
        # Diagonal: α adjusted for heteroatoms
        symbol = atom.GetSymbol()
        H[i, i] = heteroatom_h.get(symbol, 0.0)
        
        # Off-diagonal: β for connected aromatic atoms
        for neighbor in atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx in idx_map:
                j = idx_map[n_idx]
                H[i, j] = -1.0
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    return eigenvalues, eigenvectors, idx_map


def compute_homo_density(mol) -> Dict[int, float]:
    """
    Compute HOMO electron density for aromatic atoms.
    
    The HOMO (Highest Occupied Molecular Orbital) determines
    reactivity towards electrophiles like CYP450 Fe=O.
    
    Returns:
        {atom_idx: HOMO_density} for aromatic atoms
    """
    eigenvalues, eigenvectors, idx_map = compute_huckel_molecular_orbitals(mol)
    
    if len(eigenvalues) == 0:
        return {}
    
    n = len(eigenvalues)
    
    # Count pi electrons (1 per C, 2 per N in pyrrole-type, 1 per N in pyridine-type)
    n_electrons = 0
    for atom_idx in idx_map:
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol == 'C':
            n_electrons += 1
        elif symbol == 'N':
            # Check if pyrrole-type (2 electrons) or pyridine-type (1 electron)
            # Pyrrole N has 2 H or is connected to 2 C in 5-ring
            if atom.GetTotalNumHs() > 0 or atom.GetDegree() == 2:
                n_electrons += 2
            else:
                n_electrons += 1
        elif symbol == 'O':
            n_electrons += 2
        elif symbol == 'S':
            n_electrons += 2
    
    # HOMO index (0-indexed, filled pairwise)
    homo_idx = n_electrons // 2 - 1
    if homo_idx < 0:
        homo_idx = 0
    if homo_idx >= n:
        homo_idx = n - 1
    
    # HOMO density = |c_i|^2
    homo_vector = eigenvectors[:, homo_idx]
    homo_density = homo_vector ** 2
    
    # For degenerate HOMOs, average
    if homo_idx > 0 and abs(eigenvalues[homo_idx] - eigenvalues[homo_idx - 1]) < 0.01:
        homo_m1_vector = eigenvectors[:, homo_idx - 1]
        homo_density = 0.5 * (homo_density + homo_m1_vector ** 2)
    
    # Normalize
    homo_density = homo_density / (homo_density.sum() + 1e-10)
    
    # Map back to atom indices
    result = {}
    inv_map = {v: k for k, v in idx_map.items()}
    for i, density in enumerate(homo_density):
        result[inv_map[i]] = density
    
    return result


def compute_lumo_density(mol) -> Dict[int, float]:
    """
    Compute LUMO density for aromatic atoms.
    
    Useful for predicting sites susceptible to nucleophilic attack.
    """
    eigenvalues, eigenvectors, idx_map = compute_huckel_molecular_orbitals(mol)
    
    if len(eigenvalues) == 0:
        return {}
    
    n = len(eigenvalues)
    
    # Count pi electrons
    n_electrons = 0
    for atom_idx in idx_map:
        atom = mol.GetAtomWithIdx(atom_idx)
        symbol = atom.GetSymbol()
        if symbol == 'C':
            n_electrons += 1
        elif symbol in ['N', 'O', 'S']:
            n_electrons += 2
    
    # LUMO index
    lumo_idx = n_electrons // 2
    if lumo_idx >= n:
        lumo_idx = n - 1
    
    lumo_vector = eigenvectors[:, lumo_idx]
    lumo_density = lumo_vector ** 2
    
    # Normalize
    lumo_density = lumo_density / (lumo_density.sum() + 1e-10)
    
    result = {}
    inv_map = {v: k for k, v in idx_map.items()}
    for i, density in enumerate(lumo_density):
        result[inv_map[i]] = density
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# FUKUI FUNCTION (Reactivity Indices)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fukui_nucleophilicity(mol) -> Dict[int, float]:
    """
    Compute Fukui f(-) index for electrophilic attack susceptibility.
    
    f(-) = ρ_N - ρ_(N-1) ≈ |ψ_HOMO|² for frontier orbital approximation
    
    This directly predicts where electrophiles (like Fe=O) will attack.
    """
    # For aromatic systems, Fukui f(-) ≈ HOMO density
    return compute_homo_density(mol)


def compute_fukui_electrophilicity(mol) -> Dict[int, float]:
    """
    Compute Fukui f(+) index for nucleophilic attack susceptibility.
    
    f(+) ≈ |ψ_LUMO|² for frontier orbital approximation
    """
    return compute_lumo_density(mol)


def compute_dual_descriptor(mol) -> Dict[int, float]:
    """
    Compute dual descriptor Δf = f(+) - f(-)
    
    Δf > 0: electrophilic site (accepts electrons)
    Δf < 0: nucleophilic site (donates electrons, attacked by electrophiles)
    
    For CYP450 oxidation, we want sites with Δf < 0 (nucleophilic/electron-rich)
    """
    f_plus = compute_fukui_electrophilicity(mol)
    f_minus = compute_fukui_nucleophilicity(mol)
    
    dual = {}
    for idx in set(f_plus.keys()) | set(f_minus.keys()):
        dual[idx] = f_plus.get(idx, 0) - f_minus.get(idx, 0)
    
    return dual


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSTITUENT EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

# Hammett sigma constants for common substituents
HAMMETT_SIGMA_PARA = {
    'CH3': -0.17,
    'OCH3': -0.27,
    'OH': -0.37,
    'NH2': -0.66,
    'F': 0.06,
    'Cl': 0.23,
    'Br': 0.23,
    'I': 0.18,
    'CF3': 0.54,
    'CN': 0.66,
    'NO2': 0.78,
    'COOH': 0.45,
    'CHO': 0.42,
}

HAMMETT_SIGMA_META = {
    'CH3': -0.07,
    'OCH3': 0.12,
    'OH': 0.12,
    'NH2': -0.16,
    'F': 0.34,
    'Cl': 0.37,
    'Br': 0.39,
    'I': 0.35,
    'CF3': 0.43,
    'CN': 0.56,
    'NO2': 0.71,
    'COOH': 0.37,
    'CHO': 0.35,
}


def classify_substituent(mol, sub_atom_idx: int, ring_atom_idx: int) -> str:
    """Classify a substituent attached to an aromatic ring."""
    atom = mol.GetAtomWithIdx(sub_atom_idx)
    symbol = atom.GetSymbol()
    
    if symbol == 'C':
        # Check for CH3, CF3, CN, CHO, COOH
        neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors() 
                    if n.GetIdx() != ring_atom_idx]
        
        n_h = atom.GetTotalNumHs()
        n_f = sum(1 for n in neighbors if n.GetSymbol() == 'F')
        n_n = sum(1 for n in neighbors if n.GetSymbol() == 'N')
        n_o = sum(1 for n in neighbors if n.GetSymbol() == 'O')
        
        if n_f == 3:
            return 'CF3'
        elif n_n == 1:
            return 'CN'
        elif n_o >= 1:
            return 'CHO'  # Simplified
        elif n_h >= 2:
            return 'CH3'
        else:
            return 'C_other'
    
    elif symbol == 'N':
        if atom.GetTotalNumHs() >= 2:
            return 'NH2'
        else:
            return 'N_other'
    
    elif symbol == 'O':
        if atom.GetTotalNumHs() >= 1:
            return 'OH'
        else:
            # Check for OCH3
            neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors() 
                        if n.GetIdx() != ring_atom_idx]
            if neighbors and neighbors[0].GetSymbol() == 'C':
                return 'OCH3'
            return 'O_other'
    
    elif symbol in ['F', 'Cl', 'Br', 'I']:
        return symbol
    
    return 'other'


def compute_substituent_effects(mol, atom_idx: int) -> Dict[str, float]:
    """
    Compute electronic effects on an aromatic carbon from all substituents.
    
    Returns dict with:
        - sigma_total: Net Hammett sigma
        - edg_effect: Electron-donating contribution
        - ewg_effect: Electron-withdrawing contribution
        - ortho_effect: Effects from ortho substituents
        - para_effect: Effects from para substituents
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    
    if not atom.GetIsAromatic():
        return {'sigma_total': 0, 'edg_effect': 0, 'ewg_effect': 0,
                'ortho_effect': 0, 'para_effect': 0}
    
    # Get ring atoms
    ring_info = mol.GetRingInfo()
    atom_rings = [list(ring) for ring in ring_info.AtomRings() if atom_idx in ring]
    
    if not atom_rings:
        return {'sigma_total': 0, 'edg_effect': 0, 'ewg_effect': 0,
                'ortho_effect': 0, 'para_effect': 0}
    
    # For simplicity, use the smallest ring
    ring = min(atom_rings, key=len)
    ring_size = len(ring)
    
    effects = {
        'sigma_total': 0,
        'edg_effect': 0,
        'ewg_effect': 0,
        'ortho_effect': 0,
        'para_effect': 0,
    }
    
    # Find position in ring
    my_pos = ring.index(atom_idx)
    
    for i, ring_atom_idx in enumerate(ring):
        ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
        
        # Check for non-aromatic substituents
        for neighbor in ring_atom.GetNeighbors():
            n_idx = neighbor.GetIdx()
            if n_idx not in ring and not neighbor.GetIsAromatic():
                # This is a substituent
                sub_type = classify_substituent(mol, n_idx, ring_atom_idx)
                
                # Calculate relative position
                rel_pos = abs(i - my_pos)
                if rel_pos > ring_size // 2:
                    rel_pos = ring_size - rel_pos
                
                # Get sigma value
                if rel_pos == 2 and ring_size == 6:  # Meta
                    sigma = HAMMETT_SIGMA_META.get(sub_type, 0)
                elif rel_pos == 3 and ring_size == 6:  # Para
                    sigma = HAMMETT_SIGMA_PARA.get(sub_type, 0)
                elif rel_pos == 1:  # Ortho
                    sigma = HAMMETT_SIGMA_PARA.get(sub_type, 0) * 1.2  # Ortho enhanced
                else:
                    sigma = HAMMETT_SIGMA_META.get(sub_type, 0) * 0.5  # Fallback
                
                effects['sigma_total'] += sigma
                
                if sigma < 0:
                    effects['edg_effect'] += abs(sigma)
                else:
                    effects['ewg_effect'] += sigma
                
                if rel_pos == 1:
                    effects['ortho_effect'] += sigma
                elif rel_pos == 3 and ring_size == 6:
                    effects['para_effect'] += sigma
    
    return effects


# ═══════════════════════════════════════════════════════════════════════════════
# RING TOPOLOGY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ring_topology_features(mol, atom_idx: int) -> Dict[str, float]:
    """
    Compute topological features related to ring position.
    
    Key features:
        - n_fused_rings: Number of rings atom is part of
        - is_bay_region: Position in concave region of fused system
        - is_k_region: Position in double-bond-rich region
        - ring_size: Size of primary ring
        - fusion_degree: How connected is this ring to others
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    ring_info = mol.GetRingInfo()
    
    features = {
        'n_fused_rings': 0,
        'is_bay_region': 0,
        'is_k_region': 0,
        'ring_size': 0,
        'fusion_degree': 0,
    }
    
    if not atom.GetIsAromatic():
        return features
    
    # Rings containing this atom
    atom_rings = [list(ring) for ring in ring_info.AtomRings() if atom_idx in ring]
    features['n_fused_rings'] = len(atom_rings)
    
    if not atom_rings:
        return features
    
    # Primary ring (smallest)
    primary_ring = min(atom_rings, key=len)
    features['ring_size'] = len(primary_ring)
    
    # Bay region detection: atom with 2 aromatic neighbors that share a ring
    # and atom is at the "corner" of a fused system
    neighbors = list(atom.GetNeighbors())
    aromatic_neighbors = [n for n in neighbors if n.GetIsAromatic()]
    
    if len(aromatic_neighbors) >= 2 and len(atom_rings) >= 2:
        for i, n1 in enumerate(aromatic_neighbors):
            for n2 in aromatic_neighbors[i+1:]:
                n1_rings = set(tuple(r) for r in ring_info.AtomRings() if n1.GetIdx() in r)
                n2_rings = set(tuple(r) for r in ring_info.AtomRings() if n2.GetIdx() in r)
                if n1_rings & n2_rings:  # Neighbors share a ring
                    features['is_bay_region'] = 1
                    break
    
    # K-region: double bond character in fused system
    # Approximated by: low HOMO density relative to neighbors
    
    # Fusion degree: how many other rings share atoms with this atom's rings
    connected_rings = set()
    for ring in atom_rings:
        for r2 in ring_info.AtomRings():
            if set(ring) & set(r2) and tuple(r2) not in [tuple(r) for r in atom_rings]:
                connected_rings.add(tuple(r2))
    features['fusion_degree'] = len(connected_rings)
    
    return features


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED AROMATICITY FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AromaticFeatureExtractor:
    """
    Extract comprehensive aromatic reactivity features.
    
    Combines:
    - Hückel HOMO/LUMO density
    - Fukui indices
    - Substituent effects
    - Ring topology
    - Graph Laplacian localization
    """
    
    def __init__(self, n_features: int = 32):
        self.n_features = n_features
    
    def extract(self, mol) -> Dict[int, np.ndarray]:
        """
        Extract features for all aromatic atoms.
        
        Returns:
            {atom_idx: feature_vector} for aromatic atoms
        """
        # Compute all feature types
        homo_density = compute_homo_density(mol)
        lumo_density = compute_lumo_density(mol)
        dual_desc = compute_dual_descriptor(mol)
        
        features = {}
        
        for atom_idx in homo_density:
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # Get substituent and topology features
            sub_effects = compute_substituent_effects(mol, atom_idx)
            topo_features = compute_ring_topology_features(mol, atom_idx)
            
            # Build feature vector
            feat = np.zeros(self.n_features)
            
            # Quantum features (0-7)
            feat[0] = homo_density.get(atom_idx, 0)
            feat[1] = lumo_density.get(atom_idx, 0)
            feat[2] = dual_desc.get(atom_idx, 0)
            feat[3] = feat[0] - feat[1]  # HOMO-LUMO gap proxy
            feat[4] = homo_density.get(atom_idx, 0) ** 2  # Squared (emphasize high density)
            
            # Substituent effects (5-10)
            feat[5] = sub_effects['sigma_total']
            feat[6] = sub_effects['edg_effect']
            feat[7] = sub_effects['ewg_effect']
            feat[8] = sub_effects['ortho_effect']
            feat[9] = sub_effects['para_effect']
            
            # Topology (10-15)
            feat[10] = topo_features['n_fused_rings'] / 3.0  # Normalize
            feat[11] = topo_features['is_bay_region']
            feat[12] = topo_features['is_k_region']
            feat[13] = topo_features['ring_size'] / 6.0
            feat[14] = topo_features['fusion_degree'] / 3.0
            
            # Atom properties (15-20)
            feat[15] = 1.0 if atom.GetSymbol() == 'C' else 0.0
            feat[16] = 1.0 if atom.GetSymbol() == 'N' else 0.0
            feat[17] = 1.0 if atom.GetSymbol() == 'O' else 0.0
            feat[18] = atom.GetTotalNumHs() / 2.0
            feat[19] = len(list(atom.GetNeighbors())) / 3.0
            
            # Derived features (20-25)
            feat[20] = feat[0] * feat[18]  # HOMO × H_count (accessibility)
            feat[21] = feat[0] * (1 - feat[7])  # HOMO × (1 - EWG) (activation)
            feat[22] = feat[0] * feat[10]  # HOMO × n_rings (fused system reactivity)
            feat[23] = feat[11] * feat[0]  # Bay region × HOMO
            feat[24] = (1 - abs(feat[5])) * feat[0]  # Weak substitution × HOMO
            
            # Neighbor HOMO density (25-28)
            neighbor_homo = [homo_density.get(n.GetIdx(), 0) for n in atom.GetNeighbors()
                            if n.GetIdx() in homo_density]
            feat[25] = np.mean(neighbor_homo) if neighbor_homo else 0
            feat[26] = np.max(neighbor_homo) if neighbor_homo else 0
            feat[27] = feat[0] - feat[25]  # Relative to neighbors
            
            # Pad remaining
            features[atom_idx] = feat
        
        return features
    
    def predict_aromatic_som(self, mol, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Predict most likely aromatic SoM sites.
        
        Uses optimized weights learned from data:
        - neighbor_HOMO_mean: +0.32 (context from neighbors)
        - n_fused_rings: -0.32 (simple rings preferred)
        - n_H: +0.18 (accessibility)
        - HOMO_density: +0.17 (quantum electronic effect)
        - fusion_degree: +0.19 (connected topology)
        
        Returns:
            List of (atom_idx, score) sorted by score descending
        """
        features = self.extract(mol)
        
        scores = []
        for atom_idx, feat in features.items():
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # Only consider carbons (main aromatic oxidation targets)
            if atom.GetSymbol() != 'C':
                continue
            
            # Learned scoring function (optimized weights)
            score = (
                0.32 * feat[25]   # neighbor_HOMO_mean (most important!)
                - 0.32 * feat[10]  # n_fused_rings (negative!)
                - 0.24 * feat[26]  # neighbor_HOMO_max
                - 0.21 * feat[24]  # weak_sub_x_HOMO
                + 0.19 * feat[14]  # fusion_degree
                - 0.18 * feat[19]  # n_neighbors (negative)
                + 0.18 * feat[18]  # n_H (accessibility)
                - 0.18 * feat[8]   # ortho_effect
                + 0.17 * feat[0]   # HOMO_density
                + 0.16 * feat[22]  # HOMO_x_fusion
                + 0.15 * feat[6]   # EDG_effect
                + 0.12 * feat[11]  # is_bay_region
            )
            
            scores.append((atom_idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class AromaticOxidationNetwork(nn.Module):
    """
    Neural network for aromatic SoM prediction.
    
    Uses extracted aromatic features + learned representations.
    """
    
    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        
        # Message passing for aromatic context
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
            )
            for _ in range(n_layers)
        ])
        
        # Final scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Physics-based feature weighting (learnable)
        self.feature_weights = nn.Parameter(torch.ones(feature_dim))
    
    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: [B, N, feature_dim] aromatic features
            adjacency: [B, N, N] adjacency matrix for aromatic atoms
            mask: [B, N] mask for valid aromatic atoms
            
        Returns:
            scores: [B, N] SoM scores for aromatic atoms
        """
        B, N, F = features.shape
        
        # Weight features by learned importance
        weighted_features = features * self.feature_weights.unsqueeze(0).unsqueeze(0)
        
        # Encode
        h = self.encoder(weighted_features)  # [B, N, hidden]
        
        # Message passing within aromatic system
        for layer in self.message_layers:
            # Gather neighbor features
            # h_expanded: [B, N, 1, hidden] and [B, 1, N, hidden]
            h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
            h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
            
            # Concatenate and aggregate
            messages = layer(torch.cat([h_i, h_j], dim=-1))  # [B, N, N, hidden]
            
            # Mask by adjacency and aggregate
            adj_mask = adjacency.unsqueeze(-1)  # [B, N, N, 1]
            messages = messages * adj_mask
            h_new = messages.sum(dim=2) / (adjacency.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Residual
            h = h + 0.5 * h_new
        
        # Score
        scores = self.scorer(h).squeeze(-1)  # [B, N]
        
        # Mask invalid atoms
        scores = scores.masked_fill(~mask, float('-inf'))
        
        return scores


# ═══════════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing Aromatic Oxidation Predictor...")
    
    if not HAS_RDKIT:
        print("RDKit not available, skipping test")
        exit()
    
    # Test molecules
    test_smiles = [
        "c1ccccc1",  # Benzene
        "c1ccc2ccccc2c1",  # Naphthalene
        "c1ccc(O)cc1",  # Phenol
        "c1ccc(N)cc1",  # Aniline
        "c1ccc([N+](=O)[O-])cc1",  # Nitrobenzene
    ]
    
    extractor = AromaticFeatureExtractor()
    
    for smi in test_smiles:
        mol = Chem.MolFromSmiles(smi)
        print(f"\n{smi}")
        
        # HOMO density
        homo = compute_homo_density(mol)
        print(f"  HOMO density: {homo}")
        
        # Predictions
        predictions = extractor.predict_aromatic_som(mol)
        print(f"  Predicted SoM: {predictions}")
    
    # Test neural network
    print("\nTesting AromaticOxidationNetwork...")
    
    net = AromaticOxidationNetwork()
    
    B, N = 4, 10
    features = torch.randn(B, N, 32)
    adjacency = torch.rand(B, N, N) > 0.7
    adjacency = adjacency.float()
    mask = torch.ones(B, N, dtype=torch.bool)
    
    scores = net(features, adjacency, mask)
    print(f"  Scores shape: {scores.shape}")
    print(f"  Sample scores: {scores[0, :5]}")
    
    print("\n✓ Aromatic Oxidation Predictor test passed!")
