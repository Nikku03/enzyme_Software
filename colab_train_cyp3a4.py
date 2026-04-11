#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║   CYP3A4 SITE-OF-METABOLISM PREDICTOR - COLAB TRAINING SCRIPT                           ║
║   ══════════════════════════════════════════════════════════════════════════════════════ ║
║                                                                                          ║
║   This script trains a Quantum Liquid Neural Network (Q-LNN) for predicting             ║
║   sites of metabolism in CYP3A4 substrates.                                             ║
║                                                                                          ║
║   KEY INNOVATION: Reverse Geometry Learning                                              ║
║   ─────────────────────────────────────────                                              ║
║   We INFER the enzyme pocket's dynamic behavior from known SoM data:                    ║
║   • If atom X was oxidized, it MUST have been near Fe during catalysis                  ║
║   • This constrains the binding pose                                                     ║
║   • The ML model learns what pocket configurations lead to oxidation                    ║
║                                                                                          ║
║   ARCHITECTURE:                                                                          ║
║   ─────────────                                                                          ║
║   1. Molecular Graph Encoder (GNN with Laplacian eigenvectors)                          ║
║   2. 3D Conformer Generator (RDKit)                                                      ║
║   3. Pocket Geometry Encoder (from 1W0E.pdb)                                            ║
║   4. Reverse Geometry Module (learns pocket state during bond breaking)                 ║
║   5. Liquid Neural Network (continuous-time ODE dynamics)                               ║
║   6. SoM Ranking Head                                                                    ║
║                                                                                          ║
║   USAGE (Google Colab):                                                                  ║
║   ─────────────────────                                                                  ║
║   !git clone https://github.com/Naresh/enzyme_Software.git                              ║
║   %cd enzyme_Software                                                                    ║
║   !pip install torch torch-geometric rdkit-pypi                                         ║
║   !python colab_train_cyp3a4.py --epochs 100 --batch_size 32                            ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Training configuration."""
    # Data
    data_path: str = "data/curated/merged_cyp3a4_extended.json"
    enzyme_pdb: str = "1W0E.pdb"
    
    # Model
    atom_feat_dim: int = 64
    hidden_dim: int = 128
    n_lnn_layers: int = 3
    n_gnn_layers: int = 4
    n_attention_heads: int = 4
    pocket_feat_dim: int = 64
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Loss weights
    ranking_weight: float = 1.0
    contrastive_weight: float = 0.5
    reverse_geometry_weight: float = 0.3
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Logging
    log_interval: int = 10
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints"


# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME POCKET GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

class EnzymePocket:
    """
    Represents the CYP3A4 enzyme pocket geometry.
    
    Key insight: During bond breaking, the pocket has a specific "reactive state"
    where the substrate atom is positioned ~3-5Å from Fe along the Fe=O axis.
    
    We encode this as learnable features that the model can use.
    """
    
    def __init__(self, pdb_path: str = "1W0E.pdb"):
        self.fe_pos = np.array([54.95, 77.69, 10.64])  # Fe center
        self.pocket_atoms = []
        self.pocket_residues = {}
        
        self._load_pocket(pdb_path)
        self._compute_pocket_features()
    
    def _load_pocket(self, pdb_path: str):
        """Load pocket atoms from PDB."""
        if not os.path.exists(pdb_path):
            print(f"Warning: {pdb_path} not found, using default pocket")
            self._create_default_pocket()
            return
            
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_name = line[12:16].strip()
                    res_name = line[17:20].strip()
                    res_num = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    pos = np.array([x, y, z])
                    dist = np.linalg.norm(pos - self.fe_pos)
                    
                    if 3.0 < dist < 15.0:
                        is_hydrophobic = res_name in [
                            'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'
                        ]
                        is_polar = res_name in ['SER', 'THR', 'ASN', 'GLN', 'TYR']
                        is_charged = res_name in ['ARG', 'LYS', 'ASP', 'GLU', 'HIS']
                        
                        self.pocket_atoms.append({
                            'pos': pos,
                            'name': atom_name,
                            'res': res_name,
                            'res_num': res_num,
                            'dist': dist,
                            'hydrophobic': is_hydrophobic,
                            'polar': is_polar,
                            'charged': is_charged,
                        })
                        
                        key = (res_name, res_num)
                        if key not in self.pocket_residues:
                            self.pocket_residues[key] = []
                        self.pocket_residues[key].append(pos)
        
        print(f"Loaded {len(self.pocket_atoms)} pocket atoms, "
              f"{len(self.pocket_residues)} residues")
    
    def _create_default_pocket(self):
        """Create a default spherical pocket if PDB not available."""
        for i in range(100):
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            r = np.random.uniform(4, 12)
            
            pos = self.fe_pos + r * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            
            self.pocket_atoms.append({
                'pos': pos,
                'dist': r,
                'hydrophobic': np.random.random() > 0.4,
                'polar': np.random.random() > 0.7,
                'charged': np.random.random() > 0.9,
            })
    
    def _compute_pocket_features(self):
        """Compute pocket geometric features."""
        if not self.pocket_atoms:
            return
            
        positions = np.array([a['pos'] for a in self.pocket_atoms])
        
        # Center on Fe
        self.pocket_centered = positions - self.fe_pos
        
        # PCA of pocket shape
        cov = np.cov(self.pocket_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        self.pocket_axes = eigvecs[:, order]
        self.pocket_extent = np.sqrt(eigvals[order])
        
        # Pocket centroid direction (main substrate entry)
        self.centroid = self.pocket_centered.mean(axis=0)
        self.feo_axis = self.centroid / (np.linalg.norm(self.centroid) + 1e-8)
        
        # Shell-based hydrophobicity profile
        self.shell_hydro = []
        for r_min, r_max in [(4, 6), (6, 8), (8, 10), (10, 12)]:
            mask = [(r_min < a['dist'] < r_max) for a in self.pocket_atoms]
            if sum(mask) > 0:
                hydro = sum(a['hydrophobic'] for a, m in zip(self.pocket_atoms, mask) if m)
                self.shell_hydro.append(hydro / sum(mask))
            else:
                self.shell_hydro.append(0.5)
        
        print(f"Pocket axes computed, Fe=O axis: {self.feo_axis}")
        print(f"Shell hydrophobicity: {self.shell_hydro}")
    
    def get_pocket_tensor(self, device: str = "cpu") -> torch.Tensor:
        """
        Get pocket features as a tensor.
        
        Returns: (N_pocket, pocket_feat_dim) tensor
        """
        features = []
        for atom in self.pocket_atoms:
            pos_centered = atom['pos'] - self.fe_pos
            pos_normalized = pos_centered / (np.linalg.norm(pos_centered) + 1e-8)
            
            feat = np.concatenate([
                pos_centered / 10.0,  # Normalized position (3)
                pos_normalized,  # Unit direction (3)
                [atom['dist'] / 15.0],  # Distance to Fe (1)
                [1.0 if atom['hydrophobic'] else 0.0],  # (1)
                [1.0 if atom.get('polar', False) else 0.0],  # (1)
                [1.0 if atom.get('charged', False) else 0.0],  # (1)
                self.feo_axis,  # Fe=O axis direction (3)
                [np.dot(pos_normalized, self.feo_axis)],  # Alignment with axis (1)
            ])
            features.append(feat)
        
        return torch.tensor(np.array(features), dtype=torch.float32, device=device)
    
    def compute_accessibility(
        self, 
        mol_coords: np.ndarray, 
        mol_features: np.ndarray
    ) -> np.ndarray:
        """
        Compute accessibility of each atom to the Fe center.
        
        This is the "reverse geometry" - given that oxidation occurred,
        what was the pocket state?
        
        Returns: (N_atoms,) accessibility scores
        """
        n = len(mol_coords)
        accessibility = np.zeros(n)
        
        pocket_pos = np.array([a['pos'] for a in self.pocket_atoms])
        
        # For each atom, compute:
        # 1. Distance to Fe (optimal: 3-5Å)
        # 2. Alignment with Fe=O axis
        # 3. Steric accessibility (not blocked by pocket)
        
        for i in range(n):
            pos = mol_coords[i]
            
            # Distance to Fe
            fe_dist = np.linalg.norm(pos - self.fe_pos)
            if fe_dist < 2.0 or fe_dist > 8.0:
                continue
            
            # Optimal distance is 3.5-4.5Å
            dist_score = np.exp(-((fe_dist - 4.0) ** 2) / 2.0)
            
            # Alignment with Fe=O axis
            direction = (pos - self.fe_pos) / fe_dist
            alignment = np.dot(direction, self.feo_axis)
            align_score = max(0, alignment) ** 2
            
            # Steric accessibility - check path to Fe
            steric_score = 1.0
            for t in [0.3, 0.5, 0.7]:
                test_pos = pos + t * (self.fe_pos - pos)
                # Check distance to nearest pocket atom
                dists = np.linalg.norm(pocket_pos - test_pos, axis=1)
                if dists.min() < 2.0:
                    steric_score *= 0.5
            
            accessibility[i] = dist_score * align_score * steric_score
        
        return accessibility


# ═══════════════════════════════════════════════════════════════════════════════
# MOLECULAR FEATURIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_atom_features(atom, mol) -> np.ndarray:
    """Compute atom-level features."""
    # Basic properties
    z = atom.GetAtomicNum()
    degree = atom.GetDegree()
    n_H = atom.GetTotalNumHs()
    formal_charge = atom.GetFormalCharge()
    is_aromatic = atom.GetIsAromatic()
    in_ring = atom.IsInRing()
    
    # Hybridization one-hot
    hyb = atom.GetHybridization()
    hyb_onehot = [
        hyb == Chem.HybridizationType.SP,
        hyb == Chem.HybridizationType.SP2,
        hyb == Chem.HybridizationType.SP3,
        hyb == Chem.HybridizationType.SP3D,
        hyb == Chem.HybridizationType.SP3D2,
    ]
    
    # Element one-hot (C, N, O, S, other)
    elem_onehot = [z == 6, z == 7, z == 8, z == 16, z not in [6, 7, 8, 16]]
    
    # Neighbor types
    neighbors = list(atom.GetNeighbors())
    has_O_neighbor = any(n.GetAtomicNum() == 8 for n in neighbors)
    has_N_neighbor = any(n.GetAtomicNum() == 7 for n in neighbors)
    has_arom_neighbor = any(n.GetIsAromatic() for n in neighbors) and not is_aromatic
    
    # Chemistry boost (alpha-het, benzylic)
    alpha_het_score = 2.45 if has_O_neighbor else (2.06 if has_N_neighbor else 1.0)
    benzylic_score = 1.25 if has_arom_neighbor else 1.0
    
    features = np.array([
        z / 20.0,  # Normalized atomic number
        degree / 4.0,
        n_H / 4.0,
        formal_charge,
        float(is_aromatic),
        float(in_ring),
        alpha_het_score / 2.5,
        benzylic_score / 1.5,
    ] + hyb_onehot + elem_onehot + [
        float(has_O_neighbor),
        float(has_N_neighbor),
        float(has_arom_neighbor),
    ])
    
    return features.astype(np.float32)


def get_bond_features(bond) -> np.ndarray:
    """Compute bond-level features."""
    bond_type = bond.GetBondType()
    
    features = np.array([
        bond_type == Chem.BondType.SINGLE,
        bond_type == Chem.BondType.DOUBLE,
        bond_type == Chem.BondType.TRIPLE,
        bond_type == Chem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ], dtype=np.float32)
    
    return features


def compute_laplacian_features(mol, k: int = 8) -> np.ndarray:
    """
    Compute graph Laplacian eigenvector features.
    These encode the topological "peripherality" and "flexibility" of each atom.
    Always outputs (n, k + 2) features with padding for small molecules.
    """
    n = mol.GetNumAtoms()
    
    # Build adjacency matrix
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = 1.5 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble()
        A[i, j] = A[j, i] = w
    
    # Graph Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Eigendecomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.ones(n)
        eigvecs = np.eye(n)
    
    # FIXED: Always output k + 2 features (pad with zeros for small molecules)
    features = np.zeros((n, k + 2), dtype=np.float32)
    
    k_actual = min(k, n - 1)
    
    for i in range(n):
        # Peripherality: contribution to highest eigenvectors
        k_start = max(1, n - k_actual)
        features[i, 0] = sum(eigvecs[i, j]**2 for j in range(k_start, n))
        
        # Flexibility: weighted contribution to low eigenvectors
        k_end = min(k_actual, n)
        features[i, 1] = sum(eigvecs[i, j]**2 / (eigvals[j] + 0.1) for j in range(1, k_end))
        
        # Eigenvector components (pad with zeros if molecule is small)
        for j in range(min(k_actual, n - 1)):
            features[i, j + 2] = eigvecs[i, j + 1]  # Skip trivial eigenvector
    
    return features


def mol_to_graph_data(
    mol, 
    pocket: EnzymePocket,
    som_atoms: List[int] = None,
) -> Dict[str, Any]:
    """
    Convert molecule to graph data with 3D coordinates and pocket features.
    """
    n = mol.GetNumAtoms()
    
    # Atom features
    atom_features = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        feat = get_atom_features(atom, mol)
        atom_features.append(feat)
    atom_features = np.array(atom_features)
    
    # Laplacian features
    lap_features = compute_laplacian_features(mol)
    
    # Combine features
    x = np.concatenate([atom_features, lap_features], axis=1)
    
    # Edge index and features
    edges_src, edges_dst = [], []
    edge_features = []
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges_src.extend([i, j])
        edges_dst.extend([j, i])
        bf = get_bond_features(bond)
        edge_features.extend([bf, bf])  # Both directions
    
    edge_index = np.array([edges_src, edges_dst])
    edge_attr = np.array(edge_features) if edge_features else np.zeros((0, 6))
    
    # Generate 3D conformer
    mol_h = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=100)
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        conf = mol_h.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] for i in range(n)])
    except:
        # Fallback: use 2D coordinates
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           0.0] for i in range(n)])
    
    # Center coordinates
    coords = coords - coords.mean(axis=0)
    
    # Compute accessibility to Fe (reverse geometry signal)
    accessibility = pocket.compute_accessibility(coords, x)
    
    # Valid atom mask (atoms that can be SoM)
    valid_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        z = atom.GetAtomicNum()
        n_H = atom.GetTotalNumHs()
        is_arom = atom.GetIsAromatic()
        
        if z == 6 and (n_H > 0 or is_arom):
            valid_mask[i] = True
        elif z == 7 and not is_arom and atom.GetTotalNumHs() == 0:
            valid_mask[i] = True
        elif z == 16 and not is_arom:
            valid_mask[i] = True
    
    # SoM labels
    som_labels = np.zeros(n, dtype=np.float32)
    if som_atoms:
        for atom_idx in som_atoms:
            if atom_idx < n:
                som_labels[atom_idx] = 1.0
    
    return {
        'x': torch.tensor(x, dtype=torch.float32),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'edge_attr': torch.tensor(edge_attr, dtype=torch.float32),
        'coords': torch.tensor(coords, dtype=torch.float32),
        'accessibility': torch.tensor(accessibility, dtype=torch.float32),
        'valid_mask': torch.tensor(valid_mask, dtype=torch.bool),
        'som_labels': torch.tensor(som_labels, dtype=torch.float32),
        'n_atoms': n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class CYP3A4Dataset(Dataset):
    """Dataset for CYP3A4 SoM prediction."""
    
    def __init__(
        self, 
        data_path: str,
        pocket: EnzymePocket,
        sources: List[str] = None,
    ):
        self.pocket = pocket
        self.data = []
        
        # Load data
        with open(data_path) as f:
            raw_data = json.load(f)['drugs']
        
        # Filter by source
        if sources:
            raw_data = [d for d in raw_data if d.get('source') in sources]
        
        # Process molecules
        print(f"Processing {len(raw_data)} molecules...")
        for d in raw_data:
            smiles = d.get('smiles', '')
            sites = d.get('site_atoms', [])
            source = d.get('source', '')
            name = d.get('name', '')
            
            if not smiles or not sites:
                continue
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                graph_data = mol_to_graph_data(mol, pocket, sites)
                graph_data['smiles'] = smiles
                graph_data['source'] = source
                graph_data['name'] = name
                self.data.append(graph_data)
            except Exception as e:
                continue
        
        print(f"Loaded {len(self.data)} valid molecules")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching variable-size graphs."""
    # Stack features with padding
    max_atoms = max(d['n_atoms'] for d in batch)
    
    batch_x = []
    batch_coords = []
    batch_accessibility = []
    batch_valid_mask = []
    batch_som_labels = []
    batch_edge_index = []
    batch_edge_attr = []
    batch_ptr = [0]
    
    offset = 0
    for d in batch:
        n = d['n_atoms']
        
        # Pad features
        x_padded = F.pad(d['x'], (0, 0, 0, max_atoms - n))
        coords_padded = F.pad(d['coords'], (0, 0, 0, max_atoms - n))
        access_padded = F.pad(d['accessibility'], (0, max_atoms - n))
        valid_padded = F.pad(d['valid_mask'], (0, max_atoms - n))
        som_padded = F.pad(d['som_labels'], (0, max_atoms - n))
        
        batch_x.append(x_padded)
        batch_coords.append(coords_padded)
        batch_accessibility.append(access_padded)
        batch_valid_mask.append(valid_padded)
        batch_som_labels.append(som_padded)
        
        # Offset edge indices
        edge_index = d['edge_index'] + offset
        batch_edge_index.append(edge_index)
        batch_edge_attr.append(d['edge_attr'])
        
        offset += n
        batch_ptr.append(offset)
    
    return {
        'x': torch.stack(batch_x),  # (B, max_atoms, feat_dim)
        'coords': torch.stack(batch_coords),  # (B, max_atoms, 3)
        'accessibility': torch.stack(batch_accessibility),  # (B, max_atoms)
        'valid_mask': torch.stack(batch_valid_mask),  # (B, max_atoms)
        'som_labels': torch.stack(batch_som_labels),  # (B, max_atoms)
        'edge_index': torch.cat(batch_edge_index, dim=1),  # (2, total_edges)
        'edge_attr': torch.cat(batch_edge_attr, dim=0),  # (total_edges, edge_dim)
        'batch_ptr': torch.tensor(batch_ptr),  # (B+1,)
        'n_atoms': torch.tensor([d['n_atoms'] for d in batch]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class GraphConvLayer(nn.Module):
    """Graph convolution with edge features."""
    
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 6):
        super().__init__()
        self.lin_node = nn.Linear(in_dim, out_dim)
        self.lin_edge = nn.Linear(edge_dim, out_dim)
        self.lin_msg = nn.Linear(out_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x, edge_index, edge_attr):
        # x: (N, in_dim)
        # edge_index: (2, E)
        # edge_attr: (E, edge_dim)
        
        row, col = edge_index
        
        # Transform nodes and edges
        x_transformed = self.lin_node(x)  # (N, out_dim)
        edge_transformed = self.lin_edge(edge_attr)  # (E, out_dim)
        
        # Message passing
        x_j = x_transformed[col]  # Source node features
        messages = torch.cat([x_j, edge_transformed], dim=-1)
        messages = F.relu(self.lin_msg(messages))
        
        # Aggregate
        out = torch.zeros_like(x_transformed)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)
        
        # Normalize by degree
        degree = torch.zeros(x.size(0), device=x.device)
        degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        degree = degree.clamp(min=1)
        out = out / degree.unsqueeze(-1)
        
        return self.norm(out + x_transformed)


class LiquidCell(nn.Module):
    """
    Liquid Time-Constant Neural Network cell.
    
    Implements continuous-time ODE dynamics:
    dh/dt = (-h + f(Wx + Wh)) / tau
    
    Where tau is a learned, input-dependent time constant.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim)
        
        # Recurrent projection
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)
        
        # Time constant network (input-dependent tau)
        self.tau_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive tau
        )
        
        # Output gate
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, h, dt: float = 0.1):
        """
        Single step of liquid dynamics.
        
        Args:
            x: (B, N, input_dim) input features
            h: (B, N, hidden_dim) hidden state
            dt: integration time step
        
        Returns:
            h_new: (B, N, hidden_dim) updated hidden state
        """
        # Compute time constant
        tau_input = torch.cat([x, h], dim=-1)
        tau = self.tau_net(tau_input) + 0.5  # Minimum tau of 0.5
        
        # Compute update
        f_x = torch.tanh(self.W_in(x))
        f_h = torch.tanh(self.W_rec(h))
        f = f_x + f_h
        
        # ODE step: dh/dt = (-h + f) / tau
        dh = (-h + f) / tau
        h_new = h + dt * dh
        
        # Gated output
        gate = self.gate(h_new)
        h_out = gate * h_new
        
        return self.norm(h_out)


class ReverseGeometryModule(nn.Module):
    """
    Learns the enzyme pocket state during bond breaking.
    
    Key insight: If we know which atom was oxidized, we can infer
    how the molecule was positioned in the pocket. The model learns
    these patterns to predict SoM for new molecules.
    """
    
    def __init__(self, feat_dim: int, pocket_dim: int, hidden_dim: int):
        super().__init__()
        
        # Encode the "reactive state" - pocket configuration during catalysis
        self.reactive_state_encoder = nn.Sequential(
            nn.Linear(feat_dim + 3 + 1, hidden_dim),  # features + coords + accessibility
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Cross-attention to pocket
        self.pocket_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        
        # Predict "binding pose score" - how likely is this atom to be positioned near Fe?
        self.pose_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Project pocket features
        self.pocket_proj = nn.Linear(pocket_dim, hidden_dim)
    
    def forward(
        self, 
        x: torch.Tensor,  # (B, N, feat_dim)
        coords: torch.Tensor,  # (B, N, 3)
        accessibility: torch.Tensor,  # (B, N)
        pocket_feat: torch.Tensor,  # (N_pocket, pocket_dim)
    ) -> torch.Tensor:
        """
        Compute reverse geometry scores.
        
        Returns:
            (B, N) scores indicating how likely each atom is to be
            in the reactive position during catalysis.
        """
        B, N, _ = x.shape
        
        # Encode atom states
        atom_input = torch.cat([
            x, 
            coords, 
            accessibility.unsqueeze(-1)
        ], dim=-1)
        atom_encoded = self.reactive_state_encoder(atom_input)  # (B, N, hidden)
        
        # Project pocket features and expand for batch
        pocket_proj = self.pocket_proj(pocket_feat)  # (N_pocket, hidden)
        pocket_expanded = pocket_proj.unsqueeze(0).expand(B, -1, -1)  # (B, N_pocket, hidden)
        
        # Cross-attention: atoms attend to pocket
        attended, _ = self.pocket_attention(
            atom_encoded,  # query
            pocket_expanded,  # key
            pocket_expanded,  # value
        )  # (B, N, hidden)
        
        # Compute pose scores
        pose_scores = self.pose_scorer(attended).squeeze(-1)  # (B, N)
        
        return pose_scores


class CYP3A4SoMPredictor(nn.Module):
    """
    Full model for CYP3A4 site of metabolism prediction.
    
    Combines:
    1. Graph neural network for molecular encoding
    2. Liquid neural network for dynamic state
    3. Reverse geometry module for pocket-aware scoring
    4. Ranking head for final prediction
    """
    
    def __init__(self, config: Config, pocket: EnzymePocket):
        super().__init__()
        self.config = config
        
        # Get input dimensions
        # atom_features (21) + laplacian_features (10)
        input_dim = 21 + 10
        
        # Initial embedding
        self.embed = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(config.hidden_dim, config.hidden_dim, edge_dim=6)
            for _ in range(config.n_gnn_layers)
        ])
        
        # Liquid neural network layers
        self.lnn_layers = nn.ModuleList([
            LiquidCell(config.hidden_dim, config.hidden_dim)
            for _ in range(config.n_lnn_layers)
        ])
        
        # Reverse geometry module
        pocket_feat = pocket.get_pocket_tensor()
        pocket_dim = pocket_feat.shape[-1]
        self.register_buffer('pocket_feat', pocket_feat)
        
        self.reverse_geometry = ReverseGeometryModule(
            feat_dim=config.hidden_dim,
            pocket_dim=pocket_dim,
            hidden_dim=config.hidden_dim,
        )
        
        # Ranking head
        self.ranking_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + 1, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        
        # Chemistry boost head (learn chemistry multipliers)
        self.chemistry_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    
    def forward(
        self,
        x: torch.Tensor,  # (B, N, feat_dim)
        coords: torch.Tensor,  # (B, N, 3)
        accessibility: torch.Tensor,  # (B, N)
        valid_mask: torch.Tensor,  # (B, N)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: torch.Tensor,  # (E, edge_dim)
        batch_ptr: torch.Tensor,  # (B+1,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            scores: (B, N) SoM scores for each atom
            pose_scores: (B, N) reverse geometry scores
            chemistry_scores: (B, N) chemistry-based scores
        """
        B, N, _ = x.shape
        device = x.device
        
        # Initial embedding
        h = self.embed(x)  # (B, N, hidden)
        
        # Flatten for GNN
        h_flat = h.view(-1, h.size(-1))  # (B*N, hidden)
        
        # GNN layers
        for gnn in self.gnn_layers:
            h_flat = F.silu(gnn(h_flat, edge_index, edge_attr))
        
        # Reshape back
        h = h_flat.view(B, N, -1)  # (B, N, hidden)
        
        # Liquid dynamics - use h as both input and hidden state
        for lnn in self.lnn_layers:
            h = lnn(h, h, dt=0.1)
        
        # Reverse geometry scores
        pose_scores = self.reverse_geometry(
            h, coords, accessibility, 
            self.pocket_feat.to(device)
        )
        
        # Chemistry scores
        chemistry_scores = self.chemistry_head(h).squeeze(-1)  # (B, N)
        
        # Combined features for ranking
        combined = torch.cat([
            h,
            h * pose_scores.unsqueeze(-1),  # Pose-modulated features
            pose_scores.unsqueeze(-1),
        ], dim=-1)
        
        # Final ranking scores
        scores = self.ranking_head(combined).squeeze(-1)  # (B, N)
        
        # Apply chemistry boost
        scores = scores * chemistry_scores
        
        # Mask invalid atoms
        scores = scores.masked_fill(~valid_mask, float('-inf'))
        
        return scores, pose_scores, chemistry_scores


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class SoMLoss(nn.Module):
    """
    Combined loss for SoM prediction.
    
    Components:
    1. Ranking loss: SoM atoms should score higher than non-SoM
    2. Contrastive loss: Pull SoM atoms together, push non-SoM apart
    3. Reverse geometry loss: Pose scores should correlate with SoM labels
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.margin = 1.0
    
    def forward(
        self,
        scores: torch.Tensor,  # (B, N)
        pose_scores: torch.Tensor,  # (B, N)
        chemistry_scores: torch.Tensor,  # (B, N)
        som_labels: torch.Tensor,  # (B, N)
        valid_mask: torch.Tensor,  # (B, N)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Returns:
            total_loss: scalar
            loss_dict: breakdown of loss components
        """
        B, N = scores.shape
        device = scores.device
        
        # Replace -inf with large negative
        scores = torch.where(
            torch.isinf(scores),
            torch.full_like(scores, -100.0),
            scores
        )
        
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        
        # 1. Ranking loss
        ranking_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for b in range(B):
            mask = valid_mask[b]
            labels = som_labels[b]
            s = scores[b]
            
            som_mask = (labels > 0) & mask
            non_som_mask = (labels == 0) & mask
            
            if som_mask.sum() == 0 or non_som_mask.sum() == 0:
                continue
            
            # SoM atoms should score higher
            som_scores = s[som_mask]
            non_som_scores = s[non_som_mask]
            
            # Margin ranking loss
            for som_s in som_scores:
                for non_som_s in non_som_scores:
                    ranking_loss = ranking_loss + F.relu(self.margin - som_s + non_som_s)
                    count += 1
        
        if count > 0:
            ranking_loss = ranking_loss / count
        
        total_loss = total_loss + self.config.ranking_weight * ranking_loss
        loss_dict['ranking'] = ranking_loss.item()
        
        # 2. Contrastive loss (InfoNCE-style)
        contrastive_loss = torch.tensor(0.0, device=device)
        
        for b in range(B):
            mask = valid_mask[b]
            labels = som_labels[b]
            s = scores[b]
            
            som_mask = (labels > 0) & mask
            
            if som_mask.sum() == 0 or mask.sum() <= 1:
                continue
            
            # Softmax over valid atoms
            logits = s[mask]
            targets = labels[mask]
            
            if targets.sum() > 0:
                # Cross-entropy with soft targets
                probs = F.softmax(logits, dim=0)
                target_dist = targets / targets.sum()
                contrastive_loss = contrastive_loss - (target_dist * torch.log(probs + 1e-8)).sum()
        
        contrastive_loss = contrastive_loss / B
        total_loss = total_loss + self.config.contrastive_weight * contrastive_loss
        loss_dict['contrastive'] = contrastive_loss.item()
        
        # 3. Reverse geometry loss
        pose_loss = F.binary_cross_entropy_with_logits(
            pose_scores[valid_mask],
            som_labels[valid_mask],
        )
        
        total_loss = total_loss + self.config.reverse_geometry_weight * pose_loss
        loss_dict['reverse_geometry'] = pose_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_accuracy(
    scores: torch.Tensor,
    som_labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[float, float, float]:
    """Compute top-1, top-3, and top-5 accuracy."""
    B = scores.shape[0]
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = 0
    
    for b in range(B):
        mask = valid_mask[b]
        labels = som_labels[b]
        s = scores[b]
        
        if mask.sum() == 0 or labels.sum() == 0:
            continue
        
        # Get valid indices sorted by score
        valid_indices = torch.where(mask)[0]
        valid_scores = s[mask]
        sorted_indices = valid_indices[torch.argsort(valid_scores, descending=True)]
        
        # Check top-k
        som_indices = set(torch.where(labels > 0)[0].tolist())
        
        if sorted_indices[0].item() in som_indices:
            top1_correct += 1
        
        if len(sorted_indices) >= 3:
            if any(idx.item() in som_indices for idx in sorted_indices[:3]):
                top3_correct += 1
        
        if len(sorted_indices) >= 5:
            if any(idx.item() in som_indices for idx in sorted_indices[:5]):
                top5_correct += 1
        
        total += 1
    
    return (
        top1_correct / max(total, 1),
        top3_correct / max(total, 1),
        top5_correct / max(total, 1),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SoMLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Config,
    epoch: int,
    scaler: torch.amp.GradScaler,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    device = config.device
    
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    n_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        # Move to device
        x = batch['x'].to(device)
        coords = batch['coords'].to(device)
        accessibility = batch['accessibility'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        som_labels = batch['som_labels'].to(device)
        edge_index = batch['edge_index'].to(device)
        edge_attr = batch['edge_attr'].to(device)
        batch_ptr = batch['batch_ptr'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=config.mixed_precision):
            scores, pose_scores, chemistry_scores = model(
                x, coords, accessibility, valid_mask,
                edge_index, edge_attr, batch_ptr
            )
            
            loss, loss_dict = loss_fn(
                scores, pose_scores, chemistry_scores,
                som_labels, valid_mask
            )
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Compute accuracy
        with torch.no_grad():
            top1, top3, _ = compute_accuracy(scores, som_labels, valid_mask)
        
        total_loss += loss.item()
        total_top1 += top1
        total_top3 += top3
        n_batches += 1
        
        if batch_idx % config.log_interval == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: "
                  f"Loss={loss.item():.4f}, Top1={top1*100:.1f}%")
    
    scheduler.step()
    
    return {
        'loss': total_loss / n_batches,
        'top1': total_top1 / n_batches,
        'top3': total_top3 / n_batches,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: SoMLoss,
    config: Config,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    device = config.device
    
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_top5 = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            coords = batch['coords'].to(device)
            accessibility = batch['accessibility'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            som_labels = batch['som_labels'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr = batch['edge_attr'].to(device)
            batch_ptr = batch['batch_ptr'].to(device)
            
            scores, pose_scores, chemistry_scores = model(
                x, coords, accessibility, valid_mask,
                edge_index, edge_attr, batch_ptr
            )
            
            loss, _ = loss_fn(
                scores, pose_scores, chemistry_scores,
                som_labels, valid_mask
            )
            
            top1, top3, top5 = compute_accuracy(scores, som_labels, valid_mask)
            
            total_loss += loss.item()
            total_top1 += top1
            total_top3 += top3
            total_top5 += top5
            n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'top1': total_top1 / n_batches,
        'top3': total_top3 / n_batches,
        'top5': total_top5 / n_batches,
    }


def train(config: Config):
    """Main training loop."""
    print("="*80)
    print("CYP3A4 SoM Predictor Training")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Mixed precision: {config.mixed_precision}")
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Load enzyme pocket
    print("\nLoading enzyme pocket...")
    pocket = EnzymePocket(config.enzyme_pdb)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = CYP3A4Dataset(
        config.data_path, pocket,
        sources=['Zaretzki', 'DrugBank', 'MetXBioDB']
    )
    val_dataset = CYP3A4Dataset(
        config.data_path, pocket,
        sources=['AZ120']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    print(f"Train: {len(train_dataset)} molecules")
    print(f"Val: {len(val_dataset)} molecules")
    
    # Create model
    print("\nCreating model...")
    model = CYP3A4SoMPredictor(config, pocket)
    model = model.to(config.device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Loss function
    loss_fn = SoMLoss(config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - config.warmup_epochs) / 
                                  (config.epochs - config.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision)
    
    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    
    best_val_top1 = 0.0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("-"*40)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            config, epoch, scaler
        )
        
        print(f"Train: Loss={train_metrics['loss']:.4f}, "
              f"Top1={train_metrics['top1']*100:.1f}%, "
              f"Top3={train_metrics['top3']*100:.1f}%")
        
        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, config)
        
        print(f"Val:   Loss={val_metrics['loss']:.4f}, "
              f"Top1={val_metrics['top1']*100:.1f}%, "
              f"Top3={val_metrics['top3']*100:.1f}%, "
              f"Top5={val_metrics['top5']*100:.1f}%")
        
        # Save checkpoint
        if val_metrics['top1'] > best_val_top1:
            best_val_top1 = val_metrics['top1']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config.__dict__,
            }
            torch.save(checkpoint, os.path.join(config.checkpoint_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (Top1={best_val_top1*100:.1f}%)")
        
        if (epoch + 1) % config.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config.__dict__,
            }
            torch.save(checkpoint, os.path.join(
                config.checkpoint_dir, f'checkpoint_epoch{epoch+1}.pt'
            ))
    
    print("\n" + "="*80)
    print("Training complete!")
    print(f"Best validation Top-1 accuracy: {best_val_top1*100:.1f}%")
    print("="*80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train CYP3A4 SoM Predictor')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                        default='data/curated/merged_cyp3a4_extended.json')
    parser.add_argument('--enzyme_pdb', type=str, default='1W0E.pdb')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_lnn_layers', type=int, default=3)
    parser.add_argument('--n_gnn_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_mixed_precision', action='store_true')
    
    # Logging arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_interval', type=int, default=10)
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        data_path=args.data_path,
        enzyme_pdb=args.enzyme_pdb,
        hidden_dim=args.hidden_dim,
        n_lnn_layers=args.n_lnn_layers,
        n_gnn_layers=args.n_gnn_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device if torch.cuda.is_available() else 'cpu',
        num_workers=args.num_workers,
        mixed_precision=not args.no_mixed_precision,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
    )
    
    # Train
    train(config)


if __name__ == '__main__':
    main()
