#!/usr/bin/env python3
"""
ML + Physics Ensemble Training on New 869-molecule Dataset
===========================================================

This script:
1. Trains a GNN model on the new merged_cyp3a4_extended.json (869 molecules)
2. Combines with physics scoring (hydrogen_theft_v3)
3. Learns optimal ensemble weights

Usage:
    python train_ml_physics_v2.py \
        --data data/curated/merged_cyp3a4_extended.json \
        --output_dir checkpoints/ml_physics_v2 \
        --epochs 50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
except ImportError:
    print("PyTorch required: pip install torch")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
except ImportError:
    print("RDKit required: pip install rdkit")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    epochs: int = 50
    patience: int = 10
    physics_weight_init: float = 0.3  # Initial weight for physics


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dataset(path: str) -> List[Dict]:
    """Load dataset from JSON file."""
    with open(path) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if 'drugs' in data:
            return data['drugs']
        elif 'molecules' in data:
            return data['molecules']
    return []


def get_smiles_and_sites(entry: Dict) -> Tuple[str, List[int]]:
    """Extract SMILES and SoM sites from entry."""
    smiles = entry.get('smiles', entry.get('SMILES', ''))
    sites = []
    
    if 'site_atoms' in entry:
        sites = entry['site_atoms']
    elif 'som_site' in entry:
        sites = entry['som_site'] if isinstance(entry['som_site'], list) else [entry['som_site']]
    elif 'atom_indices' in entry:
        sites = entry['atom_indices']
    elif 'positive_atoms' in entry:
        sites = entry['positive_atoms']
    
    if not isinstance(sites, list):
        sites = [sites] if sites is not None else []
    
    return smiles, sites


# =============================================================================
# PHYSICS SCORER (from hydrogen_theft_v3)
# =============================================================================

# Optimized BDE table from v3
BDE_TABLE = {
    'ALPHA_O': 80, 'S_OXIDE': 80, 'ALPHA_N': 84, 'ALPHA_S': 85,
    'ALLYLIC': 86, 'PRIMARY': 96, 'AROMATIC': 97, 'SECONDARY': 98,
    'BENZYLIC': 99, 'AROMATIC_NO_H': 107, 'TERTIARY': 110, 'N_OXIDE': 111
}

def classify_carbon(mol, idx: int) -> str:
    """Classify carbon by type."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() != 6:
        return 'NON_CARBON'
    
    n_H = atom.GetTotalNumHs()
    if n_H == 0:
        return 'AROMATIC_NO_H' if atom.GetIsAromatic() else 'TERTIARY'
    
    neighbors = [n for n in atom.GetNeighbors()]
    n_heavy = len(neighbors)
    
    # Check for alpha positions
    for n in neighbors:
        sym = n.GetSymbol()
        if sym == 'O':
            return 'ALPHA_O'
        if sym == 'N':
            return 'ALPHA_N'
        if sym == 'S':
            return 'ALPHA_S'
    
    # Check for benzylic/allylic
    for n in neighbors:
        if n.GetIsAromatic():
            return 'BENZYLIC'
        for bond in n.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0:
                other = bond.GetOtherAtom(n)
                if other.GetIdx() != idx and other.GetAtomicNum() == 6:
                    return 'ALLYLIC'
    
    if atom.GetIsAromatic():
        return 'AROMATIC'
    
    # By degree
    if n_heavy == 1:
        return 'PRIMARY'
    elif n_heavy == 2:
        return 'SECONDARY'
    else:
        return 'TERTIARY'


def compute_physics_scores(smiles: str) -> Optional[torch.Tensor]:
    """Compute physics-based scores for each atom."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    n_atoms = mol.GetNumAtoms()
    scores = torch.zeros(n_atoms)
    
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        
        # Only score carbons with H
        if atom.GetAtomicNum() != 6 or atom.GetTotalNumHs() == 0:
            scores[i] = -10.0  # Very low score
            continue
        
        # Get BDE-based score
        c_type = classify_carbon(mol, i)
        bde = BDE_TABLE.get(c_type, 100)
        
        # Lower BDE = more reactive = higher score
        # Normalize to ~0-1 range
        scores[i] = (110 - bde) / 30.0  # Range roughly 0-1
    
    return scores


# =============================================================================
# GNN MODEL
# =============================================================================

class AtomEncoder(nn.Module):
    """Encode atom features."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(100, hidden_dim // 2)  # Atomic number
        self.feature_proj = nn.Linear(hidden_dim // 2 + 16, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, atomic_nums: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(atomic_nums)
        x = torch.cat([emb, features], dim=-1)
        return self.norm(F.silu(self.feature_proj(x)))


class MessagePassingLayer(nn.Module):
    """Simple message passing layer."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h
        
        src, dst = edge_index[0], edge_index[1]
        
        # Compute messages
        msg_input = torch.cat([h[src], h[dst]], dim=-1)
        messages = self.msg_net(msg_input)
        
        # Aggregate
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        degree = torch.bincount(dst, minlength=h.size(0)).clamp(min=1).float()
        agg = agg / degree.unsqueeze(-1)
        
        # Update
        update = self.update_net(torch.cat([h, agg], dim=-1))
        return self.norm(h + update)


class SoMGNN(nn.Module):
    """GNN for SoM prediction."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.encoder = AtomEncoder(config.hidden_dim)
        
        self.layers = nn.ModuleList([
            MessagePassingLayer(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        
        # Learnable physics weight
        self.physics_weight = nn.Parameter(
            torch.tensor(config.physics_weight_init)
        )
    
    def encode_molecule(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode molecule to tensors."""
        # Atomic numbers
        atomic_nums = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
        
        # Extra features
        features = []
        for atom in mol.GetAtoms():
            f = [
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
            f.append(en / 4.0)
            
            # Pad to 16
            f = f + [0.0] * (16 - len(f))
            features.append(f)
        
        features = torch.tensor(features, dtype=torch.float32)
        
        # Edges
        src, dst = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([i, j])
            dst.extend([j, i])
        
        if src:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        
        return atomic_nums, features, edge_index
    
    def forward(self, smiles: str, physics_scores: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            ml_scores: Raw ML scores
            combined_scores: ML + Physics ensemble
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        atomic_nums, features, edge_index = self.encode_molecule(mol)
        
        # Encode
        h = self.encoder(atomic_nums, features)
        
        # Message passing
        for layer in self.layers:
            h = layer(h, edge_index)
        
        # Predict
        ml_scores = self.head(h).squeeze(-1)
        
        # Combine with physics
        if physics_scores is None:
            physics_scores = compute_physics_scores(smiles)
        
        if physics_scores is not None:
            # Learned weighted combination
            w = torch.sigmoid(self.physics_weight)  # Keep in [0, 1]
            combined = (1 - w) * ml_scores + w * physics_scores
        else:
            combined = ml_scores
        
        return ml_scores, combined


# =============================================================================
# TRAINING
# =============================================================================

def ranking_loss(scores: torch.Tensor, labels: torch.Tensor, 
                 mask: torch.Tensor) -> torch.Tensor:
    """ListMLE ranking loss."""
    if not mask.any() or labels[mask].sum() == 0:
        return scores.new_zeros(1)
    
    valid_scores = scores[mask]
    valid_labels = labels[mask]
    
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


def evaluate(model: SoMGNN, data: List[Dict], device: torch.device
             ) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    top1, top3, total = 0, 0, 0
    
    with torch.no_grad():
        for entry in data:
            smiles, sites = get_smiles_and_sites(entry)
            if not smiles or not sites:
                continue
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                _, combined = model(smiles)
                
                # Mask non-candidates
                candidate_mask = torch.tensor([
                    a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0
                    for a in mol.GetAtoms()
                ], dtype=torch.bool)
                
                masked_scores = combined.clone()
                masked_scores[~candidate_mask] = float('-inf')
                
                top_k = min(3, candidate_mask.sum().item())
                if top_k == 0:
                    continue
                
                top_idx = torch.topk(masked_scores, top_k).indices.tolist()
                
                top1 += int(top_idx[0] in sites)
                top3 += int(any(i in sites for i in top_idx[:3]))
                total += 1
                
            except Exception:
                continue
    
    return {
        'top1': top1 / max(total, 1),
        'top3': top3 / max(total, 1),
        'n': total,
    }


def train_epoch(model: SoMGNN, data: List[Dict], optimizer: torch.optim.Optimizer,
                device: torch.device) -> Tuple[float, int]:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    random.shuffle(data)
    
    for entry in data:
        smiles, sites = get_smiles_and_sites(entry)
        if not smiles or not sites:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        n_atoms = mol.GetNumAtoms()
        labels = torch.zeros(n_atoms)
        for s in sites:
            if s < n_atoms:
                labels[s] = 1.0
        
        candidate_mask = torch.tensor([
            a.GetAtomicNum() == 6 and a.GetTotalNumHs() > 0
            for a in mol.GetAtoms()
        ], dtype=torch.bool)
        
        try:
            optimizer.zero_grad()
            
            ml_scores, combined = model(smiles)
            
            # Loss on combined scores
            loss = ranking_loss(combined, labels, candidate_mask)
            
            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_samples += 1
                
        except Exception:
            continue
    
    return total_loss / max(n_samples, 1), n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to dataset JSON')
    parser.add_argument('--output_dir', default='checkpoints/ml_physics_v2')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--physics_weight', type=float, default=0.3)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("ML + PHYSICS ENSEMBLE TRAINING (v2)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    
    # Load data
    print("\nLoading data...")
    data = load_dataset(args.data)
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create model
    print("\nCreating model...")
    config = Config(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        epochs=args.epochs,
        physics_weight_init=args.physics_weight,
    )
    model = SoMGNN(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Initial physics weight: {args.physics_weight}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Training loop
    print("\nTraining...")
    print("-" * 70)
    
    best_top1 = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, n = train_epoch(model, train_data, optimizer, device)
        scheduler.step()
        
        # Evaluate
        val_metrics = evaluate(model, val_data, device)
        
        # Get current physics weight
        w = torch.sigmoid(model.physics_weight).item()
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Top-1: {val_metrics['top1']*100:.1f}% | "
              f"Top-3: {val_metrics['top3']*100:.1f}% | "
              f"w_phys: {w:.2f}")
        
        # Save best
        if val_metrics['top1'] > best_top1:
            best_top1 = val_metrics['top1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
                'physics_weight': w,
            }, os.path.join(args.output_dir, 'ml_physics_best.pt'))
            print(f"  → Saved best (Top-1: {best_top1*100:.1f}%)")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    # Load best model
    ckpt = torch.load(os.path.join(args.output_dir, 'ml_physics_best.pt'))
    model.load_state_dict(ckpt['model_state_dict'])
    
    final_metrics = evaluate(model, val_data, device)
    print(f"Best Top-1: {best_top1*100:.1f}%")
    print(f"Final Top-1: {final_metrics['top1']*100:.1f}%")
    print(f"Final Top-3: {final_metrics['top3']*100:.1f}%")
    print(f"Learned physics weight: {torch.sigmoid(model.physics_weight).item():.3f}")
    
    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': final_metrics,
    }, os.path.join(args.output_dir, 'ml_physics_final.pt'))
    
    print(f"\nModels saved to {args.output_dir}")


if __name__ == '__main__':
    main()
