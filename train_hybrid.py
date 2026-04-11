#!/usr/bin/env python3
"""
Training script for HybridNexusDynamic model.

Combines:
1. Dynamic enzyme state discovery
2. Liquid neural network dynamics
3. Hyperbolic memory with analogical fusion

Usage:
    python train_hybrid.py --epochs 50 --batch_size 32
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hybrid_nexus_dynamic import HybridNexusDynamic, HybridLoss


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class CYP3A4Dataset(Dataset):
    """Dataset for CYP3A4 SoM prediction."""
    
    def __init__(
        self,
        data_path: str,
        sources: Optional[List[str]] = None,
        max_atoms: int = 200,
    ):
        self.max_atoms = max_atoms
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.molecules = []
        
        for entry in data.get('drugs', []):
            # Filter by source
            if sources and entry.get('source') not in sources:
                continue
            
            smiles = entry.get('smiles', '')
            som_atoms = entry.get('site_atoms', [])
            
            if not smiles or not som_atoms:
                continue
            
            self.molecules.append({
                'smiles': smiles,
                'som_atoms': som_atoms,
                'source': entry.get('source', 'unknown'),
            })
    
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx):
        mol_data = self.molecules[idx]
        
        # Generate features from SMILES
        features, coords, n_atoms = self._smiles_to_features(mol_data['smiles'])
        
        # Create SoM mask
        som_mask = torch.zeros(self.max_atoms)
        for atom_idx in mol_data['som_atoms']:
            if atom_idx < n_atoms:
                som_mask[atom_idx] = 1.0
        
        # Valid mask
        valid_mask = torch.zeros(self.max_atoms, dtype=torch.bool)
        valid_mask[:n_atoms] = True
        
        return {
            'features': features,
            'coords': coords,
            'som_mask': som_mask,
            'valid_mask': valid_mask,
            'n_atoms': n_atoms,
        }
    
    def _smiles_to_features(self, smiles: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Convert SMILES to feature tensors."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            mol = Chem.AddHs(mol)
            
            # Generate 3D coordinates
            AllChem.EmbedMolecule(mol, randomSeed=42)
            if mol.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
            
            n_atoms = mol.GetNumAtoms()
            n_atoms = min(n_atoms, self.max_atoms)
            
            # Extract features
            features = torch.zeros(self.max_atoms, 128)
            coords = torch.zeros(self.max_atoms, 3)
            
            conf = mol.GetConformer() if mol.GetNumConformers() > 0 else None
            
            for i in range(n_atoms):
                atom = mol.GetAtomWithIdx(i)
                
                # Basic features
                features[i, 0] = atom.GetAtomicNum() / 20.0
                features[i, 1] = atom.GetDegree() / 4.0
                features[i, 2] = atom.GetTotalNumHs() / 4.0
                features[i, 3] = atom.GetFormalCharge()
                features[i, 4] = 1.0 if atom.GetIsAromatic() else 0.0
                features[i, 5] = atom.GetMass() / 32.0
                
                # Hybridization one-hot
                hyb = atom.GetHybridization()
                if hyb == Chem.HybridizationType.SP:
                    features[i, 6] = 1.0
                elif hyb == Chem.HybridizationType.SP2:
                    features[i, 7] = 1.0
                elif hyb == Chem.HybridizationType.SP3:
                    features[i, 8] = 1.0
                
                # Atom type one-hot
                z = atom.GetAtomicNum()
                if z == 6:  # C
                    features[i, 10] = 1.0
                elif z == 7:  # N
                    features[i, 11] = 1.0
                elif z == 8:  # O
                    features[i, 12] = 1.0
                elif z == 16:  # S
                    features[i, 13] = 1.0
                elif z == 9:  # F
                    features[i, 14] = 1.0
                elif z == 17:  # Cl
                    features[i, 15] = 1.0
                elif z == 35:  # Br
                    features[i, 16] = 1.0
                elif z == 1:  # H
                    features[i, 17] = 1.0
                
                # Ring info
                features[i, 20] = 1.0 if atom.IsInRing() else 0.0
                features[i, 21] = 1.0 if atom.IsInRingSize(5) else 0.0
                features[i, 22] = 1.0 if atom.IsInRingSize(6) else 0.0
                
                # Coordinates
                if conf is not None:
                    pos = conf.GetAtomPosition(i)
                    coords[i, 0] = pos.x
                    coords[i, 1] = pos.y
                    coords[i, 2] = pos.z
            
            return features, coords, n_atoms
            
        except Exception as e:
            # Fallback: random features
            n_atoms = 20
            features = torch.randn(self.max_atoms, 128) * 0.1
            coords = torch.randn(self.max_atoms, 3) * 5
            return features, coords, n_atoms


def collate_fn(batch):
    """Collate function for DataLoader."""
    features = torch.stack([b['features'] for b in batch])
    coords = torch.stack([b['coords'] for b in batch])
    som_mask = torch.stack([b['som_mask'] for b in batch])
    valid_mask = torch.stack([b['valid_mask'] for b in batch])
    n_atoms = [b['n_atoms'] for b in batch]
    
    return {
        'features': features,
        'coords': coords,
        'som_mask': som_mask,
        'valid_mask': valid_mask,
        'n_atoms': n_atoms,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# POCKET LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_pocket_features(pdb_path: str, max_atoms: int = 100) -> torch.Tensor:
    """Load pocket features from PDB."""
    features = []
    
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    # Parse atom info
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    atom_name = line[12:16].strip()
                    residue = line[17:20].strip()
                    
                    # Compute features
                    feat = [
                        x / 100.0, y / 100.0, z / 100.0,  # Coords
                        1.0 if atom_name.startswith('C') else 0.0,  # Carbon
                        1.0 if atom_name.startswith('N') else 0.0,  # Nitrogen
                        1.0 if atom_name.startswith('O') else 0.0,  # Oxygen
                        1.0 if atom_name.startswith('S') else 0.0,  # Sulfur
                        1.0 if residue in ['PHE', 'TRP', 'TYR'] else 0.0,  # Aromatic
                        1.0 if residue in ['ALA', 'VAL', 'LEU', 'ILE', 'MET'] else 0.0,  # Hydrophobic
                        1.0 if residue in ['ASP', 'GLU'] else 0.0,  # Acidic
                        1.0 if residue in ['LYS', 'ARG', 'HIS'] else 0.0,  # Basic
                        1.0 if residue in ['SER', 'THR', 'ASN', 'GLN'] else 0.0,  # Polar
                        1.0 if 'FE' in atom_name else 0.0,  # Iron
                        1.0,  # Bias
                    ]
                    features.append(feat)
                    
                    if len(features) >= max_atoms:
                        break
    except Exception as e:
        print(f"Warning: Could not load pocket from {pdb_path}: {e}")
        features = [[0.0] * 14 for _ in range(max_atoms)]
    
    # Pad or truncate
    while len(features) < max_atoms:
        features.append([0.0] * 14)
    
    features = features[:max_atoms]
    
    return torch.tensor(features, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    scores: torch.Tensor,
    som_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute Top-1 and Top-3 accuracy."""
    B = scores.shape[0]
    
    top1_correct = 0
    top3_correct = 0
    
    for b in range(B):
        valid_scores = scores[b][valid_mask[b]]
        valid_som = som_mask[b][valid_mask[b]]
        
        if valid_som.sum() == 0:
            continue
        
        # Get predictions
        _, top_k = valid_scores.topk(min(3, len(valid_scores)))
        
        # Check if any true SoM is in top-k
        true_som_indices = (valid_som > 0).nonzero(as_tuple=True)[0]
        
        if len(true_som_indices) > 0:
            if top_k[0] in true_som_indices:
                top1_correct += 1
            if any(idx in true_som_indices for idx in top_k):
                top3_correct += 1
    
    return {
        'top1': top1_correct / B * 100,
        'top3': top3_correct / B * 100,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    pocket_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()
    
    total_loss = 0
    total_top1 = 0
    total_top3 = 0
    n_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        features = batch['features'].to(device)
        coords = batch['coords'].to(device)
        som_mask = batch['som_mask'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            features,
            coords,
            pocket_features.to(device),
            som_mask,
            valid_mask,
        )
        
        loss, metrics = loss_fn(outputs, som_mask, valid_mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            acc_metrics = compute_metrics(
                outputs['final_scores'],
                som_mask,
                valid_mask,
            )
        
        total_loss += loss.item()
        total_top1 += acc_metrics['top1']
        total_top3 += acc_metrics['top3']
        n_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: "
                  f"Loss={loss.item():.4f}, Top1={acc_metrics['top1']:.1f}%")
    
    return {
        'loss': total_loss / n_batches,
        'top1': total_top1 / n_batches,
        'top3': total_top3 / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    pocket_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0
    total_top1 = 0
    total_top3 = 0
    n_batches = 0
    
    for batch in loader:
        features = batch['features'].to(device)
        coords = batch['coords'].to(device)
        som_mask = batch['som_mask'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        outputs = model(
            features,
            coords,
            pocket_features.to(device),
            som_mask=None,  # Don't update memory during validation
            valid_mask=valid_mask,
        )
        
        loss, _ = loss_fn(outputs, som_mask, valid_mask)
        
        acc_metrics = compute_metrics(
            outputs['final_scores'],
            som_mask,
            valid_mask,
        )
        
        total_loss += loss.item()
        total_top1 += acc_metrics['top1']
        total_top3 += acc_metrics['top3']
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'top1': total_top1 / n_batches,
        'top3': total_top3 / n_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--max_states', type=int, default=32)
    parser.add_argument('--similarity_threshold', type=float, default=0.7)
    parser.add_argument('--n_liquid_steps', type=int, default=5)
    parser.add_argument('--memory_k', type=int, default=3)
    parser.add_argument('--train_sources', type=str, default='Zaretzki,DrugBank,MetXBioDB,AZ120')
    parser.add_argument('--val_sources', type=str, default='Zaretzki,DrugBank,MetXBioDB,AZ120')
    parser.add_argument('--data_path', type=str, default='data/curated/merged_cyp3a4_extended.json')
    parser.add_argument('--pdb_path', type=str, default='1W0E.pdb')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("CYP3A4 SoM Predictor - Hybrid NEXUS + Dynamic States")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Max states: {args.max_states}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    print(f"Liquid steps: {args.n_liquid_steps}")
    print(f"Memory k: {args.memory_k}")
    
    # Load pocket
    pocket_features = load_pocket_features(args.pdb_path)
    print(f"Loaded {pocket_features.shape[0]} pocket atoms")
    
    # Parse sources
    train_sources = args.train_sources.split(',')
    val_sources = args.val_sources.split(',')
    
    print(f"\nTrain sources: {train_sources}")
    print(f"Val sources: {val_sources}")
    
    # Load dataset
    full_dataset = CYP3A4Dataset(args.data_path, sources=train_sources)
    
    # Split if same sources
    if set(train_sources) == set(val_sources):
        print("Same sources - using 80/20 split")
        n_train = int(0.8 * len(full_dataset))
        n_val = len(full_dataset) - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )
    else:
        train_dataset = full_dataset
        val_dataset = CYP3A4Dataset(args.data_path, sources=val_sources)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Model
    model = HybridNexusDynamic(
        mol_dim=128,
        hidden_dim=args.hidden_dim,
        max_states=args.max_states,
        similarity_threshold=args.similarity_threshold,
        n_liquid_steps=args.n_liquid_steps,
        memory_k=args.memory_k,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Loss and optimizer
    loss_fn = HybridLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    
    # Training loop
    best_val_top1 = 0
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn,
            pocket_features, device,
        )
        
        val_metrics = validate(
            model, val_loader, loss_fn,
            pocket_features, device,
        )
        
        scheduler.step()
        
        # Report
        print(f"Train: Loss={train_metrics['loss']:.4f}, "
              f"Top1={train_metrics['top1']:.1f}%")
        print(f"Val:   Loss={val_metrics['loss']:.4f}, "
              f"Top1={val_metrics['top1']:.1f}%, Top3={val_metrics['top3']:.1f}%")
        print(f"Active states: {model.state_bank.num_active_states}, "
              f"Memory: {model.memory.n_entries}")
        
        # Save best
        if val_metrics['top1'] > best_val_top1:
            best_val_top1 = val_metrics['top1']
            torch.save(model.state_dict(), 'best_hybrid_model.pt')
            print(f"  ✓ Saved best model (Top1={best_val_top1:.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best Val Top-1: {best_val_top1:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
