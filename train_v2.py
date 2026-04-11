#!/usr/bin/env python3
"""
CYP3A4 SoM Predictor v2 - With Full Reverse Geometry Learning

This version properly learns enzyme pocket dynamics from SoM data.
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

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Import our reverse geometry engine
from reverse_geometry_engine import ReverseGeometryEngine, ReverseGeometryLoss


@dataclass
class Config:
    """Training configuration."""
    data_path: str = "data/curated/merged_cyp3a4_extended.json"
    enzyme_pdb: str = "1W0E.pdb"
    
    # Model
    hidden_dim: int = 128
    state_dim: int = 64
    n_states: int = 8  # Number of pocket states to learn
    n_gnn_layers: int = 4
    dropout: float = 0.1
    
    # Training
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-3  # Higher LR
    weight_decay: float = 1e-4  # Lower weight decay
    warmup_epochs: int = 5
    
    # Loss weights
    ranking_weight: float = 1.0
    state_weight: float = 0.5
    diversity_weight: float = 0.1
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Logging
    log_interval: int = 10
    checkpoint_dir: str = "checkpoints_v2"


# ═══════════════════════════════════════════════════════════════════════════════
# ENZYME POCKET
# ═══════════════════════════════════════════════════════════════════════════════

class EnzymePocket:
    """CYP3A4 enzyme pocket geometry."""
    
    def __init__(self, pdb_path: str = "1W0E.pdb"):
        self.fe_pos = np.array([54.95, 77.69, 10.64])
        self.pocket_atoms = []
        
        self._load_pocket(pdb_path)
        self._compute_features()
    
    def _load_pocket(self, pdb_path: str):
        if not os.path.exists(pdb_path):
            self._create_default_pocket()
            return
            
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    res = line[17:20].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    
                    pos = np.array([x, y, z])
                    dist = np.linalg.norm(pos - self.fe_pos)
                    
                    if 3.0 < dist < 15.0:
                        is_hydro = res in ['ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO']
                        is_polar = res in ['SER', 'THR', 'ASN', 'GLN', 'TYR']
                        is_charged = res in ['ARG', 'LYS', 'ASP', 'GLU', 'HIS']
                        
                        self.pocket_atoms.append({
                            'pos': pos,
                            'dist': dist,
                            'hydrophobic': is_hydro,
                            'polar': is_polar,
                            'charged': is_charged,
                        })
        
        print(f"Loaded {len(self.pocket_atoms)} pocket atoms")
    
    def _create_default_pocket(self):
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
    
    def _compute_features(self):
        if not self.pocket_atoms:
            return
            
        positions = np.array([a['pos'] for a in self.pocket_atoms])
        self.pocket_centered = positions - self.fe_pos
        self.centroid = self.pocket_centered.mean(axis=0)
        self.feo_axis = self.centroid / (np.linalg.norm(self.centroid) + 1e-8)
    
    def get_pocket_tensor(self, device: str = "cpu") -> torch.Tensor:
        features = []
        for atom in self.pocket_atoms:
            pos_centered = atom['pos'] - self.fe_pos
            pos_normalized = pos_centered / (np.linalg.norm(pos_centered) + 1e-8)
            
            feat = np.concatenate([
                pos_centered / 10.0,
                pos_normalized,
                [atom['dist'] / 15.0],
                [1.0 if atom['hydrophobic'] else 0.0],
                [1.0 if atom.get('polar', False) else 0.0],
                [1.0 if atom.get('charged', False) else 0.0],
                self.feo_axis,
                [np.dot(pos_normalized, self.feo_axis)],
            ])
            features.append(feat)
        
        return torch.tensor(np.array(features), dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════════════════════════
# MOLECULAR FEATURIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_atom_features(atom, mol) -> np.ndarray:
    z = atom.GetAtomicNum()
    degree = atom.GetDegree()
    n_H = atom.GetTotalNumHs()
    formal_charge = atom.GetFormalCharge()
    is_aromatic = atom.GetIsAromatic()
    in_ring = atom.IsInRing()
    
    hyb = atom.GetHybridization()
    hyb_onehot = [
        hyb == Chem.HybridizationType.SP,
        hyb == Chem.HybridizationType.SP2,
        hyb == Chem.HybridizationType.SP3,
        hyb == Chem.HybridizationType.SP3D,
        hyb == Chem.HybridizationType.SP3D2,
    ]
    
    elem_onehot = [z == 6, z == 7, z == 8, z == 16, z not in [6, 7, 8, 16]]
    
    neighbors = list(atom.GetNeighbors())
    has_O_neighbor = any(n.GetAtomicNum() == 8 for n in neighbors)
    has_N_neighbor = any(n.GetAtomicNum() == 7 for n in neighbors)
    has_arom_neighbor = any(n.GetIsAromatic() for n in neighbors) and not is_aromatic
    
    alpha_het_score = 2.45 if has_O_neighbor else (2.06 if has_N_neighbor else 1.0)
    benzylic_score = 1.25 if has_arom_neighbor else 1.0
    
    features = np.array([
        z / 20.0,
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


def compute_laplacian_features(mol, k: int = 8) -> np.ndarray:
    n = mol.GetNumAtoms()
    
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = 1.5 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble()
        A[i, j] = A[j, i] = w
    
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.ones(n)
        eigvecs = np.eye(n)
    
    features = np.zeros((n, k + 2), dtype=np.float32)
    k_actual = min(k, n - 1)
    
    for i in range(n):
        k_start = max(1, n - k_actual)
        features[i, 0] = sum(eigvecs[i, j]**2 for j in range(k_start, n))
        
        k_end = min(k_actual, n)
        features[i, 1] = sum(eigvecs[i, j]**2 / (eigvals[j] + 0.1) for j in range(1, k_end))
        
        for j in range(min(k_actual, n - 1)):
            features[i, j + 2] = eigvecs[i, j + 1]
    
    return features


def mol_to_data(mol, pocket: EnzymePocket, som_atoms: List[int] = None) -> Dict:
    n = mol.GetNumAtoms()
    
    # Atom features
    atom_features = np.array([get_atom_features(mol.GetAtomWithIdx(i), mol) for i in range(n)])
    lap_features = compute_laplacian_features(mol)
    x = np.concatenate([atom_features, lap_features], axis=1)
    
    # 3D coordinates
    mol_h = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol_h, randomSeed=42, maxAttempts=100)
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        conf = mol_h.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] for i in range(n)])
    except:
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           0.0] for i in range(n)])
    
    coords = coords - coords.mean(axis=0)
    
    # Valid mask
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
        for idx in som_atoms:
            if idx < n:
                som_labels[idx] = 1.0
    
    return {
        'x': torch.tensor(x, dtype=torch.float32),
        'coords': torch.tensor(coords, dtype=torch.float32),
        'valid_mask': torch.tensor(valid_mask, dtype=torch.bool),
        'som_labels': torch.tensor(som_labels, dtype=torch.float32),
        'n_atoms': n,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class CYP3A4Dataset(Dataset):
    def __init__(self, data_path: str, pocket: EnzymePocket, sources: List[str] = None,
                 split: str = None, val_fraction: float = 0.2, seed: int = 42):
        self.pocket = pocket
        self.data = []
        
        with open(data_path) as f:
            raw_data = json.load(f)['drugs']
        
        if sources:
            raw_data = [d for d in raw_data if d.get('source') in sources]
        
        all_data = []
        for d in raw_data:
            smiles = d.get('smiles', '')
            sites = d.get('site_atoms', [])
            
            if not smiles or not sites:
                continue
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                data = mol_to_data(mol, pocket, sites)
                data['smiles'] = smiles
                all_data.append(data)
            except:
                continue
        
        if split is not None:
            np.random.seed(seed)
            indices = np.random.permutation(len(all_data))
            n_val = int(len(all_data) * val_fraction)
            
            if split == 'val':
                indices = indices[:n_val]
            elif split == 'train':
                indices = indices[n_val:]
            
            self.data = [all_data[i] for i in indices]
        else:
            self.data = all_data
        
        print(f"Loaded {len(self.data)} molecules ({split if split else 'all'})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_atoms = max(d['n_atoms'] for d in batch)
    
    batch_x = []
    batch_coords = []
    batch_valid_mask = []
    batch_som_labels = []
    
    for d in batch:
        n = d['n_atoms']
        
        x_padded = F.pad(d['x'], (0, 0, 0, max_atoms - n))
        coords_padded = F.pad(d['coords'], (0, 0, 0, max_atoms - n))
        valid_padded = F.pad(d['valid_mask'], (0, max_atoms - n))
        som_padded = F.pad(d['som_labels'], (0, max_atoms - n))
        
        batch_x.append(x_padded)
        batch_coords.append(coords_padded)
        batch_valid_mask.append(valid_padded)
        batch_som_labels.append(som_padded)
    
    return {
        'x': torch.stack(batch_x),
        'coords': torch.stack(batch_coords),
        'valid_mask': torch.stack(batch_valid_mask),
        'som_labels': torch.stack(batch_som_labels),
        'n_atoms': torch.tensor([d['n_atoms'] for d in batch]),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple message passing within batch
        B, N, D = x.shape
        
        # Self + mean of neighbors (simplified)
        x_mean = x.mean(dim=1, keepdim=True).expand(-1, N, -1)
        out = self.lin(x + 0.5 * x_mean)
        return self.norm(F.silu(out))


class CYP3A4Model(nn.Module):
    """
    Full model with reverse geometry learning.
    """
    
    def __init__(self, config: Config, pocket: EnzymePocket):
        super().__init__()
        self.config = config
        
        # Input dimension: atom_features (21) + laplacian (10)
        input_dim = 31
        
        # Initial embedding
        self.embed = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(config.hidden_dim, config.hidden_dim)
            for _ in range(config.n_gnn_layers)
        ])
        
        # Reverse geometry engine
        pocket_feat = pocket.get_pocket_tensor()
        self.register_buffer('pocket_feat', pocket_feat)
        
        self.reverse_geometry = ReverseGeometryEngine(
            mol_dim=config.hidden_dim,
            state_dim=config.state_dim,
            n_states=config.n_states,
            pocket_dim=pocket_feat.shape[-1],
        )
        
        # Chemistry head
        self.chemistry_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        valid_mask: torch.Tensor,
        som_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N, _ = x.shape
        
        # Embed
        h = self.embed(x)
        
        # GNN
        for gnn in self.gnn_layers:
            h = gnn(h)
        
        # Chemistry scores
        chem_scores = self.chemistry_head(h).squeeze(-1)
        
        # Reverse geometry
        output = self.reverse_geometry(
            h, coords, self.pocket_feat,
            som_mask=som_labels,
            valid_mask=valid_mask,
        )
        
        # Combine scores
        scores = output['scores'] * chem_scores
        scores = scores.masked_fill(~valid_mask, float('-inf'))
        
        output['final_scores'] = scores
        output['chem_scores'] = chem_scores
        
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_accuracy(scores, som_labels, valid_mask):
    B = scores.shape[0]
    top1_correct = 0
    top3_correct = 0
    total = 0
    
    for b in range(B):
        mask = valid_mask[b]
        labels = som_labels[b]
        s = scores[b]
        
        if mask.sum() == 0 or labels.sum() == 0:
            continue
        
        valid_indices = torch.where(mask)[0]
        valid_scores = s[mask]
        sorted_indices = valid_indices[torch.argsort(valid_scores, descending=True)]
        
        som_indices = set(torch.where(labels > 0)[0].tolist())
        
        if sorted_indices[0].item() in som_indices:
            top1_correct += 1
        
        if len(sorted_indices) >= 3:
            if any(idx.item() in som_indices for idx in sorted_indices[:3]):
                top3_correct += 1
        
        total += 1
    
    return top1_correct / max(total, 1), top3_correct / max(total, 1)


def train_epoch(model, loader, loss_fn, optimizer, config, scaler):
    model.train()
    device = config.device
    
    total_loss = 0
    total_top1 = 0
    n_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        x = batch['x'].to(device)
        coords = batch['coords'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        som_labels = batch['som_labels'].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=config.mixed_precision):
            output = model(x, coords, valid_mask, som_labels)
            loss, _ = loss_fn(output, som_labels, valid_mask)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            top1, _ = compute_accuracy(output['final_scores'], som_labels, valid_mask)
        
        total_loss += loss.item()
        total_top1 += top1
        n_batches += 1
        
        if batch_idx % config.log_interval == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: Loss={loss.item():.4f}, Top1={top1*100:.1f}%")
    
    return total_loss / n_batches, total_top1 / n_batches


def evaluate(model, loader, loss_fn, config):
    model.eval()
    device = config.device
    
    total_loss = 0
    total_top1 = 0
    total_top3 = 0
    n_batches = 0
    
    # Track state usage
    state_usage = torch.zeros(config.n_states)
    
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            coords = batch['coords'].to(device)
            valid_mask = batch['valid_mask'].to(device)
            som_labels = batch['som_labels'].to(device)
            
            output = model(x, coords, valid_mask, som_labels)
            loss, _ = loss_fn(output, som_labels, valid_mask)
            
            top1, top3 = compute_accuracy(output['final_scores'], som_labels, valid_mask)
            
            # Track which states are being used
            state_probs = output['state_probs']  # (B, n_states)
            state_usage += state_probs.sum(dim=0).cpu()
            
            total_loss += loss.item()
            total_top1 += top1
            total_top3 += top3
            n_batches += 1
    
    # Normalize state usage
    state_usage = state_usage / state_usage.sum()
    
    return {
        'loss': total_loss / n_batches,
        'top1': total_top1 / n_batches,
        'top3': total_top3 / n_batches,
        'state_usage': state_usage.numpy(),
    }


def train(config: Config):
    print("="*70)
    print("CYP3A4 SoM Predictor v2 - Reverse Geometry Learning")
    print("="*70)
    print(f"Device: {config.device}")
    print(f"Pocket states: {config.n_states}")
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Load pocket
    pocket = EnzymePocket(config.enzyme_pdb)
    
    # Load data
    train_sources = getattr(config, 'train_sources', 'Zaretzki').split(',')
    val_sources = getattr(config, 'val_sources', 'Zaretzki').split(',')
    
    print(f"\nTrain sources: {train_sources}")
    print(f"Val sources: {val_sources}")
    
    if set(train_sources) == set(val_sources):
        print("Same sources - using 80/20 split")
        train_dataset = CYP3A4Dataset(config.data_path, pocket, train_sources, 'train')
        val_dataset = CYP3A4Dataset(config.data_path, pocket, val_sources, 'val')
    else:
        train_dataset = CYP3A4Dataset(config.data_path, pocket, train_sources)
        val_dataset = CYP3A4Dataset(config.data_path, pocket, val_sources)
    
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, config.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=config.num_workers)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = CYP3A4Model(config, pocket).to(config.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # Loss
    loss_fn = ReverseGeometryLoss(
        ranking_weight=config.ranking_weight,
        state_weight=config.state_weight,
        diversity_weight=config.diversity_weight,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    # Scheduler
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return epoch / config.warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - config.warmup_epochs) / (config.epochs - config.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision)
    
    # Training
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    best_val_top1 = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("-"*40)
        
        train_loss, train_top1 = train_epoch(model, train_loader, loss_fn, optimizer, config, scaler)
        scheduler.step()
        
        val_metrics = evaluate(model, val_loader, loss_fn, config)
        
        print(f"Train: Loss={train_loss:.4f}, Top1={train_top1*100:.1f}%")
        print(f"Val:   Loss={val_metrics['loss']:.4f}, Top1={val_metrics['top1']*100:.1f}%, Top3={val_metrics['top3']*100:.1f}%")
        print(f"State usage: {val_metrics['state_usage']}")
        
        if val_metrics['top1'] > best_val_top1:
            best_val_top1 = val_metrics['top1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_metrics': val_metrics,
            }, os.path.join(config.checkpoint_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (Top1={best_val_top1*100:.1f}%)")
    
    print("\n" + "="*70)
    print(f"Training complete! Best Val Top-1: {best_val_top1*100:.1f}%")
    print("="*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_states', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--train_sources', type=str, default='Zaretzki')
    parser.add_argument('--val_sources', type=str, default='Zaretzki')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v2')
    
    args = parser.parse_args()
    
    config = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        n_states=args.n_states,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
    )
    config.train_sources = args.train_sources
    config.val_sources = args.val_sources
    
    train(config)


if __name__ == '__main__':
    main()
