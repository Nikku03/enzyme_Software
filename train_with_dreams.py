#!/usr/bin/env python3
"""
Training script with Dream Phase for CYP3A4 SoM prediction.

The dream phase runs between epochs to:
1. Replay difficult examples with noise (robustness)
2. Imagine synthetic molecules (generalization)
3. Consolidate patterns into rules (abstraction)
4. Refine enzyme state representations (efficiency)

Usage:
    python train_with_dreams.py --epochs 50 --dream_every 3
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
from dream_phase import DreamPhase


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET (same as train_hybrid.py)
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
        features, coords, n_atoms = self._smiles_to_features(mol_data['smiles'])
        
        som_mask = torch.zeros(self.max_atoms)
        for atom_idx in mol_data['som_atoms']:
            if atom_idx < n_atoms:
                som_mask[atom_idx] = 1.0
        
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
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            mol = Chem.AddHs(mol)
            n_atoms = min(mol.GetNumAtoms(), self.max_atoms)
            
            # Generate 3D coordinates
            conf = None
            try:
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                if result == 0 and mol.GetNumConformers() > 0:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
                    conf = mol.GetConformer()
            except:
                pass
            
            if conf is None:
                try:
                    result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
                    if result == 0 and mol.GetNumConformers() > 0:
                        conf = mol.GetConformer()
                except:
                    pass
            
            features = torch.zeros(self.max_atoms, 128)
            coords = torch.zeros(self.max_atoms, 3)
            
            for i in range(n_atoms):
                atom = mol.GetAtomWithIdx(i)
                z = atom.GetAtomicNum()
                
                features[i, 0] = z / 20.0
                features[i, 1] = atom.GetDegree() / 4.0
                features[i, 2] = atom.GetTotalNumHs() / 4.0
                features[i, 3] = atom.GetFormalCharge() / 2.0
                features[i, 4] = 1.0 if atom.GetIsAromatic() else 0.0
                features[i, 5] = atom.GetMass() / 32.0
                
                hyb = atom.GetHybridization()
                if hyb == Chem.HybridizationType.SP:
                    features[i, 6] = 1.0
                elif hyb == Chem.HybridizationType.SP2:
                    features[i, 7] = 1.0
                elif hyb == Chem.HybridizationType.SP3:
                    features[i, 8] = 1.0
                
                if z == 6: features[i, 10] = 1.0
                elif z == 7: features[i, 11] = 1.0
                elif z == 8: features[i, 12] = 1.0
                elif z == 16: features[i, 13] = 1.0
                elif z == 9: features[i, 14] = 1.0
                elif z == 17: features[i, 15] = 1.0
                elif z == 35: features[i, 16] = 1.0
                elif z == 1: features[i, 17] = 1.0
                
                features[i, 20] = 1.0 if atom.IsInRing() else 0.0
                features[i, 21] = 1.0 if atom.IsInRingSize(5) else 0.0
                features[i, 22] = 1.0 if atom.IsInRingSize(6) else 0.0
                
                if conf is not None:
                    try:
                        pos = conf.GetAtomPosition(i)
                        coords[i, 0] = pos.x
                        coords[i, 1] = pos.y
                        coords[i, 2] = pos.z
                    except:
                        coords[i, 0] = (i * 1.5) % 10 - 5
                        coords[i, 1] = (z * 0.7) % 10 - 5
                        coords[i, 2] = (i * z * 0.3) % 10 - 5
                else:
                    coords[i, 0] = (i * 1.5) % 10 - 5
                    coords[i, 1] = (z * 0.7) % 10 - 5
                    coords[i, 2] = (i * z * 0.3) % 10 - 5
            
            return features, coords, n_atoms
            
        except Exception as e:
            n_atoms = 20
            features = torch.zeros(self.max_atoms, 128)
            features[:n_atoms, 0] = 0.3
            features[:n_atoms, 10] = 1.0
            coords = torch.zeros(self.max_atoms, 3)
            for i in range(n_atoms):
                coords[i, 0] = (i * 1.5) % 10 - 5
                coords[i, 1] = (i * 0.7) % 10 - 5
                coords[i, 2] = i * 0.3
            return features, coords, n_atoms


def collate_fn(batch):
    return {
        'features': torch.stack([b['features'] for b in batch]),
        'coords': torch.stack([b['coords'] for b in batch]),
        'som_mask': torch.stack([b['som_mask'] for b in batch]),
        'valid_mask': torch.stack([b['valid_mask'] for b in batch]),
        'n_atoms': [b['n_atoms'] for b in batch],
    }


def load_pocket_features(pdb_path: str, max_atoms: int = 100) -> torch.Tensor:
    """Load pocket features from PDB."""
    features = []
    
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atom_name = line[12:16].strip()
                    residue = line[17:20].strip()
                    
                    feat = [
                        x / 100.0, y / 100.0, z / 100.0,
                        1.0 if atom_name.startswith('C') else 0.0,
                        1.0 if atom_name.startswith('N') else 0.0,
                        1.0 if atom_name.startswith('O') else 0.0,
                        1.0 if atom_name.startswith('S') else 0.0,
                        1.0 if residue in ['PHE', 'TRP', 'TYR'] else 0.0,
                        1.0 if residue in ['ALA', 'VAL', 'LEU', 'ILE', 'MET'] else 0.0,
                        1.0 if residue in ['ASP', 'GLU'] else 0.0,
                        1.0 if residue in ['LYS', 'ARG', 'HIS'] else 0.0,
                        1.0 if residue in ['SER', 'THR', 'ASN', 'GLN'] else 0.0,
                        1.0 if 'FE' in atom_name else 0.0,
                        1.0,
                    ]
                    features.append(feat)
                    if len(features) >= max_atoms:
                        break
    except Exception as e:
        features = [[0.0] * 14 for _ in range(max_atoms)]
    
    while len(features) < max_atoms:
        features.append([0.0] * 14)
    
    return torch.tensor(features[:max_atoms], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING WITH DREAMS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(scores, som_mask, valid_mask):
    B = scores.shape[0]
    top1_correct = 0
    top3_correct = 0
    
    for b in range(B):
        valid_scores = scores[b][valid_mask[b]]
        valid_som = som_mask[b][valid_mask[b]]
        
        if valid_som.sum() == 0:
            continue
        
        _, top_k = valid_scores.topk(min(3, len(valid_scores)))
        true_som_indices = (valid_som > 0).nonzero(as_tuple=True)[0]
        
        if len(true_som_indices) > 0:
            if top_k[0] in true_som_indices:
                top1_correct += 1
            if any(idx in true_som_indices for idx in top_k):
                top3_correct += 1
    
    return {'top1': top1_correct / B * 100, 'top3': top3_correct / B * 100}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    dream_phase: DreamPhase,
    pocket_features: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Train one epoch with dream recording."""
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
            features, coords,
            pocket_features.to(device),
            som_mask, valid_mask,
        )
        
        loss, metrics = loss_fn(outputs, som_mask, valid_mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record experiences for dreaming
        with torch.no_grad():
            mol_encoded = model.mol_encoder(features)
            mol_global = mol_encoded.mean(dim=1)
            
            B = features.shape[0]
            for b in range(B):
                dream_phase.record_experience(
                    embedding=mol_global[b],
                    som_mask=som_mask[b],
                    prediction=outputs['final_scores'][b],
                    loss=metrics['main_loss'],
                )
        
        with torch.no_grad():
            acc_metrics = compute_metrics(outputs['final_scores'], som_mask, valid_mask)
        
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
def validate(model, loader, loss_fn, pocket_features, device):
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
        
        outputs = model(features, coords, pocket_features.to(device), None, valid_mask)
        loss, _ = loss_fn(outputs, som_mask, valid_mask)
        
        acc_metrics = compute_metrics(outputs['final_scores'], som_mask, valid_mask)
        
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
    parser.add_argument('--dream_every', type=int, default=3, help='Dream every N epochs')
    parser.add_argument('--n_replay_steps', type=int, default=20)
    parser.add_argument('--n_imagination_steps', type=int, default=10)
    parser.add_argument('--train_sources', type=str, default='Zaretzki,DrugBank,MetXBioDB,AZ120')
    parser.add_argument('--val_sources', type=str, default='Zaretzki,DrugBank,MetXBioDB,AZ120')
    parser.add_argument('--data_path', type=str, default='data/curated/merged_cyp3a4_extended.json')
    parser.add_argument('--pdb_path', type=str, default='1W0E.pdb')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("CYP3A4 SoM Predictor - Hybrid with Dream Phase")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Dream every: {args.dream_every} epochs")
    print(f"Replay steps: {args.n_replay_steps}")
    print(f"Imagination steps: {args.n_imagination_steps}")
    
    pocket_features = load_pocket_features(args.pdb_path)
    print(f"Loaded {pocket_features.shape[0]} pocket atoms")
    
    train_sources = args.train_sources.split(',')
    val_sources = args.val_sources.split(',')
    
    print(f"\nTrain sources: {train_sources}")
    print(f"Val sources: {val_sources}")
    
    full_dataset = CYP3A4Dataset(args.data_path, sources=train_sources)
    
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
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    
    # Model
    model = HybridNexusDynamic(
        mol_dim=128,
        hidden_dim=args.hidden_dim,
        max_states=args.max_states,
        similarity_threshold=args.similarity_threshold,
    ).to(device)
    
    # Dream phase
    dream_phase = DreamPhase(
        hidden_dim=args.hidden_dim,
        n_replay_steps=args.n_replay_steps,
        n_imagination_steps=args.n_imagination_steps,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_dream_params = sum(p.numel() for p in dream_phase.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Dream parameters: {n_dream_params:,}")
    
    loss_fn = HybridLoss()
    
    # Optimizer includes dream phase parameters
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(dream_phase.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_top1 = 0
    
    print("\n" + "=" * 70)
    print("Starting training with dream phases...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn,
            dream_phase, pocket_features, device,
        )
        
        # Validation
        val_metrics = validate(model, val_loader, loss_fn, pocket_features, device)
        
        print(f"Train: Loss={train_metrics['loss']:.4f}, Top1={train_metrics['top1']:.1f}%")
        print(f"Val:   Loss={val_metrics['loss']:.4f}, "
              f"Top1={val_metrics['top1']:.1f}%, Top3={val_metrics['top3']:.1f}%")
        print(f"Active states: {model.state_bank.num_active_states}, "
              f"Memory: {model.memory.n_entries}, "
              f"Dream memories: {len(dream_phase.dream_memory)}")
        
        # Dream phase
        if epoch % args.dream_every == 0:
            print("\n  💤 Dreaming...")
            dream_metrics = dream_phase.dream(model, optimizer, device)
            print(f"  Dream: replay_loss={dream_metrics['replay_loss']:.4f}, "
                  f"imagination_loss={dream_metrics['imagination_loss']:.4f}, "
                  f"rules={dream_metrics['rules_discovered']}")
            
            # Re-validate after dreaming
            val_metrics_post = validate(model, val_loader, loss_fn, pocket_features, device)
            print(f"  Post-dream Val: Top1={val_metrics_post['top1']:.1f}%, "
                  f"Top3={val_metrics_post['top3']:.1f}%")
            
            # Use post-dream metrics for best model selection
            val_metrics = val_metrics_post
        
        scheduler.step()
        
        if val_metrics['top1'] > best_val_top1:
            best_val_top1 = val_metrics['top1']
            torch.save({
                'model': model.state_dict(),
                'dream': dream_phase.state_dict(),
            }, 'best_dream_model.pt')
            print(f"  ✓ Saved best model (Top1={best_val_top1:.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best Val Top-1: {best_val_top1:.1f}%")
    print(f"Total dreams: {dream_phase.n_dreams}")
    print(f"Rules discovered: {dream_phase.rules_discovered}")
    print("=" * 70)


if __name__ == '__main__':
    main()
