#!/usr/bin/env python3
"""
Training with Dynamic Binding Pose Model.

Models the fact that CYP450 metabolism involves:
1. Multiple substrate binding orientations
2. Enzyme conformational flexibility
3. Dynamic competition between reactive sites

Usage:
    python train_dynamic.py --epochs 50 --n_poses 10
"""

import argparse
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dynamic_binding import DynamicSoMPredictor, extract_pose_features, PoseEnsemble


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET WITH POSE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class CYP3A4DynamicDataset(Dataset):
    """Dataset with dynamic pose features."""
    
    def __init__(
        self,
        data_path: str,
        sources: Optional[List[str]] = None,
        max_atoms: int = 200,
        n_poses: int = 10,
        precompute_poses: bool = False,
    ):
        self.max_atoms = max_atoms
        self.n_poses = n_poses
        self.pose_ensemble = PoseEnsemble(n_poses=n_poses)
        
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
        features, coords, n_atoms, pose_features = self._smiles_to_features(mol_data['smiles'])
        
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
            'pose_features': pose_features,
            'n_atoms': n_atoms,
        }
    
    def _smiles_to_features(self, smiles: str) -> Tuple[torch.Tensor, ...]:
        """Convert SMILES to features including pose features."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES")
            
            n_heavy = mol.GetNumAtoms()
            
            # Generate pose features (on mol without H)
            pose_features = extract_pose_features(mol, n_poses=self.n_poses, max_atoms=self.max_atoms)
            
            mol_with_h = Chem.AddHs(mol)
            n_atoms = min(mol_with_h.GetNumAtoms(), self.max_atoms)
            
            # Generate 3D coordinates
            conf = None
            try:
                result = AllChem.EmbedMolecule(mol_with_h, AllChem.ETKDGv3())
                if result == 0 and mol_with_h.GetNumConformers() > 0:
                    AllChem.MMFFOptimizeMolecule(mol_with_h, maxIters=100)
                    conf = mol_with_h.GetConformer()
            except:
                pass
            
            if conf is None:
                try:
                    AllChem.EmbedMolecule(mol_with_h, useRandomCoords=True, randomSeed=42)
                    if mol_with_h.GetNumConformers() > 0:
                        conf = mol_with_h.GetConformer()
                except:
                    pass
            
            features = torch.zeros(self.max_atoms, 128)
            coords = torch.zeros(self.max_atoms, 3)
            
            for i in range(n_atoms):
                atom = mol_with_h.GetAtomWithIdx(i)
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
                
                # Add pose-derived features
                if i < n_heavy:
                    # Average accessibility across poses
                    avg_accessible = pose_features[i, :, 1].mean().item()
                    avg_fe_dist = pose_features[i, :, 0].mean().item()
                    features[i, 30] = avg_accessible
                    features[i, 31] = avg_fe_dist
                
                if conf is not None:
                    try:
                        pos = conf.GetAtomPosition(i)
                        coords[i] = torch.tensor([pos.x, pos.y, pos.z])
                    except:
                        coords[i] = torch.tensor([(i * 1.5) % 10 - 5, (z * 0.7) % 10 - 5, i * 0.3])
                else:
                    coords[i] = torch.tensor([(i * 1.5) % 10 - 5, (z * 0.7) % 10 - 5, i * 0.3])
            
            return features, coords, n_atoms, pose_features
            
        except Exception as e:
            n_atoms = 20
            features = torch.zeros(self.max_atoms, 128)
            features[:n_atoms, 0] = 0.3
            coords = torch.zeros(self.max_atoms, 3)
            pose_features = torch.zeros(self.max_atoms, self.n_poses, 3)
            return features, coords, n_atoms, pose_features


def collate_fn(batch):
    return {
        'features': torch.stack([b['features'] for b in batch]),
        'coords': torch.stack([b['coords'] for b in batch]),
        'som_mask': torch.stack([b['som_mask'] for b in batch]),
        'valid_mask': torch.stack([b['valid_mask'] for b in batch]),
        'pose_features': torch.stack([b['pose_features'] for b in batch]),
        'n_atoms': [b['n_atoms'] for b in batch],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicLoss(nn.Module):
    """Loss for dynamic model with pose diversity encouragement."""
    
    def __init__(self, pose_diversity_weight: float = 0.1):
        super().__init__()
        self.pose_diversity_weight = pose_diversity_weight
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        som_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        final_scores = outputs['final_scores']
        
        # Normalize SoM mask
        som_sum = som_mask.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        som_normalized = som_mask / som_sum
        
        # Main ranking loss
        log_probs = F.log_softmax(final_scores, dim=-1).clamp(min=-100)
        main_loss = -(som_normalized * log_probs).sum(dim=-1).mean()
        
        # Pose diversity loss: encourage different poses to predict different atoms
        pose_weights = outputs['pose_weights']  # [B, N, n_poses]
        pose_reactivities = outputs['pose_reactivities']  # [B, N, n_poses]
        
        # Entropy over poses (higher = more diverse)
        pose_entropy = -(pose_weights * (pose_weights + 1e-8).log()).sum(dim=-1).mean()
        diversity_loss = -pose_entropy  # Minimize negative entropy = maximize entropy
        
        total_loss = main_loss + self.pose_diversity_weight * diversity_loss
        
        metrics = {
            'main_loss': main_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'pose_entropy': pose_entropy.item(),
        }
        
        return total_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
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


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    
    total_loss = 0
    total_top1 = 0
    total_top3 = 0
    total_entropy = 0
    n_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        features = batch['features'].to(device)
        pose_features = batch['pose_features'].to(device)
        som_mask = batch['som_mask'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(features, pose_features, valid_mask)
        loss, metrics = loss_fn(outputs, som_mask, valid_mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        with torch.no_grad():
            acc_metrics = compute_metrics(outputs['final_scores'], som_mask, valid_mask)
        
        total_loss += loss.item()
        total_top1 += acc_metrics['top1']
        total_top3 += acc_metrics['top3']
        total_entropy += metrics['pose_entropy']
        n_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: "
                  f"Loss={loss.item():.4f}, Top1={acc_metrics['top1']:.1f}%, "
                  f"Entropy={metrics['pose_entropy']:.3f}")
    
    return {
        'loss': total_loss / n_batches,
        'top1': total_top1 / n_batches,
        'top3': total_top3 / n_batches,
        'pose_entropy': total_entropy / n_batches,
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    
    total_loss = 0
    total_top1 = 0
    total_top3 = 0
    n_batches = 0
    
    # Track pose usage
    all_pose_weights = []
    
    for batch in loader:
        features = batch['features'].to(device)
        pose_features = batch['pose_features'].to(device)
        som_mask = batch['som_mask'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        
        outputs = model(features, pose_features, valid_mask)
        loss, _ = loss_fn(outputs, som_mask, valid_mask)
        
        acc_metrics = compute_metrics(outputs['final_scores'], som_mask, valid_mask)
        
        total_loss += loss.item()
        total_top1 += acc_metrics['top1']
        total_top3 += acc_metrics['top3']
        n_batches += 1
        
        # Track pose weights
        all_pose_weights.append(outputs['pose_weights'].cpu())
    
    # Analyze pose usage
    all_weights = torch.cat(all_pose_weights, dim=0)  # [total_samples, N, n_poses]
    avg_pose_usage = all_weights.mean(dim=(0, 1))  # [n_poses]
    
    return {
        'loss': total_loss / n_batches,
        'top1': total_top1 / n_batches,
        'top3': total_top3 / n_batches,
        'pose_usage': avg_pose_usage.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_poses', type=int, default=10)
    parser.add_argument('--n_enzyme_confs', type=int, default=4)
    parser.add_argument('--pose_diversity_weight', type=float, default=0.1)
    parser.add_argument('--train_sources', type=str, default='Zaretzki,DrugBank,MetXBioDB,AZ120')
    parser.add_argument('--val_sources', type=str, default='Zaretzki,DrugBank,MetXBioDB,AZ120')
    parser.add_argument('--data_path', type=str, default='data/curated/merged_cyp3a4_extended.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("CYP3A4 SoM Predictor - DYNAMIC BINDING MODEL")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Number of poses: {args.n_poses}")
    print(f"Enzyme conformations: {args.n_enzyme_confs}")
    print(f"Pose diversity weight: {args.pose_diversity_weight}")
    
    train_sources = args.train_sources.split(',')
    val_sources = args.val_sources.split(',')
    
    print(f"\nTrain sources: {train_sources}")
    print(f"Val sources: {val_sources}")
    
    print("\nLoading dataset (generating poses - this may take a while)...")
    full_dataset = CYP3A4DynamicDataset(
        args.data_path, sources=train_sources, n_poses=args.n_poses
    )
    
    if set(train_sources) == set(val_sources):
        print("Same sources - using 80/20 split")
        n_train = int(0.8 * len(full_dataset))
        n_val = len(full_dataset) - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )
    else:
        train_dataset = full_dataset
        val_dataset = CYP3A4DynamicDataset(
            args.data_path, sources=val_sources, n_poses=args.n_poses
        )
    
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
    model = DynamicSoMPredictor(
        atom_dim=128,
        hidden_dim=args.hidden_dim,
        n_poses=args.n_poses,
        n_enzyme_confs=args.n_enzyme_confs,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    loss_fn = DynamicLoss(pose_diversity_weight=args.pose_diversity_weight)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_top1 = 0
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate(model, val_loader, loss_fn, device)
        
        scheduler.step()
        
        print(f"Train: Loss={train_metrics['loss']:.4f}, Top1={train_metrics['top1']:.1f}%, "
              f"Entropy={train_metrics['pose_entropy']:.3f}")
        print(f"Val:   Loss={val_metrics['loss']:.4f}, "
              f"Top1={val_metrics['top1']:.1f}%, Top3={val_metrics['top3']:.1f}%")
        print(f"Pose usage: {[f'{p:.2f}' for p in val_metrics['pose_usage'][:5]]}")
        
        if val_metrics['top1'] > best_val_top1:
            best_val_top1 = val_metrics['top1']
            torch.save(model.state_dict(), 'best_dynamic_model.pt')
            print(f"  ✓ Saved best model (Top1={best_val_top1:.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best Val Top-1: {best_val_top1:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
