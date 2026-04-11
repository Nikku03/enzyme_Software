#!/usr/bin/env python3
"""
Training script with Aromatic Oxidation Features + Dream Phase.

Integrates:
1. Hückel HOMO density features (32 dims)
2. Dream phase consolidation
3. Hybrid NEXUS architecture

Usage:
    python train_with_aromatic.py --epochs 50 --dream_every 5
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

from hybrid_nexus_dynamic import HybridNexusDynamic, HybridLoss
from dream_phase import DreamPhase
from aromatic_oxidation import AromaticFeatureExtractor


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET WITH AROMATIC FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class CYP3A4DatasetWithAromatic(Dataset):
    """Dataset with integrated aromatic oxidation features."""
    
    def __init__(
        self,
        data_path: str,
        sources: Optional[List[str]] = None,
        max_atoms: int = 200,
    ):
        self.max_atoms = max_atoms
        self.aromatic_extractor = AromaticFeatureExtractor(n_features=32)
        
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
        features, coords, n_atoms, aromatic_features = self._smiles_to_features(mol_data['smiles'])
        
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
            'aromatic_features': aromatic_features,
            'n_atoms': n_atoms,
        }
    
    def _smiles_to_features(self, smiles: str) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Convert SMILES to feature tensors including aromatic features."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            
            mol_with_h = Chem.AddHs(mol)
            n_atoms = min(mol_with_h.GetNumAtoms(), self.max_atoms)
            
            # Generate 3D coordinates
            conf = None
            try:
                result = AllChem.EmbedMolecule(mol_with_h, AllChem.ETKDGv3())
                if result == 0 and mol_with_h.GetNumConformers() > 0:
                    AllChem.MMFFOptimizeMolecule(mol_with_h, maxIters=200)
                    conf = mol_with_h.GetConformer()
            except:
                pass
            
            if conf is None:
                try:
                    result = AllChem.EmbedMolecule(mol_with_h, useRandomCoords=True, randomSeed=42)
                    if result == 0 and mol_with_h.GetNumConformers() > 0:
                        conf = mol_with_h.GetConformer()
                except:
                    pass
            
            # Extract aromatic features (on mol without H for indexing consistency)
            try:
                aromatic_feats = self.aromatic_extractor.extract(mol)
            except:
                aromatic_feats = {}
            
            features = torch.zeros(self.max_atoms, 128)
            coords = torch.zeros(self.max_atoms, 3)
            aromatic_features = torch.zeros(self.max_atoms, 32)
            
            for i in range(n_atoms):
                atom = mol_with_h.GetAtomWithIdx(i)
                z = atom.GetAtomicNum()
                
                # Basic features (0-30)
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
                
                # Add aromatic features if available
                # Map from mol (no H) to mol_with_h indices
                if i < mol.GetNumAtoms() and i in aromatic_feats:
                    aromatic_features[i, :] = torch.tensor(aromatic_feats[i], dtype=torch.float32)
                    # Also add key aromatic features to main features
                    features[i, 30:35] = aromatic_features[i, :5]  # HOMO, LUMO, dual, gap, HOMO_sq
                
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
            
            return features, coords, n_atoms, aromatic_features
            
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
            aromatic_features = torch.zeros(self.max_atoms, 32)
            return features, coords, n_atoms, aromatic_features


def collate_fn(batch):
    return {
        'features': torch.stack([b['features'] for b in batch]),
        'coords': torch.stack([b['coords'] for b in batch]),
        'som_mask': torch.stack([b['som_mask'] for b in batch]),
        'valid_mask': torch.stack([b['valid_mask'] for b in batch]),
        'aromatic_features': torch.stack([b['aromatic_features'] for b in batch]),
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
# ENHANCED MODEL WITH AROMATIC HEAD
# ═══════════════════════════════════════════════════════════════════════════════

class HybridNexusWithAromatic(nn.Module):
    """Hybrid model with dedicated aromatic oxidation head."""
    
    def __init__(
        self,
        mol_dim: int = 128,
        hidden_dim: int = 64,
        aromatic_dim: int = 32,
        max_states: int = 32,
        similarity_threshold: float = 0.7,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Base model
        self.base_model = HybridNexusDynamic(
            mol_dim=mol_dim,
            hidden_dim=hidden_dim,
            max_states=max_states,
            similarity_threshold=similarity_threshold,
        )
        
        # Aromatic-specific head
        self.aromatic_encoder = nn.Sequential(
            nn.Linear(aromatic_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Aromatic scorer (learned weights based on our analysis)
        self.aromatic_scorer = nn.Sequential(
            nn.Linear(aromatic_dim + hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Gate to combine base and aromatic predictions
        self.aromatic_gate = nn.Sequential(
            nn.Linear(hidden_dim + aromatic_dim, 1),
            nn.Sigmoid(),
        )
        
        # Initialize aromatic scorer with learned weights
        self._init_aromatic_weights()
    
    def _init_aromatic_weights(self):
        """Initialize with learned optimal weights from logistic regression."""
        # These weights were learned from the dataset
        learned_weights = torch.tensor([
            0.17,   # 0: HOMO_density
            0.0,    # 1: LUMO_density
            0.0,    # 2: dual_desc
            0.0,    # 3: HOMO-LUMO_gap
            0.0,    # 4: HOMO_sq
            0.0,    # 5: sigma_total
            0.15,   # 6: EDG_effect
            0.0,    # 7: EWG_effect
            -0.18,  # 8: ortho_effect
            0.0,    # 9: para_effect
            -0.32,  # 10: n_fused_rings
            0.12,   # 11: is_bay_region
            0.0,    # 12: is_k_region
            0.0,    # 13: ring_size
            0.19,   # 14: fusion_degree
            0.0,    # 15: is_C
            0.0,    # 16: is_N
            0.0,    # 17: is_O
            0.18,   # 18: n_H
            -0.18,  # 19: n_neighbors
            0.0,    # 20: HOMO_x_H
            0.0,    # 21: HOMO_x_EDG
            0.16,   # 22: HOMO_x_fusion
            0.0,    # 23: bay_x_HOMO
            -0.21,  # 24: weak_sub_x_HOMO
            0.32,   # 25: neighbor_HOMO_mean
            -0.24,  # 26: neighbor_HOMO_max
            0.0,    # 27: HOMO_rel_neighbors
            0.0, 0.0, 0.0, 0.0,  # 28-31: padding
        ])
        
        # Set as learnable but initialized
        with torch.no_grad():
            # Initialize first layer of aromatic_scorer
            self.aromatic_scorer[0].weight[:, :32] = learned_weights.unsqueeze(0).expand(32, -1) * 0.1
    
    @property
    def mol_encoder(self):
        return self.base_model.mol_encoder
    
    @property
    def state_bank(self):
        return self.base_model.state_bank
    
    @property
    def memory(self):
        return self.base_model.memory
    
    @property
    def fusion(self):
        return self.base_model.fusion
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        pocket_features: torch.Tensor,
        aromatic_features: torch.Tensor,
        som_mask: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass with aromatic integration."""
        # Get base model outputs
        base_outputs = self.base_model(
            features, coords, pocket_features, som_mask, valid_mask
        )
        
        B, N = features.shape[:2]
        
        # Encode aromatic features
        aromatic_encoded = self.aromatic_encoder(aromatic_features)  # [B, N, hidden]
        
        # Aromatic-specific scores
        aromatic_input = torch.cat([aromatic_features, aromatic_encoded], dim=-1)
        aromatic_scores = self.aromatic_scorer(aromatic_input).squeeze(-1)  # [B, N]
        
        # Check which atoms are aromatic (aromatic_features sum > 0)
        is_aromatic = (aromatic_features.abs().sum(dim=-1) > 0.01).float()
        
        # Compute gate (how much to trust aromatic vs base)
        mol_encoded = self.mol_encoder(features)
        gate_input = torch.cat([mol_encoded, aromatic_features], dim=-1)
        gate = self.aromatic_gate(gate_input).squeeze(-1)  # [B, N]
        
        # Only apply aromatic gate to aromatic atoms
        gate = gate * is_aromatic
        
        # Combine scores
        base_scores = base_outputs['final_scores']
        combined_scores = (1 - gate) * base_scores + gate * aromatic_scores
        
        # Mask invalid
        if valid_mask is not None:
            combined_scores = combined_scores.masked_fill(~valid_mask, -1e4)
            aromatic_scores = aromatic_scores.masked_fill(~valid_mask, -1e4)
        
        return {
            'final_scores': combined_scores,
            'base_scores': base_scores,
            'aromatic_scores': aromatic_scores,
            'aromatic_gate': gate,
            'physics_scores': base_outputs['physics_scores'],
            'state_probs': base_outputs.get('state_probs'),
        }


class HybridAromaticLoss(nn.Module):
    """Loss function with aromatic-specific terms."""
    
    def __init__(
        self,
        aromatic_weight: float = 0.3,
        physics_weight: float = 0.2,
    ):
        super().__init__()
        self.aromatic_weight = aromatic_weight
        self.physics_weight = physics_weight
        self.base_loss = HybridLoss(physics_weight=physics_weight)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        som_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        aromatic_mask: Optional[torch.Tensor] = None,
    ):
        # Base loss
        base_loss, base_metrics = self.base_loss(outputs, som_mask, valid_mask)
        
        # Aromatic-specific loss (only for aromatic SoMs)
        aromatic_loss = torch.tensor(0.0, device=som_mask.device)
        
        if aromatic_mask is not None and aromatic_mask.sum() > 0:
            aromatic_scores = outputs['aromatic_scores']
            
            # Target: aromatic SoM atoms
            aromatic_som = som_mask * aromatic_mask
            
            if aromatic_som.sum() > 0:
                # Cross-entropy style loss for aromatic predictions
                log_probs = F.log_softmax(aromatic_scores, dim=-1).clamp(min=-100)
                aromatic_som_norm = aromatic_som / (aromatic_som.sum(dim=-1, keepdim=True) + 1e-8)
                aromatic_loss = -(aromatic_som_norm * log_probs).sum(dim=-1)
                aromatic_loss = aromatic_loss[aromatic_som.sum(dim=-1) > 0].mean()
        
        total_loss = base_loss + self.aromatic_weight * aromatic_loss
        
        metrics = base_metrics.copy()
        metrics['aromatic_loss'] = aromatic_loss.item() if torch.is_tensor(aromatic_loss) else aromatic_loss
        
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


def train_epoch(model, loader, optimizer, loss_fn, dream_phase, pocket_features, device):
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
        aromatic_features = batch['aromatic_features'].to(device)
        
        # Create aromatic mask (which atoms are aromatic)
        aromatic_mask = (aromatic_features.abs().sum(dim=-1) > 0.01).float()
        
        optimizer.zero_grad()
        
        outputs = model(
            features, coords,
            pocket_features.to(device),
            aromatic_features,
            som_mask, valid_mask,
        )
        
        loss, metrics = loss_fn(outputs, som_mask, valid_mask, aromatic_mask)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record for dreaming
        if dream_phase is not None:
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
                  f"Loss={loss.item():.4f}, Top1={acc_metrics['top1']:.1f}%, "
                  f"Arom={metrics['aromatic_loss']:.3f}")
    
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
    aromatic_top1 = 0
    aromatic_count = 0
    n_batches = 0
    
    for batch in loader:
        features = batch['features'].to(device)
        coords = batch['coords'].to(device)
        som_mask = batch['som_mask'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        aromatic_features = batch['aromatic_features'].to(device)
        
        aromatic_mask = (aromatic_features.abs().sum(dim=-1) > 0.01).float()
        
        outputs = model(features, coords, pocket_features.to(device), aromatic_features, None, valid_mask)
        loss, _ = loss_fn(outputs, som_mask, valid_mask, aromatic_mask)
        
        acc_metrics = compute_metrics(outputs['final_scores'], som_mask, valid_mask)
        
        # Track aromatic-specific accuracy
        B = features.shape[0]
        for b in range(B):
            aromatic_som = som_mask[b] * aromatic_mask[b]
            if aromatic_som.sum() > 0:
                aromatic_count += 1
                scores = outputs['final_scores'][b]
                pred = scores.argmax()
                if aromatic_som[pred] > 0:
                    aromatic_top1 += 1
        
        total_loss += loss.item()
        total_top1 += acc_metrics['top1']
        total_top3 += acc_metrics['top3']
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'top1': total_top1 / n_batches,
        'top3': total_top3 / n_batches,
        'aromatic_top1': aromatic_top1 / max(aromatic_count, 1) * 100,
        'aromatic_count': aromatic_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--max_states', type=int, default=32)
    parser.add_argument('--similarity_threshold', type=float, default=0.7)
    parser.add_argument('--dream_every', type=int, default=5)
    parser.add_argument('--aromatic_weight', type=float, default=0.3)
    parser.add_argument('--train_sources', type=str, default='Zaretzki,DrugBank,MetXBioDB,AZ120')
    parser.add_argument('--val_sources', type=str, default='Zaretzki,DrugBank,MetXBioDB,AZ120')
    parser.add_argument('--data_path', type=str, default='data/curated/merged_cyp3a4_extended.json')
    parser.add_argument('--pdb_path', type=str, default='1W0E.pdb')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_dream', action='store_true', help='Disable dream phase')
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("CYP3A4 SoM Predictor - Hybrid + Aromatic + Dreams")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Aromatic weight: {args.aromatic_weight}")
    print(f"Dream every: {args.dream_every} epochs" if not args.no_dream else "Dreams: DISABLED")
    
    pocket_features = load_pocket_features(args.pdb_path)
    print(f"Loaded {pocket_features.shape[0]} pocket atoms")
    
    train_sources = args.train_sources.split(',')
    val_sources = args.val_sources.split(',')
    
    print(f"\nTrain sources: {train_sources}")
    print(f"Val sources: {val_sources}")
    
    full_dataset = CYP3A4DatasetWithAromatic(args.data_path, sources=train_sources)
    
    if set(train_sources) == set(val_sources):
        print("Same sources - using 80/20 split")
        n_train = int(0.8 * len(full_dataset))
        n_val = len(full_dataset) - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )
    else:
        train_dataset = full_dataset
        val_dataset = CYP3A4DatasetWithAromatic(args.data_path, sources=val_sources)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    
    # Model with aromatic head
    model = HybridNexusWithAromatic(
        mol_dim=128,
        hidden_dim=args.hidden_dim,
        aromatic_dim=32,
        max_states=args.max_states,
        similarity_threshold=args.similarity_threshold,
    ).to(device)
    
    # Dream phase
    dream_phase = None
    if not args.no_dream:
        dream_phase = DreamPhase(hidden_dim=args.hidden_dim).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    loss_fn = HybridAromaticLoss(aromatic_weight=args.aromatic_weight)
    
    params = list(model.parameters())
    if dream_phase:
        params += list(dream_phase.parameters())
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_top1 = 0
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn,
            dream_phase, pocket_features, device,
        )
        
        val_metrics = validate(model, val_loader, loss_fn, pocket_features, device)
        
        print(f"Train: Loss={train_metrics['loss']:.4f}, Top1={train_metrics['top1']:.1f}%")
        print(f"Val:   Loss={val_metrics['loss']:.4f}, "
              f"Top1={val_metrics['top1']:.1f}%, Top3={val_metrics['top3']:.1f}%")
        print(f"Aromatic Val Top1: {val_metrics['aromatic_top1']:.1f}% "
              f"(n={val_metrics['aromatic_count']})")
        print(f"States: {model.state_bank.num_active_states}, "
              f"Memory: {model.memory.n_entries}")
        
        # Dream phase
        if dream_phase and epoch % args.dream_every == 0:
            print("\n  💤 Dreaming...")
            dream_metrics = dream_phase.dream(model.base_model, optimizer, device)
            print(f"  Dream: replay={dream_metrics['replay_loss']:.4f}, "
                  f"imagination={dream_metrics['imagination_loss']:.4f}, "
                  f"rules={dream_metrics['rules_discovered']}")
            
            val_metrics_post = validate(model, val_loader, loss_fn, pocket_features, device)
            print(f"  Post-dream: Top1={val_metrics_post['top1']:.1f}%, "
                  f"Aromatic={val_metrics_post['aromatic_top1']:.1f}%")
            val_metrics = val_metrics_post
        
        scheduler.step()
        
        if val_metrics['top1'] > best_val_top1:
            best_val_top1 = val_metrics['top1']
            torch.save({
                'model': model.state_dict(),
                'dream': dream_phase.state_dict() if dream_phase else None,
                'aromatic_top1': val_metrics['aromatic_top1'],
            }, 'best_aromatic_model.pt')
            print(f"  ✓ Saved best model (Top1={best_val_top1:.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best Val Top-1: {best_val_top1:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
