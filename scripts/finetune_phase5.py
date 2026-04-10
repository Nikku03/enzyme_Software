#!/usr/bin/env python3
"""
Fine-tune Phase 5 Checkpoint on New 869-molecule Dataset
=========================================================

This script:
1. Loads the existing Phase 5 checkpoint (47% Top-1)
2. Fine-tunes on the new merged_cyp3a4_extended.json (869 molecules)
3. Should improve beyond 47%

Usage:
    python finetune_phase5.py \
        --checkpoint /path/to/hybrid_full_xtb_best.pt \
        --data data/curated/merged_cyp3a4_extended.json \
        --output_dir checkpoints/phase5_finetuned \
        --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
except ImportError:
    print("PyTorch required")
    sys.exit(1)

try:
    from rdkit import Chem
except ImportError:
    print("RDKit required")
    sys.exit(1)

# Try to import the hybrid model
try:
    from enzyme_software.liquid_nn_v2.model.hybrid_model import HybridLNNModel
    from enzyme_software.liquid_nn_v2 import LiquidMetabolismNetV2, ModelConfig
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import hybrid model: {e}")
    HYBRID_AVAILABLE = False


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
# CHECKPOINT LOADING
# =============================================================================

def load_phase5_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load Phase 5 checkpoint and reconstruct the model.
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    config_data = ckpt.get('config', ckpt.get('model_config', None))
    
    if config_data is None:
        print("No config in checkpoint, using defaults...")
        config = ModelConfig()
    elif isinstance(config_data, dict):
        print("Converting dict config to ModelConfig...")
        config = ModelConfig(**{k: v for k, v in config_data.items() 
                                if hasattr(ModelConfig, k) or k in ModelConfig.__dataclass_fields__})
    else:
        config = config_data
    
    # Create base LNN model
    base_lnn = LiquidMetabolismNetV2(config)
    
    # Wrap in hybrid model
    model = HybridLNNModel(base_lnn)
    
    # Load state dict
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    
    # Try to load, handling missing/extra keys
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint (non-strict)")
    except Exception as e:
        print(f"Warning loading state dict: {e}")
        # Try loading just the base LNN
        try:
            base_lnn.load_state_dict(state_dict, strict=False)
            print("Loaded base LNN only")
        except:
            print("Could not load state dict, starting fresh")
    
    return model.to(device), config


# =============================================================================
# SIMPLE BATCH CREATION (no external featurizer needed)
# =============================================================================

def featurize_atom(atom) -> List[float]:
    """Simple atom featurization."""
    features = [
        atom.GetAtomicNum() / 50.0,
        atom.GetTotalDegree() / 4.0,
        atom.GetTotalNumHs() / 4.0,
        atom.GetFormalCharge() / 2.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        atom.GetMass() / 100.0,
    ]
    # Hybridization one-hot
    hyb = atom.GetHybridization()
    features.extend([
        1.0 if hyb == Chem.HybridizationType.SP else 0.0,
        1.0 if hyb == Chem.HybridizationType.SP2 else 0.0,
        1.0 if hyb == Chem.HybridizationType.SP3 else 0.0,
    ])
    return features


# BDE lookup from hydrogen_theft_v3
BDE_TABLE = {
    'ALPHA_O': 80, 'S_OXIDE': 80, 'ALPHA_N': 84, 'ALPHA_S': 85,
    'ALLYLIC': 86, 'PRIMARY': 96, 'AROMATIC': 97, 'SECONDARY': 98,
    'BENZYLIC': 99, 'AROMATIC_NO_H': 107, 'TERTIARY': 110, 'N_OXIDE': 111
}

def classify_carbon(mol, idx: int) -> str:
    """Classify carbon type for BDE estimation."""
    atom = mol.GetAtomWithIdx(idx)
    if atom.GetAtomicNum() != 6:
        return 'NON_CARBON'
    
    n_H = atom.GetTotalNumHs()
    if n_H == 0:
        return 'AROMATIC_NO_H' if atom.GetIsAromatic() else 'TERTIARY'
    
    neighbors = list(atom.GetNeighbors())
    
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
    
    if atom.GetIsAromatic():
        return 'AROMATIC'
    
    n_heavy = len(neighbors)
    if n_heavy == 1:
        return 'PRIMARY'
    elif n_heavy == 2:
        return 'SECONDARY'
    else:
        return 'TERTIARY'


def compute_physics_features(mol, device) -> Dict[str, torch.Tensor]:
    """Compute physics features dict for the model."""
    n_atoms = mol.GetNumAtoms()
    
    # BDE values
    bde_values = torch.zeros(n_atoms, device=device)
    for i in range(n_atoms):
        c_type = classify_carbon(mol, i)
        bde_values[i] = BDE_TABLE.get(c_type, 100)
    
    # Simple physics tensor
    physics_tensor = torch.zeros(n_atoms, 8, device=device)
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        physics_tensor[i, 0] = atom.GetAtomicNum() / 50.0
        physics_tensor[i, 1] = atom.GetTotalDegree() / 4.0
        physics_tensor[i, 2] = atom.GetTotalNumHs() / 4.0
        physics_tensor[i, 3] = 1.0 if atom.GetIsAromatic() else 0.0
        physics_tensor[i, 4] = bde_values[i] / 110.0  # Normalized BDE
        physics_tensor[i, 5] = 1.0 if atom.GetAtomicNum() == 6 else 0.0  # Is carbon
        physics_tensor[i, 6] = 1.0 if atom.IsInRing() else 0.0
        physics_tensor[i, 7] = atom.GetMass() / 100.0
    
    return {
        'bde_values': bde_values,
        'tensor': physics_tensor,
    }


def create_batch(smiles_list: List[str], sites_list: List[List[int]], 
                 device: torch.device) -> Optional[Dict]:
    """
    Create a batch for the hybrid model using simple featurization.
    """
    all_atom_features = []
    all_edge_index = []
    all_labels = []
    all_candidate_mask = []
    all_bde_values = []
    all_physics_tensors = []
    batch_idx = []
    all_smiles = []
    
    atom_offset = 0
    
    for mol_idx, (smiles, sites) in enumerate(zip(smiles_list, sites_list)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        n_atoms = mol.GetNumAtoms()
        
        # Atom features
        atom_feats = []
        for atom in mol.GetAtoms():
            atom_feats.append(featurize_atom(atom))
        atom_feats = torch.tensor(atom_feats, dtype=torch.float32, device=device)
        
        # Pad to expected dimension (128)
        if atom_feats.shape[1] < 128:
            padding = torch.zeros(n_atoms, 128 - atom_feats.shape[1], device=device)
            atom_feats = torch.cat([atom_feats, padding], dim=1)
        
        # Edge index
        edges_src, edges_dst = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges_src.extend([i + atom_offset, j + atom_offset])
            edges_dst.extend([j + atom_offset, i + atom_offset])
        
        # Labels
        labels = torch.zeros(n_atoms, device=device)
        for s in sites:
            if s < n_atoms:
                labels[s] = 1.0
        
        # Candidate mask (all carbons)
        candidate_mask = torch.tensor([
            mol.GetAtomWithIdx(i).GetAtomicNum() == 6
            for i in range(n_atoms)
        ], dtype=torch.bool, device=device)
        
        # Physics features
        physics = compute_physics_features(mol, device)
        all_bde_values.append(physics['bde_values'])
        all_physics_tensors.append(physics['tensor'])
        
        all_atom_features.append(atom_feats)
        all_labels.append(labels)
        all_candidate_mask.append(candidate_mask)
        batch_idx.extend([len(all_smiles)] * n_atoms)
        all_smiles.append(smiles)
        
        if edges_src:
            all_edge_index.append(torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device))
        
        atom_offset += n_atoms
    
    if not all_smiles:
        return None
    
    # Stack everything
    batch = {
        'atom_features': torch.cat(all_atom_features, dim=0),
        'x': torch.cat(all_atom_features, dim=0),
        'labels': torch.cat(all_labels, dim=0),
        'candidate_mask': torch.cat(all_candidate_mask, dim=0),
        'batch': torch.tensor(batch_idx, dtype=torch.long, device=device),
        'smiles': all_smiles,
        'num_atoms': atom_offset,
        'physics_features': {
            'bde_values': torch.cat(all_bde_values, dim=0),
            'tensor': torch.cat(all_physics_tensors, dim=0),
        },
    }
    
    if all_edge_index:
        batch['edge_index'] = torch.cat(all_edge_index, dim=1)
    else:
        batch['edge_index'] = torch.zeros(2, 0, dtype=torch.long, device=device)
    
    return batch


# =============================================================================
# TRAINING
# =============================================================================

def ranking_loss(scores: torch.Tensor, labels: torch.Tensor, 
                 mask: torch.Tensor) -> torch.Tensor:
    """Margin ranking loss."""
    pos_mask = (labels > 0) & mask
    neg_mask = (labels == 0) & mask
    
    if not pos_mask.any() or not neg_mask.any():
        if mask.any():
            return F.binary_cross_entropy_with_logits(
                scores[mask], labels[mask], reduction='mean'
            )
        return scores.new_zeros(1, requires_grad=True)
    
    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    
    # Sample negatives for efficiency
    n_neg = min(len(neg_scores), len(pos_scores) * 5)
    if n_neg < len(neg_scores):
        neg_idx = torch.randperm(len(neg_scores))[:n_neg]
        neg_scores = neg_scores[neg_idx]
    
    # Margin loss
    loss = 0.0
    n_pairs = 0
    for p in pos_scores:
        margin_violations = F.relu(1.0 - (p - neg_scores))
        loss = loss + margin_violations.sum()
        n_pairs += len(neg_scores)
    
    return loss / max(n_pairs, 1)


def evaluate_simple(model: nn.Module, data: List[Dict], device: torch.device
                    ) -> Dict[str, float]:
    """Simple evaluation without full batch pipeline."""
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
                # Create single-molecule batch
                batch = create_batch([smiles], [sites], device)
                if batch is None:
                    continue
                
                # Forward
                output = model(batch)
                
                # Get scores
                if isinstance(output, dict):
                    scores = output.get('site_logits', output.get('logits', None))
                else:
                    scores = output
                
                if scores is None:
                    continue
                
                scores = scores.squeeze()
                if scores.dim() == 0:
                    continue
                
                # Mask non-candidates
                candidate_mask = batch['candidate_mask']
                masked_scores = scores.clone()
                masked_scores[~candidate_mask] = float('-inf')
                
                top_k = min(3, candidate_mask.sum().item())
                if top_k == 0:
                    continue
                
                top_idx = torch.topk(masked_scores, top_k).indices.tolist()
                
                top1 += int(top_idx[0] in sites)
                top3 += int(any(i in sites for i in top_idx[:3]))
                total += 1
                
            except Exception as e:
                continue
    
    return {
        'top1': top1 / max(total, 1),
        'top3': top3 / max(total, 1),
        'n': total,
    }


def train_epoch(model: nn.Module, data: List[Dict], optimizer: torch.optim.Optimizer,
                device: torch.device, batch_size: int = 8, debug: bool = False) -> Tuple[float, int]:
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_errors = 0
    
    random.shuffle(data)
    
    # Process in batches
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        
        smiles_list = []
        sites_list = []
        for entry in batch_data:
            smiles, sites = get_smiles_and_sites(entry)
            if smiles and sites:
                smiles_list.append(smiles)
                sites_list.append(sites)
        
        if not smiles_list:
            continue
        
        try:
            batch = create_batch(smiles_list, sites_list, device)
            if batch is None:
                continue
            
            optimizer.zero_grad()
            
            # Forward
            output = model(batch)
            
            if debug and n_batches == 0:
                print(f"  DEBUG: output type = {type(output)}")
                if isinstance(output, dict):
                    print(f"  DEBUG: output keys = {output.keys()}")
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor):
                            print(f"    {k}: shape={v.shape}")
            
            # Get scores
            if isinstance(output, dict):
                scores = output.get('site_logits', output.get('logits', output.get('som_logits', None)))
                aux_loss = output.get('aux_loss', 0.0)
            else:
                scores = output
                aux_loss = 0.0
            
            if scores is None:
                if debug and n_errors < 3:
                    print(f"  DEBUG: No scores found in output")
                n_errors += 1
                continue
            
            scores = scores.squeeze()
            
            if debug and n_batches == 0:
                print(f"  DEBUG: scores shape = {scores.shape}, labels shape = {batch['labels'].shape}")
            
            # Compute loss
            loss = ranking_loss(scores, batch['labels'], batch['candidate_mask'])
            
            if isinstance(aux_loss, torch.Tensor):
                loss = loss + 0.1 * aux_loss
            
            if loss.requires_grad and loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
                
        except Exception as e:
            if debug and n_errors < 5:
                print(f"  DEBUG: Exception in batch {i}: {e}")
            n_errors += 1
            continue
    
    if n_batches == 0 and debug:
        print(f"  DEBUG: No batches trained! Errors: {n_errors}")
    
    return total_loss / max(n_batches, 1), n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to Phase 5 checkpoint')
    parser.add_argument('--data', required=True, help='Path to dataset JSON')
    parser.add_argument('--output_dir', default='checkpoints/phase5_finetuned')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)  # Lower LR for fine-tuning
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--freeze_backbone', action='store_true', 
                        help='Freeze base LNN, only train heads')
    args = parser.parse_args()
    
    if not HYBRID_AVAILABLE:
        print("ERROR: Hybrid model not available")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("PHASE 5 FINE-TUNING ON NEW DATA")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data: {args.data}")
    
    # Load data
    print("\nLoading data...")
    data = load_dataset(args.data)
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Load model
    print("\nLoading Phase 5 model...")
    model, config = load_phase5_checkpoint(args.checkpoint, device)
    
    # Initialize lazy parameters with a dummy forward pass
    print("Initializing model with dummy forward pass...")
    try:
        dummy_smiles = "CCO"  # Simple ethanol
        dummy_batch = create_batch([dummy_smiles], [[0]], device)
        if dummy_batch is not None:
            with torch.no_grad():
                _ = model(dummy_batch)
            print("Model initialized")
    except Exception as e:
        print(f"Warning during initialization: {e}")
    
    # Count parameters (only initialized ones)
    n_params = sum(p.numel() for p in model.parameters() if p.data.numel() > 0)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad and p.data.numel() > 0)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {n_trainable:,}")
    
    # Optionally freeze backbone
    if args.freeze_backbone:
        print("Freezing backbone...")
        for param in model.base_lnn.parameters():
            param.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable after freeze: {n_trainable:,}")
    
    # Initial evaluation
    print("\nInitial evaluation...")
    init_metrics = evaluate_simple(model, val_data[:50], device)
    print(f"Initial Top-1: {init_metrics['top1']*100:.1f}%, Top-3: {init_metrics['top3']*100:.1f}%")
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\nFine-tuning...")
    print("-" * 70)
    
    best_top1 = init_metrics['top1']
    patience_counter = 0
    patience = 5
    
    for epoch in range(args.epochs):
        # Train (debug on first epoch)
        train_loss, n = train_epoch(model, train_data, optimizer, device, args.batch_size, 
                                     debug=(epoch == 0))
        scheduler.step()
        
        # Evaluate
        val_metrics = evaluate_simple(model, val_data, device)
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Top-1: {val_metrics['top1']*100:.1f}% | "
              f"Top-3: {val_metrics['top3']*100:.1f}%")
        
        # Save best
        if val_metrics['top1'] > best_top1:
            best_top1 = val_metrics['top1']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
            }, os.path.join(args.output_dir, 'phase5_finetuned_best.pt'))
            print(f"  → Saved best (Top-1: {best_top1*100:.1f}%)")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    best_path = os.path.join(args.output_dir, 'phase5_finetuned_best.pt')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    
    final_metrics = evaluate_simple(model, val_data, device)
    print(f"Initial Top-1: {init_metrics['top1']*100:.1f}%")
    print(f"Best Top-1: {best_top1*100:.1f}%")
    print(f"Final Top-1: {final_metrics['top1']*100:.1f}%")
    print(f"Final Top-3: {final_metrics['top3']*100:.1f}%")
    
    # Save final
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': final_metrics,
    }, os.path.join(args.output_dir, 'phase5_finetuned_final.pt'))
    
    print(f"\nModels saved to {args.output_dir}")


if __name__ == '__main__':
    main()
