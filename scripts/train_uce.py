#!/usr/bin/env python3
"""
UCE Training Script
===================

Trains the Unified Cognitive Engine on CYP3A4 data.

Usage:
    python train_uce.py --data path/to/data.json --epochs 50
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
except ImportError:
    print("PyTorch required")
    sys.exit(1)

try:
    from rdkit import Chem
except ImportError:
    print("RDKit required")
    sys.exit(1)

from src.enzyme_software.liquid_nn_v2.model.uce import (
    UnifiedCognitiveEngine, UCEConfig
)


def load_data(path: str) -> List[Dict]:
    with open(path) as f:
        raw = json.load(f)
    
    # Handle different JSON structures
    if isinstance(raw, list):
        return raw
    elif isinstance(raw, dict):
        # Check for common keys
        if 'drugs' in raw:
            return raw['drugs']
        elif 'molecules' in raw:
            return raw['molecules']
        elif 'data' in raw:
            return raw['data']
        else:
            # Maybe it's a dict of molecules keyed by name/id
            return list(raw.values()) if all(isinstance(v, dict) for v in raw.values()) else []
    return []


def get_smiles_and_sites(entry: Dict) -> Tuple[str, List[int]]:
    smiles = entry.get('smiles', entry.get('SMILES', ''))
    sites = []
    
    # Try different field names
    if 'site_atoms' in entry:
        sites = entry['site_atoms']
    elif 'som_site' in entry:
        sites = entry['som_site'] if isinstance(entry['som_site'], list) else [entry['som_site']]
    elif 'atom_indices' in entry:
        sites = entry['atom_indices']
    elif 'positive_atoms' in entry:
        sites = entry['positive_atoms']
    
    # Ensure it's a list
    if not isinstance(sites, list):
        sites = [sites] if sites is not None else []
    
    return smiles, sites


def evaluate(model: UnifiedCognitiveEngine, data: List[Dict]) -> Dict[str, float]:
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
                preds = model.predict(smiles, top_k=3)
                pred_idx = [p[0] for p in preds]
                
                top1 += int(pred_idx[0] in sites) if pred_idx else 0
                top3 += int(any(p in sites for p in pred_idx[:3]))
                total += 1
            except:
                continue
    
    return {
        'top1': top1 / max(total, 1),
        'top3': top3 / max(total, 1),
        'n': total,
    }


def train_epoch(model: UnifiedCognitiveEngine, data: List[Dict],
                optimizer: torch.optim.Optimizer) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    
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
        
        try:
            optimizer.zero_grad()
            output = model.forward(smiles, labels=labels)
            
            if output['loss'] is not None:
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += output['loss'].item()
                n_batches += 1
        except Exception as e:
            continue
    
    return total_loss / max(n_batches, 1), n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', default='checkpoints/uce')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=96)
    parser.add_argument('--wave_dim', type=int, default=32)
    parser.add_argument('--memory_dim', type=int, default=32)
    parser.add_argument('--max_ode_steps', type=int, default=15)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("UNIFIED COGNITIVE ENGINE (UCE) TRAINING")
    print("=" * 70)
    
    # Data
    print("\nLoading data...")
    data = load_data(args.data)
    random.shuffle(data)
    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Model
    print("\nCreating model...")
    config = UCEConfig(
        hidden_dim=args.hidden_dim,
        wave_dim=args.wave_dim,
        memory_dim=args.memory_dim,
        max_ode_steps=args.max_ode_steps,
    )
    model = UnifiedCognitiveEngine(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"State dim: {config.state_dim} (h={config.hidden_dim}, ρ={config.wave_dim}, m={config.memory_dim})")
    
    # Populate memory
    print("\nPopulating memory...")
    for entry in train_data:
        smiles, sites = get_smiles_and_sites(entry)
        if smiles and sites:
            model.add_to_memory(smiles, sites)
    print(f"Memory: {len(model.memory.memory_mols)} molecules")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Train
    print("\nTraining...")
    print("-" * 70)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        loss, n = train_epoch(model, train_data, optimizer)
        scheduler.step()
        
        metrics = evaluate(model, val_data)
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {loss:.4f} | "
              f"Top-1: {metrics['top1']*100:.1f}% | "
              f"Top-3: {metrics['top3']*100:.1f}%")
        
        if metrics['top1'] > best_acc:
            best_acc = metrics['top1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'metrics': metrics,
            }, os.path.join(args.output_dir, 'uce_best.pt'))
            print(f"  → Saved best (Top-1: {best_acc*100:.1f}%)")
    
    # Final
    print("\n" + "=" * 70)
    final = evaluate(model, val_data)
    print(f"Final: Top-1={final['top1']*100:.1f}%, Top-3={final['top3']*100:.1f}%")
    print(f"Best:  Top-1={best_acc*100:.1f}%")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, os.path.join(args.output_dir, 'uce_final.pt'))
    
    print(f"\nSaved to {args.output_dir}")


if __name__ == '__main__':
    main()
