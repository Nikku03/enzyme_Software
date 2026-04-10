#!/usr/bin/env python3
"""
UCME Training Script
====================

Trains the Unified Cognitive Metabolism Engine on CYP3A4 data.

Usage:
    python train_ucme.py --data path/to/data.json --epochs 50 --output_dir checkpoints/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available")
    TORCH_AVAILABLE = False
    sys.exit(1)

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    print("RDKit not available")
    RDKIT_AVAILABLE = False
    sys.exit(1)

from src.enzyme_software.liquid_nn_v2.model.ucme import UCME, UCMEConfig, create_ucme


def load_data(data_path: str) -> List[Dict]:
    """Load CYP3A4 dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def prepare_molecule(entry: Dict) -> Tuple[str, List[int]]:
    """Extract SMILES and SoM sites from data entry."""
    smiles = entry.get('smiles', entry.get('SMILES', ''))
    
    # Get SoM sites
    som_sites = []
    if 'som_site' in entry:
        if isinstance(entry['som_site'], list):
            som_sites = entry['som_site']
        else:
            som_sites = [entry['som_site']]
    elif 'atom_indices' in entry:
        som_sites = entry['atom_indices']
    elif 'positive_atoms' in entry:
        som_sites = entry['positive_atoms']
    
    return smiles, som_sites


def evaluate(model: UCME, data: List[Dict], device: torch.device) -> Dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()
    
    top1_correct = 0
    top3_correct = 0
    total = 0
    
    with torch.no_grad():
        for entry in data:
            smiles, som_sites = prepare_molecule(entry)
            if not smiles or not som_sites:
                continue
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                predictions = model.predict(smiles, top_k=3)
            except Exception as e:
                continue
            
            predicted_indices = [p[0] for p in predictions]
            
            # Check if any ground truth site is in predictions
            top1_correct += int(predicted_indices[0] in som_sites) if predicted_indices else 0
            top3_correct += int(any(p in som_sites for p in predicted_indices[:3]))
            total += 1
    
    return {
        'top1_accuracy': top1_correct / max(total, 1),
        'top3_accuracy': top3_correct / max(total, 1),
        'total': total,
    }


def train_epoch(model: UCME, data: List[Dict], optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for entry in data:
        smiles, som_sites = prepare_molecule(entry)
        if not smiles or not som_sites:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        # Create labels
        n_atoms = mol.GetNumAtoms()
        labels = torch.zeros(n_atoms)
        for site in som_sites:
            if site < n_atoms:
                labels[site] = 1.0
        
        try:
            optimizer.zero_grad()
            output = model.forward(smiles, labels=labels)
            
            if output['loss'] is not None:
                loss = output['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        except Exception as e:
            continue
    
    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train UCME model")
    parser.add_argument('--data', required=True, help='Path to data JSON')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--use_wave', action='store_true', default=True)
    parser.add_argument('--use_liquid', action='store_true', default=True)
    parser.add_argument('--use_analogical', action='store_true', default=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print("=" * 70)
    print("UNIFIED COGNITIVE METABOLISM ENGINE (UCME)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Load data
    print("Loading data...")
    data = load_data(args.data)
    print(f"Loaded {len(data)} molecules")
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create model
    print("\nCreating UCME model...")
    config = UCMEConfig(
        hidden_dim=args.hidden_dim,
        use_wave=args.use_wave,
        use_liquid=args.use_liquid,
        use_analogical=args.use_analogical,
    )
    model = UCME(config).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    # Populate memory with training data
    if model.analogical is not None:
        print("\nPopulating analogical memory...")
        for entry in train_data:
            smiles, som_sites = prepare_molecule(entry)
            if smiles and som_sites:
                model.add_to_memory(smiles, som_sites)
        print(f"Memory size: {len(model.analogical.memory.memory)}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\nStarting training...")
    print("-" * 70)
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_data, optimizer, device)
        scheduler.step()
        
        # Evaluate
        val_metrics = evaluate(model, val_data, device)
        
        # Log
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Top-1: {val_metrics['top1_accuracy']*100:.1f}% | "
              f"Val Top-3: {val_metrics['top3_accuracy']*100:.1f}%")
        
        # Save best
        if val_metrics['top1_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['top1_accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_metrics': val_metrics,
            }, os.path.join(args.output_dir, 'ucme_best.pt'))
            print(f"  → Saved best model (Top-1: {best_val_acc*100:.1f}%)")
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)
    
    final_metrics = evaluate(model, val_data, device)
    print(f"Top-1 Accuracy: {final_metrics['top1_accuracy']*100:.1f}%")
    print(f"Top-3 Accuracy: {final_metrics['top3_accuracy']*100:.1f}%")
    print(f"Best Top-1: {best_val_acc*100:.1f}%")
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': final_metrics,
    }, os.path.join(args.output_dir, 'ucme_final.pt'))
    
    print(f"\nModels saved to {args.output_dir}")


if __name__ == '__main__':
    main()
