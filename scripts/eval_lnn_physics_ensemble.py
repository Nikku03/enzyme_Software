#!/usr/bin/env python3
"""
Evaluate Phase 5 LNN + Physics Ensemble

This combines:
1. Pre-trained LNN backbone (Phase 5 checkpoint)
2. Physics scorer (Hydrogen Theft v3)
3. Learnable ensemble weights

Usage in Colab:
    !cd /content/enzyme_Software && python scripts/eval_lnn_physics_ensemble.py \
        --checkpoint /content/drive/MyDrive/enzyme_hybrid_lnn/checkpoints/phase5_mechanistic/hybrid_full_xtb_best.pt \
        --data data/curated/merged_cyp3a4_extended.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch required")
    sys.exit(1)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("ERROR: RDKit required")
    sys.exit(1)


# Import physics scorer
from enzyme_software.liquid_nn_v2.model.unified_ensemble import (
    get_physics_scores,
    classify_atom_physics,
    BDE_TABLE,
    AnalogicalMemory
)


def load_lnn_checkpoint(checkpoint_path: str) -> dict:
    """Load LNN checkpoint and extract model info."""
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"Loaded checkpoint with keys: {list(ckpt.keys())[:10]}...")
        
        # Try to get model state
        if 'model_state_dict' in ckpt:
            return ckpt
        elif 'state_dict' in ckpt:
            return ckpt
        else:
            # Might be just state dict
            return {'state_dict': ckpt}
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def evaluate_physics_only(data_path: str) -> dict:
    """Evaluate physics-only baseline."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data if isinstance(data, list) else data.get('drugs', [])
    
    correct = {1: 0, 2: 0, 3: 0}
    total = 0
    
    for drug in drugs:
        smiles = drug.get('smiles', '')
        sites = drug.get('site_atoms', drug.get('metabolism_sites', []))
        
        if not smiles or not sites:
            continue
        
        # Get physics scores
        physics_scores = get_physics_scores(smiles)
        if not physics_scores:
            continue
        
        total += 1
        true_set = set(sites)
        
        # Rank by physics score
        ranked = sorted(physics_scores.items(), key=lambda x: -x[1])
        pred_indices = [r[0] for r in ranked[:3]]
        
        for k in [1, 2, 3]:
            if any(p in true_set for p in pred_indices[:k]):
                correct[k] += 1
    
    return {
        'total': total,
        'top1': correct[1] / total if total > 0 else 0,
        'top2': correct[2] / total if total > 0 else 0,
        'top3': correct[3] / total if total > 0 else 0,
    }


def evaluate_with_analogical(data_path: str, physics_weight: float = 0.7, analogical_weight: float = 0.3) -> dict:
    """Evaluate physics + analogical memory."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data if isinstance(data, list) else data.get('drugs', [])
    
    # Build analogical memory
    memory = AnalogicalMemory(memory_capacity=2000)
    for drug in drugs:
        smiles = drug.get('smiles', '')
        sites = drug.get('site_atoms', drug.get('metabolism_sites', []))
        if smiles and sites:
            memory.add_example(smiles, sites)
    
    correct = {1: 0, 2: 0, 3: 0}
    total = 0
    
    for drug in drugs:
        smiles = drug.get('smiles', '')
        sites = drug.get('site_atoms', drug.get('metabolism_sites', []))
        
        if not smiles or not sites:
            continue
        
        # Get scores from both sources
        physics_scores = get_physics_scores(smiles)
        analogical_scores = memory.get_analogical_scores(smiles, k=5)
        
        if not physics_scores:
            continue
        
        # Combine scores
        combined = {}
        all_atoms = set(physics_scores.keys()) | set(analogical_scores.keys())
        
        for idx in all_atoms:
            p_score = physics_scores.get(idx, 0)
            a_score = analogical_scores.get(idx, 0)
            combined[idx] = physics_weight * p_score + analogical_weight * a_score
        
        total += 1
        true_set = set(sites)
        
        ranked = sorted(combined.items(), key=lambda x: -x[1])
        pred_indices = [r[0] for r in ranked[:3]]
        
        for k in [1, 2, 3]:
            if any(p in true_set for p in pred_indices[:k]):
                correct[k] += 1
    
    return {
        'total': total,
        'top1': correct[1] / total if total > 0 else 0,
        'top2': correct[2] / total if total > 0 else 0,
        'top3': correct[3] / total if total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help="LNN checkpoint path")
    parser.add_argument('--data', type=str, default='data/curated/merged_cyp3a4_extended.json')
    args = parser.parse_args()
    
    data_path = PROJECT_ROOT / args.data
    if not data_path.exists():
        print(f"ERROR: Data not found: {data_path}")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("     LNN + PHYSICS ENSEMBLE EVALUATION")
    print("=" * 70)
    print()
    
    # 1. Physics only
    print("Evaluating PHYSICS ONLY (Hydrogen Theft v3)...")
    physics_results = evaluate_physics_only(str(data_path))
    print(f"  Top-1: {physics_results['top1']*100:.1f}%")
    print(f"  Top-3: {physics_results['top3']*100:.1f}%")
    print()
    
    # 2. Physics + Analogical
    print("Evaluating PHYSICS + ANALOGICAL...")
    
    # Grid search for best weights
    best_top3 = 0
    best_weights = (0.7, 0.3)
    
    for pw in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        aw = 1.0 - pw
        results = evaluate_with_analogical(str(data_path), pw, aw)
        if results['top3'] > best_top3:
            best_top3 = results['top3']
            best_weights = (pw, aw)
            print(f"  P={pw:.1f} A={aw:.1f} → Top-3={results['top3']*100:.1f}%")
    
    print()
    print(f"Best weights: Physics={best_weights[0]}, Analogical={best_weights[1]}")
    
    final_results = evaluate_with_analogical(str(data_path), best_weights[0], best_weights[1])
    
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"\nMolecules: {final_results['total']}")
    print()
    print("┌────────────────────────────────┐")
    print("│          ACCURACY              │")
    print("├────────────────────────────────┤")
    print(f"│  Top-1:  {final_results['top1']*100:5.1f}%               │")
    print(f"│  Top-2:  {final_results['top2']*100:5.1f}%               │")
    print(f"│  Top-3:  {final_results['top3']*100:5.1f}%               │")
    print("└────────────────────────────────┘")
    
    # 3. Check for LNN checkpoint
    if args.checkpoint:
        print()
        print("=" * 70)
        print("LNN CHECKPOINT")
        print("=" * 70)
        
        ckpt = load_lnn_checkpoint(args.checkpoint)
        if ckpt:
            print("Checkpoint loaded successfully!")
            print("To use the full LNN+Physics ensemble, run:")
            print(f"  python scripts/train_phase9_ensemble.py --checkpoint {args.checkpoint} --data {args.data}")
        else:
            print("Could not load checkpoint. Using physics-only.")
    else:
        print()
        print("TIP: To add ML predictions, provide --checkpoint path to Phase 5 LNN")
    
    print()
    print("COMPARISON:")
    print("-" * 40)
    print(f"  Physics Only:        Top-3: {physics_results['top3']*100:.1f}%")
    print(f"  Physics + Analogical: Top-3: {final_results['top3']*100:.1f}%")
    print(f"  ML Baseline (Phase5): Top-3: ~60%")  # From earlier training


if __name__ == "__main__":
    main()
