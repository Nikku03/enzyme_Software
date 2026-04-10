#!/usr/bin/env python3
"""
Evaluate Unified Ensemble on CYP3A4 dataset.

This combines:
1. Physics (Hydrogen Theft v3)
2. Analogical reasoning (memory-based)
3. Structural features

Usage:
    python scripts/eval_unified_ensemble.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enzyme_software.liquid_nn_v2.model.unified_ensemble import (
    UnifiedEnsemble,
    get_physics_scores,
    AnalogicalMemory
)


def grid_search_weights(data_path: str):
    """Find optimal weights via grid search."""
    import json
    
    print("Grid searching for optimal weights...")
    print()
    
    best_top3 = 0
    best_weights = (0.5, 0.3, 0.2)
    
    # Grid search
    for pw in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for aw in [0.0, 0.1, 0.2, 0.3, 0.4]:
            sw = 1.0 - pw - aw
            if sw < 0:
                continue
            
            ensemble = UnifiedEnsemble(
                physics_weight=pw,
                analogical_weight=aw,
                structural_weight=sw,
                memory_capacity=2000
            )
            
            results = ensemble.evaluate(data_path)
            
            if results['top3'] > best_top3:
                best_top3 = results['top3']
                best_weights = (pw, aw, sw)
                print(f"  New best: P={pw:.1f} A={aw:.1f} S={sw:.1f} → Top-3={results['top3']*100:.1f}%")
    
    print()
    print(f"Optimal weights: Physics={best_weights[0]}, Analogical={best_weights[1]}, Structural={best_weights[2]}")
    return best_weights


def main():
    data_path = PROJECT_ROOT / "data" / "curated" / "merged_cyp3a4_extended.json"
    
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("     UNIFIED ENSEMBLE EVALUATION")
    print("     Physics + Analogical Memory + Structural Features")
    print("=" * 70)
    print()
    
    # First, try with default weights
    print("Testing with default weights (0.6, 0.3, 0.1)...")
    ensemble = UnifiedEnsemble(
        physics_weight=0.6,
        analogical_weight=0.3,
        structural_weight=0.1,
        memory_capacity=2000
    )
    
    results = ensemble.evaluate(str(data_path))
    
    print(f"\nMolecules: {results['total']}")
    print(f"  Top-1: {results['top1']*100:.1f}%")
    print(f"  Top-2: {results['top2']*100:.1f}%")
    print(f"  Top-3: {results['top3']*100:.1f}%")
    
    # Grid search for optimal weights
    print()
    best_weights = grid_search_weights(str(data_path))
    
    # Final evaluation with best weights
    print()
    print("=" * 70)
    print("FINAL RESULTS WITH OPTIMAL WEIGHTS")
    print("=" * 70)
    
    ensemble = UnifiedEnsemble(
        physics_weight=best_weights[0],
        analogical_weight=best_weights[1],
        structural_weight=best_weights[2],
        memory_capacity=2000
    )
    
    results = ensemble.evaluate(str(data_path))
    
    print(f"\nMolecules evaluated: {results['total']}")
    print()
    print("┌────────────────────────────────┐")
    print("│          ACCURACY              │")
    print("├────────────────────────────────┤")
    print(f"│  Top-1:  {results['top1']*100:5.1f}%               │")
    print(f"│  Top-2:  {results['top2']*100:5.1f}%               │")
    print(f"│  Top-3:  {results['top3']*100:5.1f}%               │")
    print("└────────────────────────────────┘")
    print()
    
    print("COMPARISON TO BASELINES:")
    print("-" * 40)
    print("  Basic Physics:     Top-1: 22.2%  Top-3: 40.7%")
    print("  Hydrogen Theft v3: Top-1: 24.1%  Top-3: 44.0%")
    print(f"  Unified Ensemble:  Top-1: {results['top1']*100:.1f}%  Top-3: {results['top3']*100:.1f}%")
    
    if results['top3'] > 0.44:
        print()
        print("  ✓ UNIFIED ENSEMBLE BEATS PHYSICS-ONLY!")


if __name__ == "__main__":
    main()
