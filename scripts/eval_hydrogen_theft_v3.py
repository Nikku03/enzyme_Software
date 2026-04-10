#!/usr/bin/env python3
"""
Evaluate Hydrogen Theft Theory v3 on the extended dataset.

Usage:
    python scripts/eval_hydrogen_theft_v3.py
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enzyme_software.liquid_nn_v2.model.hydrogen_theft_v3 import (
    HydrogenTheftScorer,
    evaluate_on_dataset
)


def main():
    data_path = PROJECT_ROOT / "data" / "curated" / "merged_cyp3a4_extended.json"
    
    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print("Make sure you've run the merge script first.")
        sys.exit(1)
    
    print()
    print("═" * 70)
    print("       HYDROGEN THEFT THEORY v3: The One Equation Model")
    print("═" * 70)
    print()
    print("  Core Insight: CYP metabolism = H-atom abstraction")
    print("  The site with the WEAKEST C-H bond wins.")
    print()
    print("  Formula: Score = 1 / BDE(C-H)")
    print()
    print("  BDE Values (kcal/mol):")
    print("    α-Nitrogen:  79-84  (weakest → most reactive)")
    print("    α-Oxygen:    83-86")
    print("    Benzylic:    85-90")
    print("    Allylic:     83-88")
    print("    Tertiary:    96")
    print("    Secondary:   98.5")
    print("    Primary:     101")
    print("    Aromatic:    113    (strongest → least reactive)")
    print()
    print("═" * 70)
    print()
    
    scorer = HydrogenTheftScorer(use_accessibility=True)
    results = evaluate_on_dataset(str(data_path), scorer)
    
    print(f"Dataset: {data_path.name}")
    print(f"Molecules evaluated: {results['total']}")
    print()
    print("┌────────────────────────────────┐")
    print("│          ACCURACY              │")
    print("├────────────────────────────────┤")
    print(f"│  Top-1:  {results['top1']*100:5.1f}%               │")
    print(f"│  Top-2:  {results['top2']*100:5.1f}%               │")
    print(f"│  Top-3:  {results['top3']*100:5.1f}%               │")
    print("└────────────────────────────────┘")
    print()
    
    print("CORRECT predictions by reaction type:")
    for t, c in sorted(results['by_pred_type']['correct'].items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print()
    
    print("WRONG predictions by reaction type:")
    for t, c in sorted(results['by_pred_type']['wrong'].items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print()
    
    # Compare with basic scorer
    print("=" * 70)
    print("COMPARISON WITH BASIC PHYSICS SCORER")
    print("=" * 70)
    print()
    print("  Basic Physics:    Top-1: 22.2%   Top-3: 40.7%")
    print(f"  Hydrogen Theft:   Top-1: {results['top1']*100:.1f}%   Top-3: {results['top3']*100:.1f}%")
    print()
    if results['top3'] > 0.407:
        print("  ✓ Hydrogen Theft BEATS basic physics!")
    else:
        print("  ✗ Need more tuning...")


if __name__ == "__main__":
    main()
