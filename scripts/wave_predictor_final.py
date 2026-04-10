#!/usr/bin/env python3
"""
WAVE METABOLISM PREDICTOR - Final Version
==========================================

A first-principles approach to site-of-metabolism prediction.

Core insight: Reactivity = Amplitude × Flexibility

- Amplitude: participation in low-frequency vibrational modes
  (where energy concentrates)
- Flexibility: inverse participation in high-frequency modes  
  (how easily the wave pattern disrupts)

This is pure wave mechanics on molecular graphs. No ML. No training.

Results on CYP3A4 dataset (869 molecules):
- Overall: ~21% Top-1, ~43% Top-3
- AZ120: 55% Top-1 (mostly alpha-N/O sites)
- DrugBank: 36% Top-1
- Zaretzki: 15% Top-1 (many aromatic sites)

Baseline comparison:
- BDE physics: 24% Top-1
- Random guess: ~7% Top-1

Author: Naresh Chhillar, 2026
Physics-only approach, no learned parameters
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit required: pip install rdkit")


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

ELECTRONEGATIVITY = {
    1: 2.20,   # H
    6: 2.55,   # C
    7: 3.04,   # N
    8: 3.44,   # O
    9: 3.98,   # F
    15: 2.19,  # P
    16: 2.58,  # S
    17: 3.16,  # Cl
    35: 2.96,  # Br
    53: 2.66,  # I
}


# =============================================================================
# WAVE MECHANICS
# =============================================================================

def build_molecular_wave_operator(mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the wave operator (Laplacian) for the molecular graph.
    
    The Laplacian encodes how waves propagate through the molecule.
    Its eigenvectors are the natural vibration modes.
    
    Returns:
        L: Graph Laplacian
        A: Adjacency matrix
        eigenvalues, eigenvectors: Spectral decomposition
    """
    n = mol.GetNumAtoms()
    
    # Build weighted adjacency matrix
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Bond order affects wave coupling strength
        weight = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            weight = 1.5  # Aromatic bonds have intermediate character
        
        A[i, j] = A[j, i] = weight
    
    # Degree matrix
    D = np.diag(A.sum(axis=1))
    
    # Graph Laplacian
    L = D - A
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    return L, eigenvalues, eigenvectors


def compute_amplitude(eigenvectors: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute wave amplitude at each atom.
    
    Amplitude measures participation in low-frequency (collective) modes.
    High amplitude = energy concentrated here = reactive.
    """
    n = eigenvectors.shape[0]
    n_modes = min(n, 6)  # First few non-trivial modes
    
    amplitude = np.zeros(n)
    for k in range(1, n_modes):  # Skip k=0 (constant mode)
        # Weight by inverse frequency (low freq = more important)
        weight = 1.0 / (eigenvalues[k] + 0.1)
        amplitude += weight * eigenvectors[:, k] ** 2
    
    # Normalize
    if amplitude.max() > 0:
        amplitude = amplitude / amplitude.max()
    
    return amplitude


def compute_flexibility(eigenvectors: np.ndarray) -> np.ndarray:
    """
    Compute flexibility at each atom.
    
    Flexibility = inverse of rigidity.
    Rigidity = participation in high-frequency (localized) modes.
    
    High flexibility = wave pattern easily disrupted = reactive.
    """
    n = eigenvectors.shape[0]
    n_high = min(3, n - 1)
    
    # Rigidity = participation in top N modes
    rigidity = np.zeros(n)
    for k in range(max(1, n - n_high), n):
        rigidity += eigenvectors[:, k] ** 2
    
    # Flexibility = inverse rigidity
    flexibility = 1.0 / (rigidity + 0.1)
    
    # Normalize
    if flexibility.max() > 0:
        flexibility = flexibility / flexibility.max()
    
    return flexibility


def compute_reactivity_score(mol) -> np.ndarray:
    """
    Compute reactivity score for each atom.
    
    Core formula: Reactivity = Amplitude × Flexibility
    
    Then apply chemical constraints:
    - Only carbons (CYP targets C-H and aromatic C)
    - Boost for alpha positions (next to N, O, S)
    - Boost for benzylic positions
    """
    n = mol.GetNumAtoms()
    
    # Wave mechanics
    L, eigenvalues, eigenvectors = build_molecular_wave_operator(mol)
    amplitude = compute_amplitude(eigenvectors, eigenvalues)
    flexibility = compute_flexibility(eigenvectors)
    
    # Base score
    scores = amplitude * flexibility
    
    # Apply chemical knowledge
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        # Only consider carbons
        if atom.GetAtomicNum() != 6:
            scores[i] = -np.inf
            continue
        
        n_H = atom.GetTotalNumHs()
        is_arom = atom.GetIsAromatic()
        
        # H-abstraction requires hydrogen
        if n_H == 0:
            if is_arom:
                # Aromatic oxidation possible but less likely
                scores[i] *= 0.5
            else:
                # No H, not aromatic = very unlikely
                scores[i] = -np.inf
                continue
        else:
            # More H = more chances for abstraction
            scores[i] *= (1 + 0.12 * n_H)
        
        # Alpha positions are activated (electronic effect)
        for neighbor in atom.GetNeighbors():
            z = neighbor.GetAtomicNum()
            if z == 7:  # Nitrogen
                scores[i] *= 1.5
                break
            elif z == 8:  # Oxygen
                scores[i] *= 1.35
                break
            elif z == 16:  # Sulfur
                scores[i] *= 1.4
                break
        
        # Benzylic positions are activated (resonance stabilization)
        if not is_arom:
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIsAromatic():
                    scores[i] *= 1.25
                    break
    
    return scores


# =============================================================================
# PREDICTION INTERFACE
# =============================================================================

@dataclass
class WavePrediction:
    """Prediction result for a single molecule."""
    smiles: str
    scores: np.ndarray
    top1: int
    top3: List[int]
    amplitude: np.ndarray
    flexibility: np.ndarray


def predict(smiles: str) -> Optional[WavePrediction]:
    """
    Predict site of metabolism for a SMILES string.
    
    Returns:
        WavePrediction with ranked atoms and scores
        None if molecule cannot be parsed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    
    # Compute scores
    scores = compute_reactivity_score(mol)
    
    # Get amplitude and flexibility for diagnostics
    L, eigenvalues, eigenvectors = build_molecular_wave_operator(mol)
    amplitude = compute_amplitude(eigenvectors, eigenvalues)
    flexibility = compute_flexibility(eigenvectors)
    
    # Rank valid atoms
    valid = [i for i in range(len(scores)) if scores[i] > -np.inf]
    if not valid:
        return None
    
    ranked = sorted(valid, key=lambda x: -scores[x])
    
    return WavePrediction(
        smiles=smiles,
        scores=scores,
        top1=ranked[0],
        top3=ranked[:3],
        amplitude=amplitude,
        flexibility=flexibility,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data_path: str) -> Dict[str, float]:
    """
    Evaluate on a JSON dataset.
    
    Expected format:
        {"drugs": [{"smiles": "...", "site_atoms": [0, 1], "source": "..."}, ...]}
    """
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get('drugs', data)
    
    top1_correct = 0
    top3_correct = 0
    total = 0
    by_source = {}
    
    print(f"\nEvaluating WAVE PREDICTOR on {len(drugs)} molecules...")
    print("-" * 60)
    
    for i, entry in enumerate(drugs):
        smiles = entry.get('smiles', '')
        sites = entry.get('site_atoms', [])
        source = entry.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        pred = predict(smiles)
        if pred is None:
            continue
        
        # Track by source
        if source not in by_source:
            by_source[source] = {'top1': 0, 'top3': 0, 'total': 0}
        
        by_source[source]['total'] += 1
        
        if pred.top1 in sites:
            top1_correct += 1
            by_source[source]['top1'] += 1
        
        if any(p in sites for p in pred.top3):
            top3_correct += 1
            by_source[source]['top3'] += 1
        
        total += 1
        
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(drugs)}: Top-1={top1_correct/total*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("WAVE METABOLISM PREDICTOR - RESULTS")
    print("=" * 60)
    print(f"\nOVERALL: Top-1={top1_correct/total*100:.1f}%, Top-3={top3_correct/total*100:.1f}% (n={total})")
    
    print("\nBY SOURCE:")
    for src, stats in sorted(by_source.items(), key=lambda x: -x[1]['total']):
        if stats['total'] >= 5:
            t1 = stats['top1'] / stats['total'] * 100
            t3 = stats['top3'] / stats['total'] * 100
            print(f"  {src:20s}: Top-1={t1:5.1f}%, Top-3={t3:5.1f}% (n={stats['total']})")
    
    return {
        'top1': top1_correct / total,
        'top3': top3_correct / total,
        'total': total,
        'by_source': by_source,
    }


def demo():
    """Demo on a few example molecules."""
    examples = [
        ("CCO", "Ethanol"),
        ("Cc1ccccc1", "Toluene"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
    ]
    
    print("\n" + "=" * 60)
    print("WAVE METABOLISM PREDICTOR - DEMO")
    print("=" * 60)
    
    for smiles, name in examples:
        pred = predict(smiles)
        if pred is None:
            print(f"\n{name}: Failed to parse")
            continue
        
        print(f"\n{name} ({smiles})")
        print(f"  Top-1 prediction: atom {pred.top1}")
        print(f"  Top-3 predictions: {pred.top3}")
        
        # Show scores for valid atoms
        valid_scores = [(i, pred.scores[i]) for i in range(len(pred.scores)) 
                       if pred.scores[i] > -np.inf]
        valid_scores.sort(key=lambda x: -x[1])
        print(f"  Scores: {[(i, round(s, 3)) for i, s in valid_scores[:5]]}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])
    else:
        demo()
        print("\n\nUsage: python wave_predictor_final.py <data.json>")
