#!/usr/bin/env python3
"""
WAVE INTERFERENCE MODEL
=======================

Forget everything about orbitals, HOMO, Fukui.

A molecule is a standing wave pattern.
An enzyme is an incoming wave.
Reaction happens where they DESTRUCTIVELY INTERFERE.

The physics:
  ψ_molecule = standing wave pattern on molecular graph
  ψ_enzyme = incoming perturbation wave
  
  Reaction site = argmax |ψ_molecule + ψ_enzyme|² - |ψ_molecule|²
                = where adding the enzyme wave changes the pattern most

But actually, for bond breaking, we want where:
  |ψ_molecule + ψ_enzyme|² is MINIMIZED
  = destructive interference
  = wave cancellation
  = bond breaking
"""

import numpy as np
from rdkit import Chem
import json


def compute_standing_wave(mol):
    """
    Compute the natural standing wave pattern of the molecule.
    
    This is NOT about electrons. It's about the natural resonant modes
    of the molecular structure itself.
    
    Like a vibrating drum - what are the fundamental modes?
    """
    n = mol.GetNumAtoms()
    
    # The molecule's "shape" is encoded in its connectivity
    # Build the adjacency matrix
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # All bonds are connections - weight by bond order
        w = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            w = 1.5
        A[i, j] = A[j, i] = w
    
    # The Laplacian gives us the wave equation on this graph
    # L = D - A, where D is degree matrix
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Solve the wave equation: L ψ = λ ψ
    # Eigenvalues = frequencies², Eigenvectors = mode shapes
    eigenvalues, modes = np.linalg.eigh(L)
    
    # The GROUND STATE is the constant mode (eigenvalue = 0)
    # The FIRST EXCITED STATES are the fundamental vibrations
    
    # Return the mode shapes (eigenvectors)
    return eigenvalues, modes


def simulate_enzyme_approach(mol, modes, target_atom):
    """
    Simulate what happens when the enzyme wave approaches an atom.
    
    The enzyme creates a localized disturbance - like dropping a pebble
    in a pond where there's already a standing wave pattern.
    
    The interference pattern tells us how the molecule responds.
    """
    n = mol.GetNumAtoms()
    
    # Enzyme wave: localized Gaussian centered at target atom
    # Decays with graph distance
    enzyme_wave = np.zeros(n)
    
    # Simple model: enzyme affects target and immediate neighbors
    enzyme_wave[target_atom] = 1.0
    
    atom = mol.GetAtomWithIdx(target_atom)
    for neighbor in atom.GetNeighbors():
        j = neighbor.GetIdx()
        enzyme_wave[j] = 0.5  # Neighbors feel half the effect
        
        # Second neighbors feel less
        for nn in neighbor.GetNeighbors():
            k = nn.GetIdx()
            if k != target_atom:
                enzyme_wave[k] = max(enzyme_wave[k], 0.25)
    
    # Now: how does this enzyme wave interfere with the molecular modes?
    
    # Project enzyme wave onto the molecular modes
    # This tells us which modes the enzyme excites
    mode_excitation = np.zeros(n)
    for k in range(n):
        mode_k = modes[:, k]
        # Overlap between enzyme wave and mode k
        overlap = np.dot(enzyme_wave, mode_k)
        mode_excitation[k] = overlap ** 2
    
    return enzyme_wave, mode_excitation


def compute_interference_score(mol):
    """
    For each atom, compute how much interference occurs when
    the enzyme wave meets the molecular wave there.
    
    High interference = unstable = reactive
    """
    n = mol.GetNumAtoms()
    eigenvalues, modes = compute_standing_wave(mol)
    
    scores = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        # Only consider carbons (CYP targets carbons)
        if atom.GetAtomicNum() != 6:
            scores[i] = -np.inf
            continue
        
        # Simulate enzyme approaching this atom
        enzyme_wave, mode_excitation = simulate_enzyme_approach(mol, modes, i)
        
        # KEY INSIGHT: The enzyme excites high-frequency modes at reactive sites
        # High-frequency = high eigenvalue = localized disturbance = bond breaking
        
        # Weighted sum: higher modes (higher eigenvalues) = more reactive
        # But we weight by how much each mode is excited
        interference_score = 0.0
        for k in range(1, n):  # Skip k=0 (constant mode)
            freq_squared = eigenvalues[k]
            excitation = mode_excitation[k]
            
            # Higher frequency modes being excited = more localized response = more reactive
            interference_score += freq_squared * excitation
        
        # Also consider: participation of this atom in low-frequency modes
        # Low frequency = delocalized = stable radical
        delocalization = 0.0
        for k in range(1, min(4, n)):  # First few non-trivial modes
            delocalization += np.abs(modes[i, k])
        
        # Combine: high interference + high delocalization = reactive
        scores[i] = interference_score * (1 + delocalization)
        
        # Require hydrogen for H-abstraction (main CYP pathway)
        n_H = atom.GetTotalNumHs()
        if n_H == 0 and not atom.GetIsAromatic():
            scores[i] = -np.inf
        elif n_H > 0:
            scores[i] *= (1 + 0.1 * n_H)
    
    return scores


def compute_wave_disruption(mol):
    """
    Alternative approach: measure how much the standing wave pattern
    would be DISRUPTED if we removed an atom.
    
    High disruption = that atom is critical to the wave pattern
    = removing it (via oxidation) is energetically accessible
    
    Wait, that's backwards. Let me think again...
    
    Actually: atoms that are NOT critical to the wave pattern
    can be modified without disrupting the whole molecule.
    These are the "soft" spots.
    
    But also: atoms where the wave has high AMPLITUDE are where
    energy is concentrated = where reactions happen.
    
    The answer is the PRODUCT: 
    - high amplitude (energy available)
    - but not critical (won't break everything)
    """
    n = mol.GetNumAtoms()
    eigenvalues, modes = compute_standing_wave(mol)
    
    # Amplitude at each atom (sum over modes, weighted by importance)
    amplitude = np.zeros(n)
    for k in range(1, n):
        # Weight by inverse frequency (low freq modes are global, high freq are local)
        weight = 1.0 / (eigenvalues[k] + 0.1)
        amplitude += weight * modes[:, k] ** 2
    
    # Criticality: how much would removing this atom change the eigenvalues?
    # (This is computationally expensive, so approximate)
    # Approximation: atoms with high degree are more critical
    criticality = np.zeros(n)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        criticality[i] = len(list(atom.GetNeighbors()))
    
    criticality = criticality / (criticality.max() + 1e-10)
    
    scores = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        if atom.GetAtomicNum() != 6:
            scores[i] = -np.inf
            continue
        
        n_H = atom.GetTotalNumHs()
        
        # High amplitude, low criticality = reactive
        scores[i] = amplitude[i] * (1 - 0.3 * criticality[i])
        
        # Must have H or be aromatic
        if n_H == 0 and not atom.GetIsAromatic():
            scores[i] = -np.inf
        elif n_H > 0:
            scores[i] *= (1 + 0.15 * n_H)
        
        # Boost for being next to heteroatoms (alpha position)
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() in [7, 8, 16]:
                scores[i] *= 1.3
                break
    
    return scores


def predict(smiles, method='interference'):
    """Predict SoM using wave interference."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    if method == 'interference':
        scores = compute_interference_score(mol)
    else:
        scores = compute_wave_disruption(mol)
    
    valid = [i for i in range(len(scores)) if scores[i] > -np.inf]
    if not valid:
        return None, None
    
    ranked = sorted(valid, key=lambda x: -scores[x])
    return ranked[:3], scores


def evaluate(data_path, method='interference'):
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get('drugs', data)
    
    top1 = top3 = total = 0
    by_source = {}
    
    for d in drugs:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        src = d.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        ranked, _ = predict(smiles, method)
        if ranked is None:
            continue
        
        if src not in by_source:
            by_source[src] = {'t1': 0, 't3': 0, 'n': 0}
        by_source[src]['n'] += 1
        
        if ranked[0] in sites:
            top1 += 1
            by_source[src]['t1'] += 1
        if any(r in sites for r in ranked):
            top3 += 1
            by_source[src]['t3'] += 1
        total += 1
    
    print(f"\nWAVE {method.upper()}: Top-1={top1/total*100:.1f}%, Top-3={top3/total*100:.1f}%")
    print("\nBy source:")
    for src, s in sorted(by_source.items(), key=lambda x: -x[1]['n']):
        if s['n'] >= 5:
            print(f"  {src:20s}: Top-1={s['t1']/s['n']*100:5.1f}% (n={s['n']})")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/curated/merged_cyp3a4_extended.json"
    
    print("=" * 60)
    print("WAVE INTERFERENCE MODEL")
    print("=" * 60)
    evaluate(path, 'interference')
    
    print("\n" + "=" * 60)
    print("WAVE DISRUPTION MODEL")  
    print("=" * 60)
    evaluate(path, 'disruption')
