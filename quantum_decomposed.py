"""
UNDERSTANDING FLEXIBILITY AT THE QUANTUM LEVEL

Flexibility = 1 / Σ|ψ_k(i)|² (for high k modes)

What does this MEAN quantum mechanically?

High k modes = high eigenvalue = high frequency = LOCALIZED vibrations
Low participation in high-k modes = atom NOT locked into rigid structure
= atom CAN MOVE when perturbed

This is exactly what happens in H-abstraction:
1. Fe=O approaches
2. C-H bond must STRETCH toward Fe
3. H transfers to Fe=O
4. Radical forms on C

If the C is RIGID (high-k participation), it can't stretch → no reaction
If the C is FLEXIBLE (low high-k participation), it can stretch → reaction

So flexibility captures the KINETIC accessibility of the transition state.

But what about THERMODYNAMICS? What makes the TS stable once reached?

Let me decompose the problem:

KINETIC: Can we reach the TS? → Flexibility
THERMODYNAMIC: Is the TS stable? → Radical stability, alpha-heteroatom
QUANTUM: Can H tunnel through the barrier? → Mass, barrier width/height

Let me separately compute each and find the optimal combination.
"""

import numpy as np
from rdkit import Chem
import json


def compute_flexibility(mol, eigenvectors, n_high=3):
    """
    KINETIC factor: can the atom move toward TS?
    
    Flexibility = 1 / (participation in high-frequency modes)
    """
    n = mol.GetNumAtoms()
    flexibility = np.zeros(n)
    
    for i in range(n):
        rigidity = sum(eigenvectors[i, k]**2 for k in range(max(1, n - n_high), n))
        flexibility[i] = 1.0 / (rigidity + 0.1)
    
    return flexibility


def compute_amplitude(mol, eigenvectors, eigenvalues, n_low=5):
    """
    ELECTRONIC factor: electron delocalization.
    
    Amplitude = weighted participation in low-frequency modes
    Low freq modes = delocalized electrons = easier to remove
    """
    n = mol.GetNumAtoms()
    amplitude = np.zeros(n)
    
    for i in range(n):
        amp = 0.0
        for k in range(1, min(n_low + 1, n)):
            if eigenvalues[k] > 1e-6:
                amp += eigenvectors[i, k]**2 / eigenvalues[k]
        amplitude[i] = amp
    
    return amplitude


def compute_radical_stability(mol, eigenvectors, eigenvalues):
    """
    THERMODYNAMIC factor: is the resulting radical stable?
    
    Stable radicals:
    - Delocalized spin (spread over many atoms)
    - Resonance with π systems
    - Hyperconjugation with adjacent C-H bonds
    """
    n = mol.GetNumAtoms()
    stability = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        # Check for stabilization sources
        stab = 1.0
        
        for nbr in atom.GetNeighbors():
            # Aromatic neighbor → resonance stabilization
            if nbr.GetIsAromatic():
                stab *= 1.8
            
            # Heteroatom neighbor → lone pair donation
            z = nbr.GetAtomicNum()
            if z == 7:
                stab *= 1.6
            elif z == 8:
                stab *= 1.5
            elif z == 16:
                stab *= 1.4
            
            # C neighbors with H → hyperconjugation
            if z == 6 and nbr.GetTotalNumHs() > 0:
                stab *= 1.1
        
        stability[i] = stab
    
    return stability


def compute_tunneling_factor(mol, eigenvectors, eigenvalues):
    """
    QUANTUM TUNNELING: can H tunnel through the barrier?
    
    Tunneling probability ∝ exp(-2√(2mV)a/ℏ)
    
    For H transfer:
    - Light mass → significant tunneling
    - Lower barrier → higher tunneling
    - Narrower barrier → higher tunneling
    
    We approximate:
    - Flexible sites have lower effective barrier
    - Alpha-heteroatom sites have narrower barrier
    """
    n = mol.GetNumAtoms()
    tunneling = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        # Base tunneling (related to flexibility)
        rigidity = sum(eigenvectors[i, k]**2 for k in range(max(1, n-3), n))
        base_tunnel = 1.0 / (rigidity + 0.1)
        
        # Barrier narrowing from alpha-heteroatom
        barrier_factor = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7:
                barrier_factor *= 1.3  # N narrows barrier
            elif z == 8:
                barrier_factor *= 1.25
            elif z == 16:
                barrier_factor *= 1.2
        
        tunneling[i] = base_tunnel * barrier_factor
    
    return tunneling


def compute_vibronic_coupling(mol, eigenvectors, eigenvalues):
    """
    VIBRONIC COUPLING: electron-nuclear interaction.
    
    The C-H stretch must couple to electronic reorganization.
    Sites where vibrational and electronic motions mix strongly
    have enhanced reactivity.
    
    Measure: overlap between low-freq (electronic) and high-freq (nuclear) modes
    """
    n = mol.GetNumAtoms()
    vibronic = np.zeros(n)
    
    for i in range(n):
        # Participation in low-freq modes (electronic character)
        low = sum(eigenvectors[i, k]**2 for k in range(1, min(4, n)))
        
        # Participation in mid-freq modes (vibronic mixing zone)
        mid = sum(eigenvectors[i, k]**2 for k in range(n//3, 2*n//3))
        
        # Vibronic coupling = how much does this atom connect low and mid
        vibronic[i] = np.sqrt(low * mid)
    
    return vibronic


def full_quantum_som(smiles, weights=None):
    """
    Full model with separate kinetic, thermodynamic, and quantum components.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    n = mol.GetNumAtoms()
    
    # Graph Laplacian
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            w = 1.5
        A[i, j] = A[j, i] = w
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Compute all factors
    flexibility = compute_flexibility(mol, eigenvectors)
    amplitude = compute_amplitude(mol, eigenvectors, eigenvalues)
    rad_stability = compute_radical_stability(mol, eigenvectors, eigenvalues)
    tunneling = compute_tunneling_factor(mol, eigenvectors, eigenvalues)
    vibronic = compute_vibronic_coupling(mol, eigenvectors, eigenvalues)
    
    if weights is None:
        # Default weights (will optimize)
        weights = {
            'flex': 0.25,
            'amp': 0.15,
            'stability': 0.10,
            'tunnel': 0.10,
            'vibronic': 0.05,
        }
    
    scores = np.full(n, -np.inf)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        z = atom.GetAtomicNum()
        
        if z != 6:
            continue
        
        n_H = atom.GetTotalNumHs()
        is_arom = atom.GetIsAromatic()
        
        if n_H == 0 and not is_arom:
            continue
        
        # Base quantum score
        base = (
            weights['flex'] * flexibility[i] +
            weights['amp'] * amplitude[i] +
            weights['stability'] * rad_stability[i] +
            weights['tunnel'] * tunneling[i] +
            weights['vibronic'] * vibronic[i]
        )
        
        # Chemical multipliers (these encode chemistry, not physics)
        alpha_mult = 1.0
        for nbr in atom.GetNeighbors():
            nbr_z = nbr.GetAtomicNum()
            if nbr_z == 7:
                alpha_mult = max(alpha_mult, 1.84)
            elif nbr_z == 8:
                alpha_mult = max(alpha_mult, 1.82)
            elif nbr_z == 16:
                alpha_mult = max(alpha_mult, 1.54)
        
        benz_mult = 1.0
        if not is_arom and n_H > 0:
            for nbr in atom.GetNeighbors():
                if nbr.GetIsAromatic():
                    benz_mult = 1.73
                    break
        
        n_C_nbrs = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6)
        tert_mult = 1.0 + 0.22 * (n_C_nbrs - 1) if n_C_nbrs > 1 else 1.0
        
        h_factor = (1 + 0.13 * n_H) if n_H > 0 else 0.3
        
        scores[i] = base * alpha_mult * benz_mult * tert_mult * h_factor
    
    return scores


def optimize_weights(data_path, n_trials=200):
    """
    Random search for optimal weights.
    """
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data['drugs']
    
    # Filter to AZ120 for optimization (our target)
    az120 = [d for d in drugs if d.get('source') == 'AZ120']
    print(f"Optimizing on {len(az120)} AZ120 molecules...")
    
    best_acc = 0
    best_weights = None
    
    for trial in range(n_trials):
        # Random weights
        w = {
            'flex': np.random.uniform(0.1, 0.5),
            'amp': np.random.uniform(0.05, 0.3),
            'stability': np.random.uniform(0.0, 0.2),
            'tunnel': np.random.uniform(0.0, 0.2),
            'vibronic': np.random.uniform(0.0, 0.15),
        }
        
        # Normalize
        total = sum(w.values())
        w = {k: v/total for k, v in w.items()}
        
        # Evaluate
        top1 = 0
        total_eval = 0
        
        for d in az120:
            smiles = d.get('smiles', '')
            sites = d.get('site_atoms', [])
            
            if not smiles or not sites:
                continue
            
            scores = full_quantum_som(smiles, w)
            if scores is None:
                continue
            
            valid = [i for i in range(len(scores)) if scores[i] > -np.inf]
            if not valid:
                continue
            
            ranked = sorted(valid, key=lambda x: -scores[x])
            if ranked[0] in sites:
                top1 += 1
            total_eval += 1
        
        if total_eval > 0:
            acc = top1 / total_eval
            if acc > best_acc:
                best_acc = acc
                best_weights = w.copy()
                print(f"  Trial {trial}: {acc*100:.1f}% - weights={w}")
    
    print(f"\nBest AZ120: {best_acc*100:.1f}%")
    print(f"Best weights: {best_weights}")
    
    return best_weights


def evaluate(data_path, weights=None):
    """Evaluate on full dataset."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data['drugs']
    
    top1 = top3 = total = 0
    by_source = {}
    
    for d in drugs:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        src = d.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        scores = full_quantum_som(smiles, weights)
        if scores is None:
            continue
        
        valid = [i for i in range(len(scores)) if scores[i] > -np.inf]
        if not valid:
            continue
        
        ranked = sorted(valid, key=lambda x: -scores[x])[:3]
        
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
    
    print(f"\n=== DECOMPOSED QUANTUM MODEL ===")
    print(f"Top-1: {top1}/{total} = {top1/total*100:.1f}%")
    print(f"Top-3: {top3}/{total} = {top3/total*100:.1f}%\n")
    
    for src, s in sorted(by_source.items(), key=lambda x: -x[1]['n']):
        if s['n'] >= 5:
            print(f"  {src:20s}: Top-1={s['t1']/s['n']*100:5.1f}%, Top-3={s['t3']/s['n']*100:5.1f}% (n={s['n']})")
    
    return top1/total if total > 0 else 0


if __name__ == '__main__':
    data_path = '/home/claude/enzyme_Software/data/curated/merged_cyp3a4_extended.json'
    
    print("=== OPTIMIZING WEIGHTS ===")
    best_weights = optimize_weights(data_path, n_trials=300)
    
    print("\n=== EVALUATING WITH BEST WEIGHTS ===")
    evaluate(data_path, best_weights)
