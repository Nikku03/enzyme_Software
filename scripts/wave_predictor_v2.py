#!/usr/bin/env python3
"""
WAVE METABOLISM PREDICTOR v2
============================

Rethinking from scratch.

The question isn't "where are the electrons?"
The question is "where do electrons WANT to go when pulled?"

CYP Fe=O is an electrophile. It creates a potential well.
Electrons flow toward it - but from where?

The answer: from wherever the electron cloud is most POLARIZABLE.

Physics:
  - Apply a test field at each point
  - Measure how much the TOTAL wavefunction distorts
  - High distortion = electrons easily pulled = reactive site

This is polarizability, not density.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import linalg

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("RDKit required")
    exit(1)


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

ELECTRONEGATIVITY = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
}

# Atomic polarizability (Å³) - THIS IS THE KEY
POLARIZABILITY = {
    1: 0.67,    # H
    6: 1.76,    # C
    7: 1.10,    # N
    8: 0.80,    # O
    9: 0.56,    # F
    16: 2.90,   # S
    17: 2.18,   # Cl
    35: 3.05,   # Br
}


# =============================================================================
# THE WAVE MODEL
# =============================================================================

def build_extended_hamiltonian(mol) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Hamiltonian that includes BOND orbitals, not just atom orbitals.
    
    Each bond is a place where electrons live.
    Atoms are just connection points.
    
    This gives us a richer basis to describe electron flow.
    """
    n_atoms = mol.GetNumAtoms()
    bonds = list(mol.GetBonds())
    n_bonds = len(bonds)
    
    # Total basis: atoms + bonds
    n_total = n_atoms + n_bonds
    
    H = np.zeros((n_total, n_total))
    
    # Atom-atom block: on-site energy from electronegativity
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        chi = ELECTRONEGATIVITY.get(atom.GetAtomicNum(), 2.5)
        H[i, i] = chi
    
    # Bond energies and atom-bond coupling
    for b_idx, bond in enumerate(bonds):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_idx = n_atoms + b_idx
        
        # Bond orbital energy (lower = more stable)
        bond_order = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            bond_order = 1.5
        
        # Bonding orbital is stabilized
        H[bond_idx, bond_idx] = -bond_order
        
        # Coupling: atoms connect to their bonds
        coupling = np.sqrt(bond_order)  # Stronger bonds = stronger coupling
        H[i, bond_idx] = H[bond_idx, i] = -coupling
        H[j, bond_idx] = H[bond_idx, j] = -coupling
    
    return H, np.array([b.GetBeginAtomIdx() for b in bonds])


def compute_polarizability_tensor(mol, H: np.ndarray) -> np.ndarray:
    """
    For each atom, compute how much the wavefunction changes
    when we apply a small field there.
    
    α_i = d⟨ψ|r_i|ψ⟩ / dE
    
    This is linear response theory applied correctly.
    """
    n_atoms = mol.GetNumAtoms()
    n_total = H.shape[0]
    
    # Solve unperturbed system
    E0, psi0 = linalg.eigh(H)
    
    # Number of "occupied" orbitals (half-filling for tight binding)
    n_occ = n_total // 2
    
    # Ground state density matrix
    rho0 = np.zeros((n_total, n_total))
    for k in range(n_occ):
        rho0 += np.outer(psi0[:, k], psi0[:, k])
    
    # Polarizability at each atom
    alpha = np.zeros(n_atoms)
    
    for i in range(n_atoms):
        # Perturbation: small potential at atom i
        dV = np.zeros(n_total)
        dV[i] = 0.01  # Small perturbation
        
        # Perturbed Hamiltonian
        H_pert = H + np.diag(dV)
        
        # Solve perturbed system
        E1, psi1 = linalg.eigh(H_pert)
        
        # New density matrix
        rho1 = np.zeros((n_total, n_total))
        for k in range(n_occ):
            rho1 += np.outer(psi1[:, k], psi1[:, k])
        
        # Change in density at atom i
        delta_rho = rho1[i, i] - rho0[i, i]
        
        # Polarizability = response / perturbation
        alpha[i] = abs(delta_rho) / 0.01
    
    return alpha


def compute_flow_susceptibility(mol, H: np.ndarray) -> np.ndarray:
    """
    Different approach: measure how easily electrons FLOW AWAY from each atom.
    
    For each atom i:
      1. Create a "sink" at that atom (electron-withdrawing perturbation)
      2. Measure total electron flow toward the sink
      3. Higher flow = electrons more easily extracted = more reactive
    
    This directly models what CYP does: it's an electron sink.
    """
    n_atoms = mol.GetNumAtoms()
    n_total = H.shape[0]
    
    # Solve unperturbed
    E0, psi0 = linalg.eigh(H)
    n_occ = n_total // 2
    
    # Ground state: total electron count at each atom
    ground_pop = np.zeros(n_atoms)
    for k in range(n_occ):
        for i in range(n_atoms):
            ground_pop[i] += abs(psi0[i, k])**2
    
    flow = np.zeros(n_atoms)
    
    for target in range(n_atoms):
        # Create sink at target atom (lower its energy = attract electrons)
        dV = np.zeros(n_total)
        dV[target] = -0.5  # Attractive perturbation
        
        H_sink = H + np.diag(dV)
        E1, psi1 = linalg.eigh(H_sink)
        
        # New population at target
        new_pop = 0.0
        for k in range(n_occ):
            new_pop += abs(psi1[target, k])**2
        
        # Flow = increase in population at sink
        flow[target] = new_pop - ground_pop[target]
    
    return flow


def compute_bond_participation(mol, H: np.ndarray) -> np.ndarray:
    """
    Which atoms participate most in the HOMO?
    
    The HOMO electrons are most easily removed.
    Where do they live?
    """
    n_atoms = mol.GetNumAtoms()
    n_total = H.shape[0]
    
    E, psi = linalg.eigh(H)
    n_occ = n_total // 2
    
    # HOMO is last occupied orbital
    if n_occ > 0:
        homo = psi[:, n_occ - 1]
    else:
        homo = psi[:, 0]
    
    # Participation of each atom in HOMO
    participation = np.zeros(n_atoms)
    for i in range(n_atoms):
        participation[i] = abs(homo[i])**2
    
    # Also include HOMO-1 and HOMO-2 for near-degeneracy
    for offset in [1, 2]:
        if n_occ - 1 - offset >= 0:
            orb = psi[:, n_occ - 1 - offset]
            weight = 0.5 ** offset
            for i in range(n_atoms):
                participation[i] += weight * abs(orb[i])**2
    
    return participation


def compute_reactivity(mol) -> Tuple[np.ndarray, Dict]:
    """
    Combine all reactivity measures into a single score.
    
    The philosophy:
      - Polarizability: how easily electrons move when pushed
      - Flow: how easily electrons are extracted
      - HOMO participation: where the "ready to go" electrons are
      - Atomic softness: intrinsic reactivity of atom type
    """
    n_atoms = mol.GetNumAtoms()
    
    # Build extended Hamiltonian
    H, bond_atoms = build_extended_hamiltonian(mol)
    
    # Compute all indices
    alpha = compute_polarizability_tensor(mol, H)
    flow = compute_flow_susceptibility(mol, H)
    homo_part = compute_bond_participation(mol, H)
    
    # Atomic softness (intrinsic)
    atomic_soft = np.zeros(n_atoms)
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        atomic_soft[i] = POLARIZABILITY.get(atom.GetAtomicNum(), 1.0)
    
    # Normalize each measure to [0, 1]
    def normalize(x):
        if x.max() - x.min() > 1e-10:
            return (x - x.min()) / (x.max() - x.min())
        return np.ones_like(x) * 0.5
    
    alpha = normalize(alpha)
    flow = normalize(flow)
    homo_part = normalize(homo_part)
    atomic_soft = normalize(atomic_soft)
    
    # Final reactivity score
    reactivity = np.zeros(n_atoms)
    
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        
        # Only carbons with H can undergo H-abstraction
        if atom.GetAtomicNum() != 6:
            reactivity[i] = -np.inf
            continue
        
        n_H = atom.GetTotalNumHs()
        if n_H == 0:
            reactivity[i] = -np.inf
            continue
        
        # Combine indices
        # Key insight: FLOW is most important - it directly measures extractability
        reactivity[i] = (
            0.40 * flow[i] +           # How easily electrons extracted
            0.30 * homo_part[i] +      # HOMO participation
            0.20 * alpha[i] +          # Polarizability
            0.10 * atomic_soft[i]      # Intrinsic softness
        )
        
        # Hydrogen count matters for abstraction
        reactivity[i] *= (1 + 0.1 * (n_H - 1))
    
    diagnostics = {
        'polarizability': alpha,
        'flow': flow,
        'homo_participation': homo_part,
        'atomic_softness': atomic_soft,
    }
    
    return reactivity, diagnostics


# =============================================================================
# PREDICTION
# =============================================================================

@dataclass
class WavePrediction:
    smiles: str
    scores: np.ndarray
    top1: int
    top3: List[int]
    diagnostics: Dict


def predict(smiles: str) -> Optional[WavePrediction]:
    """Predict site of metabolism."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    
    reactivity, diagnostics = compute_reactivity(mol)
    
    # Rank valid atoms
    valid_idx = [i for i in range(len(reactivity)) if reactivity[i] > -np.inf]
    if not valid_idx:
        return None
    
    ranked = sorted(valid_idx, key=lambda i: -reactivity[i])
    
    return WavePrediction(
        smiles=smiles,
        scores=reactivity,
        top1=ranked[0],
        top3=ranked[:3],
        diagnostics=diagnostics,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data_path: str):
    """Evaluate on dataset."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get('drugs', data)
    
    top1_correct = 0
    top3_correct = 0
    total = 0
    
    # Track by source
    by_source = {}
    
    print(f"\nEvaluating WAVE v2 on {len(drugs)} molecules...")
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
    print("WAVE v2 - POLARIZABILITY & FLOW MODEL")
    print("=" * 60)
    print(f"\nOVERALL: Top-1={top1_correct/total*100:.1f}%, Top-3={top3_correct/total*100:.1f}% (n={total})")
    
    print("\nBY SOURCE:")
    for src, stats in sorted(by_source.items(), key=lambda x: -x[1]['total']):
        if stats['total'] >= 5:
            t1 = stats['top1'] / stats['total'] * 100
            t3 = stats['top3'] / stats['total'] * 100
            print(f"  {src:20s}: Top-1={t1:5.1f}%, Top-3={t3:5.1f}% (n={stats['total']})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])
    else:
        # Demo
        test = ["CCO", "Cc1ccccc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]
        for s in test:
            p = predict(s)
            if p:
                print(f"{s}: top3={p.top3}, scores={p.scores[p.scores > -np.inf].round(3)}")
