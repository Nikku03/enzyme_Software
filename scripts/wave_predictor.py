#!/usr/bin/env python3
"""
WAVE METABOLISM PREDICTOR
=========================

First principles approach to site-of-metabolism prediction.

No neural networks. No learned parameters. Just wave mechanics.

The idea:
  1. Molecule = graph
  2. Electrons = waves on that graph
  3. Enzyme = perturbation
  4. Reactivity = how easily the wave is disturbed

Physics:
  H = L + V  (Hamiltonian = Laplacian + Potential)
  H ψₙ = Eₙ ψₙ  (Eigenstates = orbitals)
  χᵢ = response to perturbation at atom i
  
  Highest χ = metabolism site

Author: Built from first principles
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
except ImportError:
    print("RDKit required: pip install rdkit")
    exit(1)


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Pauling electronegativity
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

# Covalent radii (Angstroms) - for distance effects
COVALENT_RADIUS = {
    1: 0.31,   # H
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    9: 0.57,   # F
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
    35: 1.20,  # Br
    53: 1.39,  # I
}

# Ionization potential (eV) - how easy to remove electron
IONIZATION_POTENTIAL = {
    1: 13.6,   # H
    6: 11.3,   # C
    7: 14.5,   # N
    8: 13.6,   # O
    9: 17.4,   # F
    16: 10.4,  # S
    17: 13.0,  # Cl
}


# =============================================================================
# STEP 1: BUILD HAMILTONIAN
# =============================================================================

def build_hamiltonian(mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the molecular Hamiltonian on the graph.
    
    H = -t * A + V
    
    Where:
      A = adjacency matrix (hopping between bonded atoms)
      V = diagonal potential (electronegativity)
      t = hopping strength (depends on bond type)
    
    Returns:
      H: Hamiltonian matrix
      A: Adjacency matrix  
      V: Potential vector
    """
    n = mol.GetNumAtoms()
    
    # Adjacency matrix with bond-type weighting
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Bond order affects hopping strength
        bond_type = bond.GetBondTypeAsDouble()
        
        # Aromatic bonds get intermediate weight
        if bond.GetIsAromatic():
            weight = 1.5
        else:
            weight = bond_type  # 1.0, 2.0, 3.0 for single, double, triple
        
        A[i, j] = weight
        A[j, i] = weight
    
    # Degree matrix
    D = np.diag(A.sum(axis=1))
    
    # Graph Laplacian (encodes kinetic energy / connectivity)
    L = D - A
    
    # Potential energy (electronegativity + local environment)
    V = np.zeros(n)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        atomic_num = atom.GetAtomicNum()
        
        # Base electronegativity
        chi = ELECTRONEGATIVITY.get(atomic_num, 2.5)
        
        # Adjust for charge
        charge = atom.GetFormalCharge()
        chi += charge * 0.5  # Positive charge = harder to remove electron
        
        # Adjust for aromaticity (delocalized = more stable)
        if atom.GetIsAromatic():
            chi -= 0.3
        
        # Adjust for hybridization
        hyb = atom.GetHybridization()
        if hyb == Chem.HybridizationType.SP:
            chi += 0.3  # More s-character = tighter electrons
        elif hyb == Chem.HybridizationType.SP2:
            chi += 0.1
        
        V[i] = chi
    
    # Normalize potential to similar scale as Laplacian
    V = V - V.mean()  # Center
    V = V / (V.std() + 1e-8)  # Normalize
    
    # Hamiltonian: kinetic (Laplacian) + potential
    # The hopping term is -A (negative = bonding lowers energy)
    t = 1.0  # hopping strength
    H = t * L + np.diag(V)
    
    return H, A, V


# =============================================================================
# STEP 2: SOLVE EIGENSTATES
# =============================================================================

def solve_eigenstates(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the eigenvalue problem H ψ = E ψ.
    
    Returns:
      energies: eigenvalues (sorted ascending)
      states: eigenvectors (columns are states)
    """
    energies, states = np.linalg.eigh(H)
    
    # Already sorted by numpy
    return energies, states


def get_electron_density(states: np.ndarray, n_electrons: int) -> np.ndarray:
    """
    Compute electron density at each atom.
    
    For graph tight-binding model:
    - We only have n_atoms orbitals (one per atom)
    - Occupy the lowest energy ones
    """
    n_atoms = states.shape[0]
    n_orbitals = states.shape[1]
    
    # For graph model, use half the orbitals as "occupied"
    # This corresponds to half-filling typical of organic molecules
    n_occupied = min(n_orbitals // 2 + 1, n_orbitals)
    
    # Sum |ψ|² over occupied orbitals
    density = np.zeros(n_atoms)
    for n in range(n_occupied):
        psi = states[:, n]
        density += np.abs(psi)**2
    
    return density


# =============================================================================
# STEP 3: PERTURBATION MODEL
# =============================================================================

def build_perturbation(mol, atom_idx: int, A: np.ndarray) -> np.ndarray:
    """
    Model the CYP enzyme approaching atom i.
    
    CYP Fe=O is an electrophile - it wants to extract electrons.
    
    The perturbation:
      - Strong electron-withdrawing potential at atom i
      - Decays to neighbors (the enzyme affects local environment)
    
    P_j = strength * exp(-distance(i,j) / decay_length)
    """
    n = mol.GetNumAtoms()
    
    # Compute graph distances from atom i
    distances = compute_graph_distances(A, atom_idx)
    
    # Perturbation strength
    strength = 1.0
    decay_length = 2.0  # Affects ~2 bonds away
    
    P = np.zeros(n)
    for j in range(n):
        if distances[j] < np.inf:
            P[j] = strength * np.exp(-distances[j] / decay_length)
    
    return P


def compute_graph_distances(A: np.ndarray, source: int) -> np.ndarray:
    """BFS to compute shortest path distances on graph."""
    n = A.shape[0]
    distances = np.full(n, np.inf)
    distances[source] = 0
    
    queue = [source]
    while queue:
        current = queue.pop(0)
        for neighbor in range(n):
            if A[current, neighbor] > 0 and distances[neighbor] == np.inf:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    
    return distances


# =============================================================================
# STEP 4: COMPUTE RESPONSE (Linear Response Theory)
# =============================================================================

def compute_susceptibility(
    H: np.ndarray,
    energies: np.ndarray,
    states: np.ndarray,
    perturbation: np.ndarray,
    n_atoms: int,
) -> float:
    """
    Compute the linear response (susceptibility) to a perturbation.
    
    χ = Σ_{n,m} |⟨ψₙ|P|ψₘ⟩|² / (Eₘ - Eₙ)
    
    For graph model: occupied = lower half, unoccupied = upper half
    """
    n_orbitals = len(energies)
    n_occupied = n_orbitals // 2 + 1
    n_occupied = min(n_occupied, n_orbitals - 1)  # Need at least one unoccupied
    
    # Build perturbation operator
    P = np.diag(perturbation)
    
    susceptibility = 0.0
    
    # Sum over occupied → unoccupied transitions
    for n in range(n_occupied):
        psi_n = states[:, n]
        E_n = energies[n]
        
        for m in range(n_occupied, n_orbitals):
            psi_m = states[:, m]
            E_m = energies[m]
            
            # Energy denominator (avoid division by zero)
            dE = E_m - E_n
            if abs(dE) < 1e-10:
                continue
            
            # Transition matrix element ⟨ψₙ|P|ψₘ⟩
            matrix_element = np.dot(psi_n, P @ psi_m)
            
            # Contribution to susceptibility
            susceptibility += np.abs(matrix_element)**2 / dE
    
    return susceptibility


def compute_fukui_index(
    states: np.ndarray,
    energies: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """
    Compute Fukui f⁺ index (susceptibility to electrophilic attack).
    
    f⁺(i) = |ψ_HOMO(i)|²
    
    For graph model: HOMO is at index n_orbitals//2
    """
    n_orbitals = states.shape[1]
    
    # HOMO index for graph model (half-filling)
    homo_idx = min(n_orbitals // 2, n_orbitals - 1)
    
    psi_homo = states[:, homo_idx]
    
    # Fukui f+ = HOMO density
    fukui = np.abs(psi_homo)**2
    
    # Add contribution from nearby orbitals
    for k in range(max(0, homo_idx - 2), homo_idx):
        psi_k = states[:, k]
        weight = 0.3 * np.exp(-(homo_idx - k))
        fukui += weight * np.abs(psi_k)**2
    
    return fukui


def compute_local_softness(
    mol,
    states: np.ndarray,
    energies: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """
    Compute local softness (from conceptual DFT).
    
    Softness = fukui * global_softness
    Global softness ∝ 1 / HOMO-LUMO gap
    """
    n_orbitals = len(energies)
    
    # HOMO-LUMO indices
    homo_idx = min(n_orbitals // 2, n_orbitals - 1)
    lumo_idx = min(homo_idx + 1, n_orbitals - 1)
    
    # HOMO-LUMO gap
    gap = abs(energies[lumo_idx] - energies[homo_idx])
    
    # Global softness
    global_softness = 1.0 / (gap + 0.1)
    
    # Fukui index
    fukui = compute_fukui_index(states, energies, n_atoms)
    
    # Local softness
    softness = fukui * global_softness
    
    return softness


# =============================================================================
# STEP 5: FULL PREDICTOR
# =============================================================================

@dataclass
class WavePrediction:
    """Prediction result for a molecule."""
    smiles: str
    atom_scores: np.ndarray
    ranked_atoms: List[int]
    top1_pred: int
    top3_pred: List[int]
    electron_density: np.ndarray
    fukui_indices: np.ndarray
    homo_lumo_gap: float


def predict_som(smiles: str) -> Optional[WavePrediction]:
    """
    Predict site of metabolism using wave mechanics.
    
    Returns ranked atoms by reactivity.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    n_atoms = mol.GetNumAtoms()
    if n_atoms < 2:
        return None
    
    # Step 1: Build Hamiltonian
    H, A, V = build_hamiltonian(mol)
    
    # Step 2: Solve eigenstates
    energies, states = solve_eigenstates(H)
    
    # Compute HOMO-LUMO gap for graph model
    n_orbitals = len(energies)
    homo_idx = min(n_orbitals // 2, n_orbitals - 1)
    lumo_idx = min(homo_idx + 1, n_orbitals - 1)
    gap = abs(energies[lumo_idx] - energies[homo_idx])
    
    # Step 3: Compute various reactivity indices
    density = get_electron_density(states, n_atoms)
    fukui = compute_fukui_index(states, energies, n_atoms)
    softness = compute_local_softness(mol, states, energies, n_atoms)
    
    # Step 4: Compute full susceptibility for each atom
    susceptibilities = np.zeros(n_atoms)
    for i in range(n_atoms):
        P = build_perturbation(mol, i, A)
        chi = compute_susceptibility(H, energies, states, P, n_atoms)
        susceptibilities[i] = chi
    
    # Normalize susceptibilities
    if susceptibilities.max() > 0:
        susceptibilities = susceptibilities / susceptibilities.max()
    
    # Step 5: Combine indices into final score
    scores = np.zeros(n_atoms)
    
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        
        # Only score carbons (CYP primarily oxidizes C-H bonds)
        if atom.GetAtomicNum() != 6:
            scores[i] = -np.inf
            continue
        
        # Must have hydrogen for H-abstraction (main CYP mechanism)
        n_H = atom.GetTotalNumHs()
        if n_H == 0:
            scores[i] = -np.inf
            continue
        
        # Combine reactivity measures
        scores[i] = (
            0.35 * softness[i] +           # Local softness
            0.35 * fukui[i] +              # Fukui index (HOMO density)
            0.20 * susceptibilities[i] +    # Linear response
            0.10 * density[i]              # Total electron density
        )
        
        # Boost for more hydrogens
        scores[i] *= (1 + 0.1 * n_H)
    
    # Normalize scores for valid atoms
    valid_mask = scores > -np.inf
    if valid_mask.sum() > 0:
        valid_scores = scores[valid_mask]
        scores[valid_mask] = (valid_scores - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-8)
    
    # Rank atoms
    ranked = np.argsort(-scores)  # Descending
    ranked = [int(i) for i in ranked if scores[i] > -np.inf]
    
    if not ranked:
        ranked = list(range(n_atoms))
    
    return WavePrediction(
        smiles=smiles,
        atom_scores=scores,
        ranked_atoms=ranked,
        top1_pred=ranked[0] if ranked else 0,
        top3_pred=ranked[:3] if len(ranked) >= 3 else ranked,
        electron_density=density,
        fukui_indices=fukui,
        homo_lumo_gap=gap,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data_path: str) -> Dict[str, float]:
    """Evaluate on dataset."""
    with open(data_path) as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'drugs' in data:
        drugs = data['drugs']
    else:
        drugs = data
    
    top1_correct = 0
    top3_correct = 0
    total = 0
    
    print(f"\nEvaluating on {len(drugs)} molecules...")
    print("-" * 60)
    
    for i, entry in enumerate(drugs):
        smiles = entry.get('smiles', '')
        sites = entry.get('site_atoms', [])
        
        if not smiles or not sites:
            continue
        
        pred = predict_som(smiles)
        if pred is None:
            continue
        
        # Check predictions
        if pred.top1_pred in sites:
            top1_correct += 1
        
        if any(p in sites for p in pred.top3_pred):
            top3_correct += 1
        
        total += 1
        
        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(drugs)}: "
                  f"Top-1={top1_correct/total*100:.1f}%, "
                  f"Top-3={top3_correct/total*100:.1f}%")
    
    results = {
        'top1': top1_correct / max(total, 1),
        'top3': top3_correct / max(total, 1),
        'total': total,
    }
    
    print("\n" + "=" * 60)
    print("WAVE METABOLISM PREDICTOR - RESULTS")
    print("=" * 60)
    print(f"Total evaluated: {total}")
    print(f"Top-1 Accuracy: {results['top1']*100:.1f}%")
    print(f"Top-3 Accuracy: {results['top3']*100:.1f}%")
    print("=" * 60)
    
    return results


def demo():
    """Demo on a few molecules."""
    test_molecules = [
        ("CCO", "Ethanol"),
        ("c1ccccc1C", "Toluene"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),
    ]
    
    print("\n" + "=" * 60)
    print("WAVE METABOLISM PREDICTOR - DEMO")
    print("=" * 60)
    
    for smiles, name in test_molecules:
        pred = predict_som(smiles)
        if pred is None:
            print(f"\n{name}: Failed to parse")
            continue
        
        print(f"\n{name} ({smiles})")
        print(f"  HOMO-LUMO gap: {pred.homo_lumo_gap:.3f}")
        print(f"  Top predictions: {pred.top3_pred}")
        print(f"  Scores: {pred.atom_scores[pred.atom_scores > -np.inf][:5].round(3)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        evaluate(data_path)
    else:
        demo()
        print("\n\nUsage: python wave_predictor.py <data.json>")
