#!/usr/bin/env python3
"""
ENVIRONMENT-AWARE WAVE METABOLISM PREDICTOR
============================================

The wave doesn't exist in isolation. It exists in an ENVIRONMENT.

Key modifications from static wave model:

1. ENZYME FIELD: CYP creates an electrostatic perturbation
   - Model as external potential approaching from accessible direction
   - Atoms "facing" the enzyme feel stronger field

2. 3D ACCESSIBILITY: Buried atoms can't react
   - Use solvent-accessible surface area (SASA)
   - Or topological accessibility from graph structure

3. CONFORMATIONAL SAMPLING: Molecule flexes
   - Generate multiple conformers
   - Average wave properties across conformers

4. ELECTRONIC ENVIRONMENT: Neighboring atoms modify local potential
   - Inductive effects propagate through bonds
   - Resonance effects in conjugated systems

The wave equation becomes:
   H(r,t) = H_molecule + V_enzyme(r) + V_solvent(r)
   
Where V_enzyme depends on where the enzyme approaches.
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.Chem import rdFreeSASA
except ImportError:
    raise ImportError("RDKit required: pip install rdkit")


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

ELECTRONEGATIVITY = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66,
}

# Atomic radii for SASA (Angstroms)
VDW_RADII = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
}


# =============================================================================
# 3D STRUCTURE & ACCESSIBILITY
# =============================================================================

def generate_3d_conformer(mol, n_conformers: int = 5):
    """
    Generate 3D conformer(s) for the molecule.
    
    Returns mol with embedded coordinates, or None if failed.
    """
    mol = Chem.AddHs(mol)
    
    # Generate conformers
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.numThreads = 1
    
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
    
    if len(conf_ids) == 0:
        # Fallback: try without ETKDG
        AllChem.EmbedMolecule(mol, randomSeed=42)
        if mol.GetNumConformers() == 0:
            return None
    
    # Optimize
    for conf_id in range(mol.GetNumConformers()):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id, maxIters=200)
        except:
            pass
    
    return mol


def compute_sasa(mol, conf_id: int = 0) -> np.ndarray:
    """
    Compute Solvent Accessible Surface Area for each atom.
    
    High SASA = exposed to solvent = accessible to enzyme.
    """
    n = mol.GetNumAtoms()
    
    if mol.GetNumConformers() == 0:
        # No 3D structure - use topological accessibility instead
        return compute_topological_accessibility(mol)
    
    try:
        # Get radii
        radii = rdFreeSASA.classifyAtoms(mol)
        
        # Compute SASA
        sasa = rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id)
        
        # Get per-atom SASA
        atom_sasa = np.zeros(n)
        for i in range(n):
            atom_sasa[i] = sasa  # Simplified - actual per-atom needs more work
        
        return atom_sasa
        
    except Exception as e:
        return compute_topological_accessibility(mol)


def compute_topological_accessibility(mol) -> np.ndarray:
    """
    Compute accessibility from graph topology (no 3D needed).
    
    Idea: atoms on the "periphery" of the graph are more accessible.
    Periphery = high eccentricity, low betweenness.
    """
    n = mol.GetNumAtoms()
    
    # Build adjacency
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1
    
    # Compute shortest path distances
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    
    # Initialize distances from adjacency
    dist = np.where(A > 0, 1, dist)
    np.fill_diagonal(dist, 0)
    
    # Floyd-Warshall again
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    
    # Eccentricity = max distance to any other atom
    eccentricity = np.max(dist, axis=1)
    eccentricity = np.where(np.isinf(eccentricity), n, eccentricity)
    
    # High eccentricity = peripheral = accessible
    accessibility = eccentricity / (eccentricity.max() + 1e-10)
    
    # Also consider degree (low degree = peripheral)
    degree = A.sum(axis=1)
    degree_factor = 1.0 / (degree + 1)
    degree_factor = degree_factor / degree_factor.max()
    
    # Combine
    accessibility = 0.6 * accessibility + 0.4 * degree_factor
    
    return accessibility


def compute_3d_accessibility(mol) -> np.ndarray:
    """
    Compute 3D accessibility averaged over conformers.
    
    Uses distance from centroid and exposure to "outside".
    """
    n_heavy = mol.GetNumHeavyAtoms()
    n = mol.GetNumAtoms()
    
    if mol.GetNumConformers() == 0:
        return compute_topological_accessibility(mol)
    
    # Average over conformers
    accessibility = np.zeros(n)
    n_confs = mol.GetNumConformers()
    
    for conf_id in range(n_confs):
        conf = mol.GetConformer(conf_id)
        positions = np.array([conf.GetAtomPosition(i) for i in range(n)])
        
        # Centroid
        centroid = positions.mean(axis=0)
        
        # Distance from centroid
        dist_from_center = np.linalg.norm(positions - centroid, axis=1)
        
        # Normalize: far from center = accessible
        if dist_from_center.max() > 0:
            conf_access = dist_from_center / dist_from_center.max()
        else:
            conf_access = np.ones(n)
        
        accessibility += conf_access
    
    accessibility /= n_confs
    
    return accessibility


# =============================================================================
# ELECTRONIC ENVIRONMENT
# =============================================================================

def compute_inductive_effect(mol) -> np.ndarray:
    """
    Compute inductive effect at each atom.
    
    Electronegative neighbors pull electron density away.
    This affects local reactivity.
    """
    n = mol.GetNumAtoms()
    
    # Base electronegativity
    chi = np.array([
        ELECTRONEGATIVITY.get(mol.GetAtomWithIdx(i).GetAtomicNum(), 2.5)
        for i in range(n)
    ])
    
    # Build adjacency with distance decay
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1
    
    # Inductive effect: weighted sum of neighbor electronegativities
    # Decays with distance
    inductive = np.zeros(n)
    
    # Direct neighbors (distance 1)
    for i in range(n):
        neighbor_chi = 0
        n_neighbors = 0
        for j in range(n):
            if A[i, j] > 0:
                neighbor_chi += chi[j]
                n_neighbors += 1
        if n_neighbors > 0:
            inductive[i] = neighbor_chi / n_neighbors - chi[i]
    
    return inductive


def compute_resonance_participation(mol) -> np.ndarray:
    """
    Compute participation in resonance/conjugated systems.
    
    Atoms in conjugated systems have delocalized electrons,
    making them more reactive.
    """
    n = mol.GetNumAtoms()
    
    # Build adjacency with bond orders
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        if bond.GetIsAromatic():
            A[i, j] = A[j, i] = 1.5
        elif bond.GetBondTypeAsDouble() >= 2:
            A[i, j] = A[j, i] = 2.0
        else:
            A[i, j] = A[j, i] = 1.0
    
    # Resonance participation = connectivity to unsaturated bonds
    resonance = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        # Check for pi electrons (aromatic or double bonds)
        if atom.GetIsAromatic():
            resonance[i] = 1.0
        else:
            # Count unsaturated neighbors
            for bond in atom.GetBonds():
                if bond.GetIsAromatic() or bond.GetBondTypeAsDouble() >= 2:
                    resonance[i] += 0.5
    
    # Normalize
    if resonance.max() > 0:
        resonance = resonance / resonance.max()
    
    return resonance


# =============================================================================
# ENZYME INTERACTION MODEL
# =============================================================================

def compute_enzyme_coupling(mol, accessibility: np.ndarray) -> np.ndarray:
    """
    Model how each atom couples to the approaching enzyme.
    
    The enzyme (CYP Fe=O) is an electrophile.
    Coupling depends on:
    1. Accessibility (can enzyme reach it?)
    2. Local electron density (what's there to oxidize?)
    3. Orientation (facing the enzyme?)
    
    We model this as a weighted combination.
    """
    n = mol.GetNumAtoms()
    
    # Base coupling from accessibility
    coupling = accessibility.copy()
    
    # Modify by electronic factors
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        # Non-carbons don't couple (for SoM)
        if atom.GetAtomicNum() != 6:
            coupling[i] = 0
            continue
        
        # Electron-rich carbons couple better
        # Check for electron-donating neighbors
        for neighbor in atom.GetNeighbors():
            z = neighbor.GetAtomicNum()
            n_H = neighbor.GetTotalNumHs()
            
            if z == 7 and n_H > 0:  # -NH2, -NHR
                coupling[i] *= 1.4
            elif z == 7:  # -NR2
                coupling[i] *= 1.3
            elif z == 8 and n_H > 0:  # -OH
                coupling[i] *= 1.3
            elif z == 8:  # -OR
                coupling[i] *= 1.2
    
    return coupling


# =============================================================================
# ENVIRONMENT-AWARE WAVE EQUATION
# =============================================================================

def build_environment_hamiltonian(
    mol,
    accessibility: np.ndarray,
    inductive: np.ndarray,
    enzyme_coupling: np.ndarray,
) -> np.ndarray:
    """
    Build Hamiltonian that includes environmental effects.
    
    H = H_molecule + V_environment
    
    Where V_environment includes:
    - Inductive effects from neighbors
    - Enzyme field (accessibility-weighted)
    - Solvent screening
    """
    n = mol.GetNumAtoms()
    
    # Molecular Hamiltonian (graph Laplacian)
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            w = 1.5
        A[i, j] = A[j, i] = w
    
    D = np.diag(A.sum(axis=1))
    H_mol = D - A
    
    # Environmental potential
    V_env = np.zeros(n)
    
    # Inductive effect modifies on-site energy
    V_env += 0.3 * inductive
    
    # Enzyme field: accessible atoms feel attractive potential
    V_env -= 0.5 * enzyme_coupling  # Negative = attractive
    
    # Full Hamiltonian
    H = H_mol + np.diag(V_env)
    
    return H


def solve_environment_wave(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the wave equation with environmental effects.
    
    Returns eigenvalues, eigenvectors, and derived quantities.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    n = len(eigenvalues)
    
    # Amplitude (low-frequency participation)
    amplitude = np.zeros(n)
    for k in range(1, min(n, 6)):
        weight = 1.0 / (eigenvalues[k] - eigenvalues[0] + 0.1)
        amplitude += weight * eigenvectors[:, k] ** 2
    
    if amplitude.max() > 0:
        amplitude /= amplitude.max()
    
    # Flexibility (inverse of high-freq participation)
    rigidity = np.zeros(n)
    for k in range(max(1, n-3), n):
        rigidity += eigenvectors[:, k] ** 2
    
    flexibility = 1.0 / (rigidity + 0.1)
    if flexibility.max() > 0:
        flexibility /= flexibility.max()
    
    return eigenvalues, amplitude, flexibility


# =============================================================================
# MAIN PREDICTOR
# =============================================================================

@dataclass
class EnvironmentPrediction:
    """Prediction with environmental factors."""
    smiles: str
    scores: np.ndarray
    top1: int
    top3: List[int]
    accessibility: np.ndarray
    enzyme_coupling: np.ndarray
    amplitude: np.ndarray
    flexibility: np.ndarray


def predict_with_environment(smiles: str, use_3d: bool = True) -> Optional[EnvironmentPrediction]:
    """
    Predict SoM using environment-aware wave model.
    
    Args:
        smiles: SMILES string
        use_3d: Whether to compute 3D conformers (slower but more accurate)
    
    Returns:
        EnvironmentPrediction with ranked atoms and diagnostics
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    
    n = mol.GetNumAtoms()  # Heavy atoms only
    
    # Compute 3D accessibility
    if use_3d:
        mol_3d = generate_3d_conformer(Chem.MolFromSmiles(smiles), n_conformers=3)
        if mol_3d is not None:
            # Get accessibility only for heavy atoms
            access_full = compute_3d_accessibility(mol_3d)
            # Map back to original indices (heavy atoms only)
            accessibility = np.zeros(n)
            heavy_idx = 0
            for i in range(mol_3d.GetNumAtoms()):
                if mol_3d.GetAtomWithIdx(i).GetAtomicNum() != 1:  # Not hydrogen
                    if heavy_idx < n:
                        accessibility[heavy_idx] = access_full[i]
                    heavy_idx += 1
            if accessibility.max() > 0:
                accessibility = accessibility / accessibility.max()
        else:
            accessibility = compute_topological_accessibility(mol)
    else:
        accessibility = compute_topological_accessibility(mol)
    
    # Compute electronic environment
    inductive = compute_inductive_effect(mol)
    resonance = compute_resonance_participation(mol)
    
    # Compute enzyme coupling
    enzyme_coupling = compute_enzyme_coupling(mol, accessibility)
    
    # Build environment-aware Hamiltonian
    H = build_environment_hamiltonian(mol, accessibility, inductive, enzyme_coupling)
    
    # Solve wave equation
    eigenvalues, amplitude, flexibility = solve_environment_wave(H)
    
    # Compute final reactivity score
    scores = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        # Only carbons
        if atom.GetAtomicNum() != 6:
            scores[i] = -np.inf
            continue
        
        n_H = atom.GetTotalNumHs()
        is_arom = atom.GetIsAromatic()
        
        # Environment-aware score
        # Combines: wave properties + accessibility + electronic effects
        scores[i] = (
            0.30 * amplitude[i] * flexibility[i] +  # Wave factor
            0.30 * accessibility[i] +                 # 3D accessibility
            0.25 * enzyme_coupling[i] +               # Enzyme interaction
            0.15 * resonance[i]                       # Resonance stabilization
        )
        
        # H requirement
        if n_H == 0:
            if is_arom:
                scores[i] *= 0.6  # Aromatic oxidation possible
            else:
                scores[i] = -np.inf
                continue
        else:
            scores[i] *= (1 + 0.10 * n_H)
        
        # Alpha positions (already partially in enzyme_coupling, but boost more)
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7:
                scores[i] *= 1.4
                break
            elif z == 8:
                scores[i] *= 1.25
                break
            elif z == 16:
                scores[i] *= 1.3
                break
        
        # Benzylic
        if not is_arom and n_H > 0:
            for nbr in atom.GetNeighbors():
                if nbr.GetIsAromatic():
                    scores[i] *= 1.2
                    break
    
    # Rank valid atoms
    valid = [i for i in range(n) if scores[i] > -np.inf]
    if not valid:
        return None
    
    ranked = sorted(valid, key=lambda x: -scores[x])
    
    return EnvironmentPrediction(
        smiles=smiles,
        scores=scores,
        top1=ranked[0],
        top3=ranked[:3],
        accessibility=accessibility,
        enzyme_coupling=enzyme_coupling,
        amplitude=amplitude,
        flexibility=flexibility,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data_path: str, use_3d: bool = False) -> Dict:
    """Evaluate on dataset."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get('drugs', data)
    
    top1 = top3 = total = 0
    by_source = {}
    
    print(f"\nEvaluating ENVIRONMENT-AWARE WAVE PREDICTOR (3D={use_3d})...")
    print("-" * 60)
    
    for i, entry in enumerate(drugs):
        smiles = entry.get('smiles', '')
        sites = entry.get('site_atoms', [])
        source = entry.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        pred = predict_with_environment(smiles, use_3d=use_3d)
        if pred is None:
            continue
        
        if source not in by_source:
            by_source[source] = {'t1': 0, 't3': 0, 'n': 0}
        by_source[source]['n'] += 1
        
        if pred.top1 in sites:
            top1 += 1
            by_source[source]['t1'] += 1
        if any(p in sites for p in pred.top3):
            top3 += 1
            by_source[source]['t3'] += 1
        total += 1
        
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(drugs)}: Top-1={top1/total*100:.1f}%")
    
    print("\n" + "=" * 60)
    print(f"ENVIRONMENT-AWARE WAVE MODEL (3D={use_3d})")
    print("=" * 60)
    print(f"\nOVERALL: Top-1={top1/total*100:.1f}%, Top-3={top3/total*100:.1f}%")
    
    print("\nBY SOURCE:")
    for src, s in sorted(by_source.items(), key=lambda x: -x[1]['n']):
        if s['n'] >= 5:
            print(f"  {src:20s}: Top-1={s['t1']/s['n']*100:5.1f}%, Top-3={s['t3']/s['n']*100:5.1f}% (n={s['n']})")
    
    return {'top1': top1/total, 'top3': top3/total, 'by_source': by_source}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Fast mode (no 3D)
        print("\n>>> TOPOLOGICAL MODE (fast)")
        evaluate(sys.argv[1], use_3d=False)
        
        # 3D mode
        print("\n\n>>> 3D MODE (with conformers)")
        evaluate(sys.argv[1], use_3d=True)
    else:
        print("Usage: python wave_env.py <data.json>")
