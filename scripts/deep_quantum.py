#!/usr/bin/env python3
"""
DEEP QUANTUM FIELD MODEL
========================

Going beyond the simple wave picture.

At the true quantum scale:

1. SPACETIME WAVES: The wave function evolves in time. Stationary states
   are just one slice. The dynamics matter - how fast does the wave
   respond to perturbation?

2. INTERFERENCE: Multiple modes superpose. The LOCAL interference pattern
   at each atom determines its properties - not just amplitude, but phase
   relationships.

3. FIELD COUPLING: The enzyme isn't a point perturbation. It's an extended
   field that couples differently to different parts of the molecule based
   on geometry and symmetry.

4. TUNNELING: Some reactions happen via quantum tunneling - electrons move
   through barriers. The tunneling probability depends on barrier height
   and width.

5. CORRELATION: Electrons are correlated (entangled). Multi-electron effects
   matter. A simple single-particle picture misses this.

6. RESPONSE FUNCTION: The key quantity is the response function - how does
   the system respond to an external perturbation at frequency ω?

Let me implement these properly.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import linalg
from scipy.special import erf

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("RDKit required")


# =============================================================================
# FUNDAMENTAL CONSTANTS (in atomic units where helpful)
# =============================================================================

# Characteristic energies (eV, for physical interpretation)
HOMO_LUMO_TYPICAL = 3.0  # Typical gap for organic molecules
ENZYME_ENERGY = 2.0  # Fe=O characteristic energy (in same units)
THERMAL_ENERGY = 0.026  # kT at 300K

# Tunneling parameters
BARRIER_WIDTH_TYPICAL = 1.0  # Angstroms
ELECTRON_MASS = 1.0  # In atomic units


# =============================================================================
# 1. TIME-DEPENDENT WAVE DYNAMICS
# =============================================================================

def compute_time_evolution(H: np.ndarray, t: float) -> np.ndarray:
    """
    Compute the time evolution operator U(t) = exp(-i H t).
    
    This tells us how the wave function evolves in time.
    Fast evolution = responsive to perturbation.
    """
    # Diagonalize H
    E, V = np.linalg.eigh(H)
    
    # U(t) = V exp(-i E t) V^†
    phases = np.exp(-1j * E * t)
    U = V @ np.diag(phases) @ V.T.conj()
    
    return U


def compute_response_time(H: np.ndarray, site: int) -> float:
    """
    Compute characteristic response time at a site.
    
    How quickly does a perturbation at this site spread?
    Fast spreading = well-coupled to rest of molecule.
    """
    n = H.shape[0]
    
    # Initial state: localized at site
    psi_0 = np.zeros(n, dtype=complex)
    psi_0[site] = 1.0
    
    # Evolve for short time
    dt = 0.1
    U = compute_time_evolution(H, dt)
    psi_t = U @ psi_0
    
    # How much has the state spread?
    # Measure by decrease in probability at original site
    spread = 1 - np.abs(psi_t[site])**2
    
    return spread


# =============================================================================
# 2. INTERFERENCE PATTERN ANALYSIS
# =============================================================================

def compute_local_interference(eigenvectors: np.ndarray, eigenvalues: np.ndarray, 
                                site: int) -> Dict:
    """
    Analyze the interference pattern at a specific site.
    
    Multiple modes superpose at each site. The relative phases
    and amplitudes determine the local "texture" of the wave.
    
    Returns:
        constructive: how much constructive interference
        destructive: how much destructive interference
        coherence: phase coherence of modes at this site
    """
    n = len(eigenvalues)
    
    # Amplitudes of each mode at this site
    amplitudes = eigenvectors[site, :]
    
    # Phase coherence: are the modes in phase or out of phase?
    # Measure by looking at sign patterns
    positive_contrib = np.sum(amplitudes[amplitudes > 0]**2)
    negative_contrib = np.sum(amplitudes[amplitudes < 0]**2)
    
    # Constructive interference when all same sign
    # Destructive when mixed signs
    total = positive_contrib + negative_contrib
    if total > 0:
        coherence = abs(positive_contrib - negative_contrib) / total
    else:
        coherence = 0
    
    # Also measure "interference energy" - weighted by eigenvalues
    weighted_pos = np.sum(amplitudes[amplitudes > 0]**2 * 
                         np.abs(eigenvalues[amplitudes > 0]))
    weighted_neg = np.sum(amplitudes[amplitudes < 0]**2 * 
                         np.abs(eigenvalues[amplitudes < 0]))
    
    return {
        'coherence': coherence,
        'constructive': positive_contrib,
        'destructive': negative_contrib,
        'interference_energy': weighted_pos + weighted_neg,
    }


# =============================================================================
# 3. EXTENDED ENZYME FIELD
# =============================================================================

def compute_enzyme_field(mol, approach_direction: np.ndarray = None) -> np.ndarray:
    """
    Model the enzyme as an extended electromagnetic field.
    
    The Fe=O creates:
    - Electric field (pulls on electrons)
    - Magnetic effects (spin coupling)
    - Exchange interaction (orbital overlap)
    
    The field has spatial extent and direction.
    """
    n = mol.GetNumAtoms()
    
    # Default: enzyme approaches from +z direction
    if approach_direction is None:
        approach_direction = np.array([0, 0, 1])
    approach_direction = approach_direction / np.linalg.norm(approach_direction)
    
    # Get 3D coordinates if available
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer(0)
        coords = np.array([conf.GetAtomPosition(i) for i in range(n)])
    else:
        # Use graph-based pseudo-coordinates
        coords = compute_pseudo_coordinates(mol)
    
    # Enzyme field strength at each atom
    # Depends on: distance from approach, alignment with field
    field = np.zeros(n)
    
    # Find molecular centroid
    centroid = coords.mean(axis=0)
    
    for i in range(n):
        # Vector from atom to centroid
        r = coords[i] - centroid
        
        # Component along approach direction
        z_component = np.dot(r, approach_direction)
        
        # Atoms "facing" the enzyme feel stronger field
        # Use sigmoid to smooth the transition
        facing_factor = 1 / (1 + np.exp(-2 * z_component))
        
        # Also depends on distance from center (peripheral atoms more exposed)
        distance = np.linalg.norm(r)
        exposure = distance / (coords.max() - coords.min() + 1)
        
        field[i] = facing_factor * (0.5 + 0.5 * exposure)
    
    return field


def compute_pseudo_coordinates(mol) -> np.ndarray:
    """
    Generate pseudo-3D coordinates from graph structure.
    Uses spectral embedding.
    """
    n = mol.GetNumAtoms()
    
    # Build Laplacian
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1
    
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Use first 3 non-trivial eigenvectors as coordinates
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    coords = eigenvectors[:, 1:4]  # Skip the constant eigenvector
    
    return coords


# =============================================================================
# 4. QUANTUM TUNNELING
# =============================================================================

def compute_tunneling_probability(
    mol, 
    site: int, 
    barrier_height: float = 1.0,
    barrier_width: float = 1.0,
) -> float:
    """
    Compute probability of electron tunneling at this site.
    
    Tunneling allows reactions that are classically forbidden.
    
    Uses WKB approximation:
    P ≈ exp(-2 ∫ √(2m(V-E)) dx)
    
    For rectangular barrier: P ≈ exp(-2 κ a)
    where κ = √(2m(V-E))/ℏ
    """
    # Effective barrier depends on local environment
    atom = mol.GetAtomWithIdx(site)
    
    # Barrier is lower for:
    # - Atoms with electron-donating neighbors
    # - Atoms in conjugated systems
    # - Atoms with high flexibility
    
    effective_barrier = barrier_height
    
    # Electron-donating neighbors lower barrier
    for neighbor in atom.GetNeighbors():
        z = neighbor.GetAtomicNum()
        if z == 7:  # N donates electrons
            effective_barrier *= 0.8
        elif z == 8:  # O donates (but also withdraws)
            effective_barrier *= 0.9
        elif z == 16:  # S donates
            effective_barrier *= 0.7
    
    # Conjugation lowers barrier (delocalized electrons tunnel easier)
    if atom.GetIsAromatic():
        effective_barrier *= 0.7
    
    # Tunneling probability (WKB)
    kappa = np.sqrt(2 * ELECTRON_MASS * effective_barrier)
    tunneling = np.exp(-2 * kappa * barrier_width)
    
    return tunneling


# =============================================================================
# 5. ELECTRON CORRELATION
# =============================================================================

def compute_correlation_effects(mol, H: np.ndarray) -> np.ndarray:
    """
    Estimate electron correlation effects.
    
    In a simple model, we capture correlation through:
    1. Pairing energy: electrons prefer to pair in bonds
    2. Exchange energy: same-spin electrons avoid each other
    3. Correlation energy: opposite-spin electrons also correlate
    
    Returns correlation-adjusted energies for each atom.
    """
    n = mol.GetNumAtoms()
    
    # Solve single-particle problem
    E, V = np.linalg.eigh(H)
    
    # Electron density from occupied orbitals
    n_occ = n // 2
    density = np.zeros(n)
    for k in range(n_occ):
        density += 2 * V[:, k]**2  # 2 electrons per orbital
    
    # On-site correlation: Hubbard-like U term
    # High density = high correlation energy (bad)
    U = 1.0  # On-site repulsion
    correlation_energy = U * density**2
    
    # Exchange energy: favors unpaired spins
    # Approximation: high spin density = low exchange cost
    exchange_energy = -0.3 * density
    
    # Net correlation effect
    corr_effect = correlation_energy + exchange_energy
    
    # Normalize
    corr_effect = corr_effect - corr_effect.mean()
    corr_effect = corr_effect / (np.abs(corr_effect).max() + 1e-10)
    
    return corr_effect


# =============================================================================
# 6. FREQUENCY-DEPENDENT RESPONSE
# =============================================================================

def compute_response_function(H: np.ndarray, omega: float, site: int) -> float:
    """
    Compute the response function χ(ω) at a specific site.
    
    This is the fundamental quantity in linear response theory.
    It tells us how the system responds to a perturbation at frequency ω.
    
    χ(ω) = Σ_{n,m} |<n|P|m>|² / (ω - (E_m - E_n) + iη)
    
    Where P is the perturbation operator.
    """
    n = H.shape[0]
    E, V = np.linalg.eigh(H)
    
    # Occupation (half-filling)
    n_occ = n // 2
    
    # Small imaginary part for broadening
    eta = 0.1
    
    # Perturbation: localized at site
    P = np.zeros((n, n))
    P[site, site] = 1.0
    
    chi = 0.0
    
    for m in range(n_occ):  # Occupied
        for k in range(n_occ, n):  # Unoccupied
            # Transition energy
            omega_mk = E[k] - E[m]
            
            # Matrix element
            P_mk = np.abs(V[:, m].T @ P @ V[:, k])**2
            
            # Response (real part)
            chi += P_mk * (omega - omega_mk) / ((omega - omega_mk)**2 + eta**2)
    
    return chi


# =============================================================================
# MAIN DEEP QUANTUM PREDICTOR
# =============================================================================

@dataclass
class DeepQuantumPrediction:
    """Prediction from deep quantum model."""
    smiles: str
    scores: np.ndarray
    top1: int
    top3: List[int]
    components: Dict[str, np.ndarray]


def build_hamiltonian(mol) -> np.ndarray:
    """Build the molecular Hamiltonian."""
    n = mol.GetNumAtoms()
    H = np.zeros((n, n))
    
    # Off-diagonal: hopping
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        t = -bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            t = -1.5
        H[i, j] = H[j, i] = t
    
    # Diagonal: on-site energy
    chi_map = {1: 0.5, 6: 0.0, 7: 0.4, 8: 0.6, 9: 1.0, 16: -0.1, 17: 0.5}
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        H[i, i] = chi_map.get(atom.GetAtomicNum(), 0.0)
    
    return H


def predict_deep_quantum(smiles: str, use_3d: bool = False) -> Optional[DeepQuantumPrediction]:
    """
    Predict SoM using deep quantum field analysis.
    
    Combines:
    1. Flexibility (high-freq mode avoidance)
    2. Time response (wave spreading speed)
    3. Interference pattern (phase coherence)
    4. Enzyme field coupling
    5. Tunneling probability
    6. Correlation effects
    7. Frequency-dependent response
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    
    n = mol.GetNumAtoms()
    
    # Generate 3D if requested
    if use_3d:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        if mol.GetNumConformers() == 0:
            use_3d = False
        else:
            mol = Chem.RemoveHs(mol)
    
    # Build Hamiltonian
    H = build_hamiltonian(mol)
    E, V = np.linalg.eigh(H)
    
    # === COMPONENT 1: FLEXIBILITY ===
    rigidity = np.zeros(n)
    for k in range(max(1, n-3), n):
        rigidity += V[:, k]**2
    flexibility = 1.0 / (rigidity + 0.1)
    flexibility = flexibility / flexibility.max()
    
    # === COMPONENT 2: TIME RESPONSE ===
    time_response = np.zeros(n)
    for i in range(n):
        time_response[i] = compute_response_time(H, i)
    time_response = time_response / (time_response.max() + 1e-10)
    
    # === COMPONENT 3: INTERFERENCE ===
    coherence = np.zeros(n)
    for i in range(n):
        interf = compute_local_interference(V, E, i)
        coherence[i] = interf['coherence']
    
    # === COMPONENT 4: ENZYME FIELD ===
    enzyme_field = compute_enzyme_field(mol)
    
    # === COMPONENT 5: TUNNELING ===
    tunneling = np.zeros(n)
    for i in range(n):
        tunneling[i] = compute_tunneling_probability(mol, i)
    tunneling = tunneling / (tunneling.max() + 1e-10)
    
    # === COMPONENT 6: CORRELATION ===
    correlation = compute_correlation_effects(mol, H)
    
    # === COMPONENT 7: FREQUENCY RESPONSE ===
    freq_response = np.zeros(n)
    omega_enzyme = ENZYME_ENERGY / HOMO_LUMO_TYPICAL  # Normalized
    for i in range(n):
        freq_response[i] = abs(compute_response_function(H, omega_enzyme, i))
    freq_response = freq_response / (freq_response.max() + 1e-10)
    
    # === COMBINE ALL COMPONENTS ===
    scores = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        if atom.GetAtomicNum() != 6:
            scores[i] = -np.inf
            continue
        
        n_H = atom.GetTotalNumHs()
        if n_H == 0 and not atom.GetIsAromatic():
            scores[i] = -np.inf
            continue
        
        # Deep quantum score: weighted combination
        scores[i] = (
            0.25 * flexibility[i] +       # Not locked in local modes
            0.20 * time_response[i] +     # Fast wave spreading
            0.15 * enzyme_field[i] +      # Coupled to enzyme
            0.15 * tunneling[i] +         # Can tunnel
            0.10 * freq_response[i] +     # Resonant with enzyme frequency
            0.10 * (1 - coherence[i]) +   # Destructive interference = unstable
            0.05 * (1 - correlation[i])   # Low correlation energy
        )
        
        # Chemical adjustments
        if n_H > 0:
            scores[i] *= (1 + 0.12 * n_H)
        else:
            scores[i] *= 0.6
        
        # Alpha positions
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() in [7, 8, 16]:
                scores[i] *= 1.35
                break
        
        # Benzylic
        if not atom.GetIsAromatic() and n_H > 0:
            for nbr in atom.GetNeighbors():
                if nbr.GetIsAromatic():
                    scores[i] *= 1.2
                    break
    
    # Rank
    valid = [i for i in range(n) if scores[i] > -np.inf]
    if not valid:
        return None
    
    ranked = sorted(valid, key=lambda x: -scores[x])
    
    return DeepQuantumPrediction(
        smiles=smiles,
        scores=scores,
        top1=ranked[0],
        top3=ranked[:3],
        components={
            'flexibility': flexibility,
            'time_response': time_response,
            'coherence': coherence,
            'enzyme_field': enzyme_field,
            'tunneling': tunneling,
            'correlation': correlation,
            'freq_response': freq_response,
        }
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data_path: str) -> Dict:
    """Evaluate deep quantum model."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get('drugs', data)
    
    top1 = top3 = total = 0
    by_source = {}
    
    print(f"\nEvaluating DEEP QUANTUM MODEL on {len(drugs)} molecules...")
    print("-" * 60)
    
    for i, entry in enumerate(drugs):
        smiles = entry.get('smiles', '')
        sites = entry.get('site_atoms', [])
        source = entry.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        pred = predict_deep_quantum(smiles)
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
    print("DEEP QUANTUM MODEL - RESULTS")
    print("=" * 60)
    print(f"\nOVERALL: Top-1={top1/total*100:.1f}%, Top-3={top3/total*100:.1f}%")
    
    print("\nBY SOURCE:")
    for src, s in sorted(by_source.items(), key=lambda x: -x[1]['n']):
        if s['n'] >= 5:
            print(f"  {src:20s}: Top-1={s['t1']/s['n']*100:5.1f}%, Top-3={s['t3']/s['n']*100:5.1f}% (n={s['n']})")
    
    return {'top1': top1/total, 'top3': top3/total}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])
    else:
        print("Usage: python deep_quantum.py <data.json>")
