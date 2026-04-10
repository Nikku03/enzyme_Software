#!/usr/bin/env python3
"""
QUANTUM RESONANCE MODEL
=======================

At the quantum scale, there are no solid objects - only waves.

A molecule is a stable interference pattern.
A bond is a standing wave between two atoms.
The enzyme is an incoming wave perturbation.

A bond breaks when:
1. The enzyme wave COUPLES to the bond wave
2. Energy transfers via RESONANCE
3. The standing wave pattern destabilizes

The question: Which bond resonates most strongly with the enzyme?

Physics:
    ω_bond = natural frequency of the bond oscillation
    ω_enzyme = characteristic frequency of Fe=O radical
    
    Coupling = ∫ ψ_bond × ψ_enzyme dr  (spatial overlap)
    Resonance = 1 / (ω_bond - ω_enzyme)²  (frequency match)
    
    Energy transfer ∝ Coupling² × Resonance
    
    Bond that breaks = maximum energy transfer
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit required")


# =============================================================================
# PHYSICAL CONSTANTS (in relative units)
# =============================================================================

# Characteristic frequencies (arbitrary units, ratios matter)
OMEGA_ENZYME = 1.0  # Fe=O stretching frequency (reference)

# Bond frequencies depend on bond strength and atomic masses
# Weaker bonds = lower frequency = closer to enzyme frequency = more resonant
BOND_FREQUENCY_BASE = {
    'C-C': 1.2,    # Strong, high frequency
    'C=C': 1.5,    # Stronger
    'C-N': 1.1,    # Slightly weaker than C-C
    'C-O': 1.15,   # 
    'C-S': 0.9,    # Weaker, heavier atom
    'C-H': 1.8,    # Very strong, high freq
    'aromatic': 1.35,  # Intermediate
}

# Atomic "wave amplitude" - how strongly each atom participates in waves
# Related to polarizability / electron density
WAVE_AMPLITUDE = {
    6: 1.0,   # C (reference)
    7: 0.9,   # N (tighter electrons)
    8: 0.8,   # O (even tighter)
    16: 1.3,  # S (more diffuse)
    1: 0.5,   # H (small)
}


# =============================================================================
# WAVE MECHANICS ON MOLECULAR GRAPH
# =============================================================================

def compute_bond_waves(mol) -> Dict[Tuple[int, int], Dict]:
    """
    Compute the wave properties of each bond.
    
    Each bond is a standing wave between two atoms.
    Properties:
        - frequency: natural oscillation frequency
        - amplitude: wave strength
        - phase: spatial distribution
    """
    n = mol.GetNumAtoms()
    bond_waves = {}
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        
        # Bond type determines base frequency
        if bond.GetIsAromatic():
            base_freq = BOND_FREQUENCY_BASE['aromatic']
        elif bond.GetBondTypeAsDouble() >= 2:
            base_freq = BOND_FREQUENCY_BASE['C=C']
        else:
            # Single bond - type depends on atoms
            key = f"C-{atom_j.GetSymbol()}" if atom_i.GetAtomicNum() == 6 else "C-C"
            base_freq = BOND_FREQUENCY_BASE.get(key, 1.2)
        
        # Frequency modified by atomic masses (heavier = slower)
        mass_i = atom_i.GetMass()
        mass_j = atom_j.GetMass()
        reduced_mass = (mass_i * mass_j) / (mass_i + mass_j)
        freq = base_freq / np.sqrt(reduced_mass / 6.0)  # Normalize to carbon
        
        # Amplitude = geometric mean of atomic wave amplitudes
        amp_i = WAVE_AMPLITUDE.get(atom_i.GetAtomicNum(), 1.0)
        amp_j = WAVE_AMPLITUDE.get(atom_j.GetAtomicNum(), 1.0)
        amplitude = np.sqrt(amp_i * amp_j)
        
        # Phase distribution on the molecular graph
        # This determines spatial extent of the bond wave
        phase = compute_bond_phase(mol, i, j)
        
        bond_waves[(i, j)] = {
            'frequency': freq,
            'amplitude': amplitude,
            'phase': phase,
            'bond_order': bond.GetBondTypeAsDouble(),
            'is_aromatic': bond.GetIsAromatic(),
        }
    
    return bond_waves


def compute_bond_phase(mol, i: int, j: int) -> np.ndarray:
    """
    Compute the spatial phase distribution of a bond wave.
    
    The bond wave is localized between atoms i and j,
    but has tails extending to neighbors.
    
    Returns an array of phase values for each atom.
    """
    n = mol.GetNumAtoms()
    phase = np.zeros(n)
    
    # Bond is centered between i and j
    phase[i] = 1.0
    phase[j] = -1.0  # Opposite phase (standing wave)
    
    # Neighbors feel the wave with reduced amplitude
    atom_i = mol.GetAtomWithIdx(i)
    atom_j = mol.GetAtomWithIdx(j)
    
    for neighbor in atom_i.GetNeighbors():
        k = neighbor.GetIdx()
        if k != j:
            phase[k] = 0.3  # Same phase as i, reduced
    
    for neighbor in atom_j.GetNeighbors():
        k = neighbor.GetIdx()
        if k != i:
            phase[k] = -0.3  # Same phase as j, reduced
    
    return phase


# =============================================================================
# ENZYME WAVE MODEL
# =============================================================================

def compute_enzyme_wave(mol, approach_atom: int) -> np.ndarray:
    """
    Model the enzyme wave as it approaches a specific atom.
    
    The Fe=O radical creates a localized perturbation that
    decays with distance from the approach point.
    
    Returns phase array representing enzyme wave at each atom.
    """
    n = mol.GetNumAtoms()
    
    # Build adjacency for distance calculation
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1
    
    # BFS distances from approach atom
    distances = np.full(n, np.inf)
    distances[approach_atom] = 0
    queue = [approach_atom]
    while queue:
        curr = queue.pop(0)
        for next_atom in range(n):
            if A[curr, next_atom] > 0 and distances[next_atom] == np.inf:
                distances[next_atom] = distances[curr] + 1
                queue.append(next_atom)
    
    # Enzyme wave: localized at approach point, decays exponentially
    decay_length = 2.0  # Affects ~2 bonds away
    enzyme_wave = np.exp(-distances / decay_length)
    
    return enzyme_wave


# =============================================================================
# RESONANCE COUPLING CALCULATION
# =============================================================================

def compute_coupling(bond_phase: np.ndarray, enzyme_wave: np.ndarray) -> float:
    """
    Compute spatial coupling between bond wave and enzyme wave.
    
    Coupling = ∫ ψ_bond × ψ_enzyme dr
    
    High coupling = large spatial overlap = strong interaction.
    """
    # Dot product gives overlap integral
    coupling = np.abs(np.dot(bond_phase, enzyme_wave))
    return coupling


def compute_resonance(omega_bond: float, omega_enzyme: float = OMEGA_ENZYME) -> float:
    """
    Compute resonance factor between two frequencies.
    
    Resonance = 1 / (ω_bond - ω_enzyme)² + damping
    
    When frequencies match → resonance → efficient energy transfer.
    """
    # Lorentzian resonance with damping
    damping = 0.1  # Prevents infinity at exact resonance
    delta_omega = omega_bond - omega_enzyme
    resonance = 1.0 / (delta_omega**2 + damping**2)
    
    return resonance


def compute_energy_transfer(
    coupling: float,
    resonance: float,
    amplitude: float,
) -> float:
    """
    Compute total energy transfer from enzyme to bond.
    
    Energy ∝ |Coupling|² × Resonance × Amplitude²
    """
    return coupling**2 * resonance * amplitude**2


# =============================================================================
# MAIN PREDICTOR
# =============================================================================

@dataclass
class ResonancePrediction:
    """Prediction result from resonance model."""
    smiles: str
    atom_scores: np.ndarray  # Score for each atom
    bond_scores: Dict[Tuple[int, int], float]  # Score for each bond
    top1_atom: int
    top3_atoms: List[int]
    resonance_data: Dict  # Diagnostic info


def predict_resonance(smiles: str) -> Optional[ResonancePrediction]:
    """
    Predict site of metabolism using quantum resonance.
    
    For each potential reaction site (C with H or aromatic C):
    1. Model enzyme approaching that site
    2. Compute coupling to each nearby bond
    3. Find which bond would receive most energy
    4. Score the site by total energy transfer
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    
    n = mol.GetNumAtoms()
    
    # Compute bond wave properties
    bond_waves = compute_bond_waves(mol)
    
    # For each potential reaction site, compute resonance energy
    atom_scores = np.full(n, -np.inf)
    all_bond_scores = {}
    resonance_data = {}
    
    for site in range(n):
        atom = mol.GetAtomWithIdx(site)
        
        # Only consider carbons
        if atom.GetAtomicNum() != 6:
            continue
        
        n_H = atom.GetTotalNumHs()
        is_arom = atom.GetIsAromatic()
        
        # Must have H (for H-abstraction) or be aromatic (for oxidation)
        if n_H == 0 and not is_arom:
            continue
        
        # Model enzyme approaching this site
        enzyme_wave = compute_enzyme_wave(mol, site)
        
        # Compute energy transfer to each bond involving this atom
        site_energy = 0.0
        site_bond_scores = {}
        
        for (i, j), wave_props in bond_waves.items():
            # Only bonds connected to this site
            if site not in (i, j):
                continue
            
            # Spatial coupling
            coupling = compute_coupling(wave_props['phase'], enzyme_wave)
            
            # Frequency resonance
            resonance = compute_resonance(wave_props['frequency'])
            
            # Total energy transfer
            energy = compute_energy_transfer(
                coupling, resonance, wave_props['amplitude']
            )
            
            site_bond_scores[(i, j)] = energy
            site_energy += energy
        
        # Boost for C-H bonds (main target of CYP)
        if n_H > 0:
            site_energy *= (1 + 0.2 * n_H)
        
        # Store
        atom_scores[site] = site_energy
        all_bond_scores.update(site_bond_scores)
        resonance_data[site] = {
            'enzyme_wave': enzyme_wave,
            'bond_energies': site_bond_scores,
        }
    
    # Rank atoms
    valid = [i for i in range(n) if atom_scores[i] > -np.inf]
    if not valid:
        return None
    
    ranked = sorted(valid, key=lambda x: -atom_scores[x])
    
    return ResonancePrediction(
        smiles=smiles,
        atom_scores=atom_scores,
        bond_scores=all_bond_scores,
        top1_atom=ranked[0],
        top3_atoms=ranked[:3],
        resonance_data=resonance_data,
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data_path: str) -> Dict:
    """Evaluate on dataset."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get('drugs', data)
    
    top1 = top3 = total = 0
    by_source = {}
    
    print(f"\nEvaluating QUANTUM RESONANCE MODEL on {len(drugs)} molecules...")
    print("-" * 60)
    
    for i, entry in enumerate(drugs):
        smiles = entry.get('smiles', '')
        sites = entry.get('site_atoms', [])
        source = entry.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        pred = predict_resonance(smiles)
        if pred is None:
            continue
        
        if source not in by_source:
            by_source[source] = {'t1': 0, 't3': 0, 'n': 0}
        by_source[source]['n'] += 1
        
        if pred.top1_atom in sites:
            top1 += 1
            by_source[source]['t1'] += 1
        if any(p in sites for p in pred.top3_atoms):
            top3 += 1
            by_source[source]['t3'] += 1
        total += 1
        
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(drugs)}: Top-1={top1/total*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("QUANTUM RESONANCE MODEL - RESULTS")
    print("=" * 60)
    print(f"\nOVERALL: Top-1={top1/total*100:.1f}%, Top-3={top3/total*100:.1f}%")
    
    print("\nBY SOURCE:")
    for src, s in sorted(by_source.items(), key=lambda x: -x[1]['n']):
        if s['n'] >= 5:
            print(f"  {src:20s}: Top-1={s['t1']/s['n']*100:5.1f}%, Top-3={s['t3']/s['n']*100:5.1f}% (n={s['n']})")
    
    return {'top1': top1/total, 'top3': top3/total, 'by_source': by_source}


def demo():
    """Demo on example molecules."""
    examples = [
        ("Cc1ccccc1", "Toluene - benzylic C should be most reactive"),
        ("CCO", "Ethanol - alpha-O carbon should be reactive"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),
    ]
    
    print("\n" + "=" * 60)
    print("QUANTUM RESONANCE MODEL - DEMO")
    print("=" * 60)
    
    for smiles, name in examples:
        pred = predict_resonance(smiles)
        if pred is None:
            print(f"\n{name}: Failed")
            continue
        
        print(f"\n{name}")
        print(f"  SMILES: {smiles}")
        print(f"  Top-1: atom {pred.top1_atom}")
        print(f"  Top-3: atoms {pred.top3_atoms}")
        
        # Show bond energies for top atom
        if pred.top1_atom in pred.resonance_data:
            bond_e = pred.resonance_data[pred.top1_atom]['bond_energies']
            print(f"  Bond energies: {[(b, round(e, 3)) for b, e in sorted(bond_e.items(), key=lambda x: -x[1])[:3]]}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])
    else:
        demo()
        print("\n\nUsage: python quantum_resonance.py <data.json>")
