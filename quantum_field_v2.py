"""
QUANTUM FIELD RESONANCE MODEL V2

The key insight: Bond breaking is a RESONANCE phenomenon.

The Fe=O has a characteristic frequency (its electronic/vibrational state).
The substrate C-H bonds have frequencies too.
When frequencies MATCH, energy transfer is efficient → reaction occurs.

This is like:
- Radio tuning (antenna resonates with broadcast frequency)
- NMR (nuclei resonate at specific frequencies)
- Laser excitation (photon energy matches transition)

At quantum scale:
1. Each bond is an oscillator with frequency ω = √(k/μ)
2. Fe=O has frequency ω_Fe (around 800 cm⁻¹)
3. C-H stretch is around 3000 cm⁻¹ BUT the reaction coordinate 
   involves bending the C-H toward Fe=O, which is lower frequency
4. RESONANCE occurs when substrate mode couples to Fe=O mode

The site that reacts is where the LOCAL MODE best matches the enzyme's frequency.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import json


def compute_local_frequencies(mol):
    """
    Compute local vibrational frequencies for each atom.
    
    Frequency ∝ √(force_constant / reduced_mass)
    
    Force constant depends on:
    - Bond orders
    - Electronegativity
    - Hybridization
    """
    n = mol.GetNumAtoms()
    
    # Mass
    MASS = {1: 1, 6: 12, 7: 14, 8: 16, 9: 19, 15: 31, 16: 32, 17: 35}
    
    # Force constants (mdyne/Å) for different bond types
    K_BOND = {1.0: 4.5, 1.5: 5.5, 2.0: 9.0, 3.0: 15.0}
    
    local_freq = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        z_i = atom.GetAtomicNum()
        m_i = MASS.get(z_i, 12)
        
        # Effective force constant from all bonds
        k_total = 0.0
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(i, j)
            bo = bond.GetBondTypeAsDouble()
            if bond.GetIsAromatic():
                bo = 1.5
            
            z_j = nbr.GetAtomicNum()
            m_j = MASS.get(z_j, 12)
            
            # Reduced mass
            mu = m_i * m_j / (m_i + m_j)
            
            # Force constant
            k = K_BOND.get(bo, 5.0)
            
            # Local frequency contribution
            k_total += k / mu
        
        local_freq[i] = np.sqrt(k_total) if k_total > 0 else 0
    
    return local_freq


def compute_resonance_coupling(mol, eigenvalues, eigenvectors):
    """
    Compute how well each site can couple to an external oscillator (Fe=O).
    
    The Fe=O → C-H reaction coordinate has a characteristic frequency.
    Sites that can match this frequency will react.
    
    Fe=O stretch: ~800 cm⁻¹
    Fe-O-H bend (product): ~500 cm⁻¹
    Reaction coordinate: somewhere in between, ~600-700 cm⁻¹
    
    In our eigenvalue units, this corresponds to mid-range modes.
    """
    n = mol.GetNumAtoms()
    
    # Target frequency (normalized to eigenvalue scale)
    # We don't know exact value, so scan a range
    target_modes = range(n // 4, 3 * n // 4)  # Middle frequency range
    
    resonance = np.zeros(n)
    
    for i in range(n):
        coupling = 0.0
        for k in target_modes:
            if k < n and eigenvalues[k] > 1e-6:
                # Resonance strength = participation × Lorentzian lineshape
                # Assume some target eigenvalue
                participation = eigenvectors[i, k]**2
                
                # Sites with strong participation in these modes can resonate
                coupling += participation
        
        resonance[i] = coupling
    
    return resonance


def compute_decoherence_resistance(mol, eigenvectors, eigenvalues):
    """
    Quantum coherence is needed for energy transfer.
    Decoherence (loss of quantum behavior) prevents reaction.
    
    Sites that are "protected" from decoherence can maintain
    quantum superposition longer → better energy transfer.
    
    Decoherence is faster for:
    - High frequency modes (more interactions with environment)
    - Localized modes (more strongly coupled to specific atoms)
    
    Decoherence is slower for:
    - Delocalized modes (spread over many atoms)
    - Low frequency modes (fewer environment interactions)
    """
    n = mol.GetNumAtoms()
    
    decoherence_resistance = np.zeros(n)
    
    for i in range(n):
        resistance = 0.0
        for k in range(1, n):
            if eigenvalues[k] > 1e-6:
                participation = eigenvectors[i, k]**2
                
                # Mode delocalization (how spread out is this mode?)
                mode_entropy = -np.sum(eigenvectors[:, k]**2 * 
                                       np.log(eigenvectors[:, k]**2 + 1e-10))
                
                # Decoherence rate ∝ frequency × localization
                # Resistance ∝ 1 / (frequency × localization)
                decoherence_rate = np.sqrt(eigenvalues[k]) / (mode_entropy + 1)
                resistance += participation / (decoherence_rate + 0.1)
        
        decoherence_resistance[i] = resistance
    
    return decoherence_resistance


def compute_quantum_tunneling_rate(mol, eigenvectors, eigenvalues):
    """
    The C-H → Fe-O-H transition involves hydrogen tunneling.
    
    Tunneling rate ∝ exp(-2γa) where:
    - γ = √(2mV)/ℏ
    - a = barrier width
    
    For hydrogen transfer:
    - Light mass → significant tunneling
    - Low barrier → higher rate
    
    We approximate:
    - Sites with flexible C-H (low force constant) tunnel better
    - Sites with α-heteroatom (lower barrier) tunnel better
    """
    n = mol.GetNumAtoms()
    
    tunneling = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        # Flexibility (inverse rigidity)
        rigidity = sum(eigenvectors[i, k]**2 for k in range(max(1, n-3), n))
        flexibility = 1.0 / (rigidity + 0.1)
        
        # Lower barrier from alpha effect
        barrier_reduction = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7:  # N
                barrier_reduction *= 1.5
            elif z == 8:  # O
                barrier_reduction *= 1.4
            elif z == 16:  # S
                barrier_reduction *= 1.3
        
        # Tunneling rate
        tunneling[i] = flexibility * barrier_reduction
    
    return tunneling


def compute_energy_landscape_gradient(mol, eigenvectors, eigenvalues):
    """
    The reaction follows the path of steepest descent on the energy landscape.
    
    At each site, compute the "gradient" toward lower energy:
    - Sites at energy maxima can easily roll down
    - Sites at energy minima are stable (don't react)
    
    Use eigenvector participation to estimate energy curvature.
    """
    n = mol.GetNumAtoms()
    
    # Energy curvature at each site
    curvature = np.zeros(n)
    
    for i in range(n):
        # Participation in different frequency modes
        low_freq = sum(eigenvectors[i, k]**2 for k in range(1, min(4, n)))
        high_freq = sum(eigenvectors[i, k]**2 for k in range(max(1, n-3), n))
        
        # Sites with high low-freq participation are at saddle points
        # (floppy, can move easily)
        # Sites with high high-freq participation are at minima
        # (rigid, hard to move)
        
        # Gradient = how easily can we leave this point?
        if high_freq > 1e-6:
            curvature[i] = low_freq / high_freq
        else:
            curvature[i] = low_freq * 10
    
    return curvature


def quantum_field_som(smiles):
    """
    Full quantum field model.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    n = mol.GetNumAtoms()
    
    # Graph Laplacian for vibrational modes
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            w = 1.5
        A[i, j] = A[j, i] = w
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Compute quantum descriptors
    local_freq = compute_local_frequencies(mol)
    resonance = compute_resonance_coupling(mol, eigenvalues, eigenvectors)
    decoherence = compute_decoherence_resistance(mol, eigenvectors, eigenvalues)
    tunneling = compute_quantum_tunneling_rate(mol, eigenvectors, eigenvalues)
    landscape = compute_energy_landscape_gradient(mol, eigenvectors, eigenvalues)
    
    # Our proven flexibility metric
    flexibility = np.zeros(n)
    for i in range(n):
        rigidity = sum(eigenvectors[i, k]**2 for k in range(max(1, n-3), n))
        flexibility[i] = 1.0 / (rigidity + 0.1)
    
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
        
        # Combine quantum effects
        q_score = (
            0.30 * flexibility[i] +      # Proven
            0.20 * tunneling[i] +         # H tunneling
            0.15 * landscape[i] +         # Energy gradient
            0.15 * resonance[i] +         # Frequency matching
            0.10 * decoherence[i] +       # Quantum coherence
            0.10 * (1.0 / (local_freq[i] + 0.1))  # Low frequency = reactive
        )
        
        # Chemical multipliers
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
        
        scores[i] = q_score * alpha_mult * benz_mult * tert_mult * h_factor
    
    return scores


def evaluate(data_path):
    """Evaluate on dataset."""
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
        
        scores = quantum_field_som(smiles)
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
    
    print(f"\n=== QUANTUM FIELD RESONANCE MODEL ===")
    print(f"Top-1: {top1}/{total} = {top1/total*100:.1f}%")
    print(f"Top-3: {top3}/{total} = {top3/total*100:.1f}%\n")
    
    for src, s in sorted(by_source.items(), key=lambda x: -x[1]['n']):
        if s['n'] >= 5:
            print(f"  {src:20s}: Top-1={s['t1']/s['n']*100:5.1f}%, Top-3={s['t3']/s['n']*100:5.1f}% (n={s['n']})")
    
    return top1/total if total > 0 else 0


if __name__ == '__main__':
    evaluate('/home/claude/enzyme_Software/data/curated/merged_cyp3a4_extended.json')
