"""
QUANTUM WAVE INTERFERENCE MODEL FOR SoM

At the quantum scale:
- No solid objects, only wave functions ψ(r,t)
- No fixed bonds, only regions of electron density
- Reactivity = where waves can reorganize

The Fe=O of CYP is itself a wave. The substrate is a wave.
Bond breaking occurs where:
1. The waves can INTERFERE constructively
2. Energy can TRANSFER between wave modes
3. The resulting state is LOWER energy than initial

We model this as:
- Each atom contributes a spherical wave
- Waves interfere according to their phase relationships
- Reaction occurs where interference creates a pathway for electron flow
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import json
from scipy import linalg


def compute_wave_field(mol):
    """
    Model each atom as a source of a quantum wave.
    
    ψ_i(r) ∝ exp(-α|r - r_i|²) × Y_lm(θ,φ)
    
    For simplicity, use s-orbital-like waves (spherical).
    The TOTAL wave function is a superposition.
    """
    n = mol.GetNumAtoms()
    
    # Get 3D coordinates (or generate them)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        conf = mol.GetConformer()
        coords = np.array([conf.GetAtomPosition(i) for i in range(n)])
    except:
        # Fallback to 2D
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x, 
                           conf.GetAtomPosition(i).y, 0] for i in range(n)])
    
    # Slater exponents (controls wave decay)
    ZETA = {1: 1.24, 6: 1.72, 7: 1.95, 8: 2.25, 9: 2.60, 
            15: 1.60, 16: 2.12, 17: 2.36, 35: 2.69}
    
    # Compute wave amplitude at each atomic position from ALL other atoms
    # This gives us the "wave density" at each site
    wave_amplitude = np.zeros(n)
    wave_interference = np.zeros(n)
    
    for i in range(n):
        atom_i = mol.GetAtomWithIdx(i)
        z_i = atom_i.GetAtomicNum()
        zeta_i = ZETA.get(z_i, 1.5)
        
        total_wave = 0.0
        interference = 0.0
        
        for j in range(n):
            if i == j:
                continue
            
            atom_j = mol.GetAtomWithIdx(j)
            z_j = atom_j.GetAtomicNum()
            zeta_j = ZETA.get(z_j, 1.5)
            
            # Distance
            r_ij = np.linalg.norm(coords[i] - coords[j])
            
            # Wave from atom j at position i
            # ψ_j(r_i) = exp(-zeta_j × r_ij)
            psi_j = np.exp(-zeta_j * r_ij)
            
            # Phase factor (depends on bond type if bonded)
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond:
                # Bonded: in-phase (bonding orbital)
                phase = 1.0
                if bond.GetIsAromatic():
                    phase = 0.5  # Partial bonding character
            else:
                # Non-bonded: phase depends on path length
                # Through 2 bonds: antibonding-like
                phase = -0.3 if r_ij < 4.0 else 0.0
            
            total_wave += psi_j
            interference += phase * psi_j
        
        wave_amplitude[i] = total_wave
        wave_interference[i] = interference
    
    return coords, wave_amplitude, wave_interference


def compute_energy_flow_susceptibility(mol, coords):
    """
    Where can energy flow through the molecule?
    
    Model energy transfer as phonon-mediated:
    - Vibrations carry energy
    - Some sites are "hubs" for energy flow
    - Bond breaking requires energy concentration at one site
    
    Use graph Laplacian eigenvectors as phonon modes.
    Energy flow susceptibility = how easily a site can receive energy.
    """
    n = mol.GetNumAtoms()
    
    # Build mass-weighted Laplacian (like in normal mode analysis)
    MASS = {1: 1, 6: 12, 7: 14, 8: 16, 9: 19, 15: 31, 16: 32, 17: 35}
    
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Force constant (higher for stronger bonds)
        bo = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            bo = 1.5
        k = bo * 500  # mdyne/Å typical
        
        A[i, j] = A[j, i] = k
    
    # Mass matrix
    M = np.zeros((n, n))
    for i in range(n):
        z = mol.GetAtomWithIdx(i).GetAtomicNum()
        M[i, i] = MASS.get(z, 12)
    
    # Mass-weighted Laplacian
    D = np.diag(A.sum(axis=1))
    L = D - A
    M_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(M) + 1e-6))
    L_mw = M_inv_sqrt @ L @ M_inv_sqrt
    
    eigenvalues, eigenvectors = np.linalg.eigh(L_mw)
    
    # Energy flow susceptibility:
    # Sites that participate strongly in LOW frequency modes
    # can easily receive/transmit energy (like a resonator)
    susceptibility = np.zeros(n)
    for i in range(n):
        # Low frequency modes (first few non-zero)
        for k in range(1, min(6, n)):
            if eigenvalues[k] > 1e-6:
                # Participation weighted by 1/frequency
                susceptibility[i] += eigenvectors[i, k]**2 / np.sqrt(eigenvalues[k])
    
    return susceptibility, eigenvalues, eigenvectors


def compute_vacuum_fluctuation_coupling(mol, coords, eigenvectors, eigenvalues):
    """
    Quantum vacuum fluctuations can trigger transitions.
    
    The zero-point energy of each mode is ℏω/2.
    Sites coupled to high zero-point energy modes have more
    "quantum noise" that can trigger reactions.
    
    This is related to quantum tunneling - fluctuations
    provide the "kick" to cross the barrier.
    """
    n = mol.GetNumAtoms()
    
    fluctuation_coupling = np.zeros(n)
    
    for i in range(n):
        coupling = 0.0
        for k in range(1, n):
            if eigenvalues[k] > 1e-6:
                omega = np.sqrt(eigenvalues[k])  # frequency
                zpe = 0.5 * omega  # zero-point energy (in ℏ units)
                
                # Coupling to this mode
                coupling += eigenvectors[i, k]**2 * zpe
        
        fluctuation_coupling[i] = coupling
    
    return fluctuation_coupling


def compute_wavefunction_collapse_probability(mol, coords, wave_amp, wave_interf, 
                                               energy_flow, fluctuations):
    """
    The "measurement" that triggers bond breaking is the Fe=O approaching.
    
    When Fe=O gets close to a C-H bond, it acts as a measurement:
    - The wave function collapses
    - Electron density localizes
    - Energy transfers
    
    Collapse probability depends on:
    1. Wave amplitude (how much ψ is there?)
    2. Interference pattern (constructive = higher probability)
    3. Energy flow (can energy reach this site?)
    4. Fluctuations (is there quantum noise to trigger collapse?)
    """
    n = mol.GetNumAtoms()
    
    # Normalize each factor
    def normalize(x):
        x = np.array(x)
        if x.max() - x.min() > 1e-10:
            return (x - x.min()) / (x.max() - x.min() + 1e-10)
        return np.ones_like(x) * 0.5
    
    wave_amp_norm = normalize(wave_amp)
    wave_interf_norm = normalize(wave_interf)
    energy_flow_norm = normalize(energy_flow)
    fluctuations_norm = normalize(fluctuations)
    
    # Collapse probability (before chemical knowledge)
    collapse_prob = (
        0.25 * wave_amp_norm +
        0.25 * wave_interf_norm +
        0.30 * energy_flow_norm +
        0.20 * fluctuations_norm
    )
    
    return collapse_prob


def quantum_wave_som(smiles):
    """
    Full quantum wave model for SoM prediction.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = Chem.AddHs(mol)
    n = mol.GetNumAtoms()
    
    # 1. Compute wave field
    coords, wave_amp, wave_interf = compute_wave_field(mol)
    
    # 2. Compute energy flow susceptibility
    energy_flow, eigenvalues, eigenvectors = compute_energy_flow_susceptibility(mol, coords)
    
    # 3. Compute vacuum fluctuation coupling
    fluctuations = compute_vacuum_fluctuation_coupling(mol, coords, eigenvectors, eigenvalues)
    
    # 4. Compute collapse probability
    collapse_prob = compute_wavefunction_collapse_probability(
        mol, coords, wave_amp, wave_interf, energy_flow, fluctuations
    )
    
    # Remove Hs and map back
    mol_noH = Chem.RemoveHs(mol)
    n_noH = mol_noH.GetNumAtoms()
    
    # Map heavy atom indices
    heavy_indices = [i for i in range(n) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    
    scores = np.full(n_noH, -np.inf)
    
    for idx_noH, idx_H in enumerate(heavy_indices[:n_noH]):
        atom = mol.GetAtomWithIdx(idx_H)
        z = atom.GetAtomicNum()
        
        if z != 6:
            continue
        
        # Check if this C has H attached (in the H-added molecule)
        n_H = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
        is_arom = atom.GetIsAromatic()
        
        if n_H == 0 and not is_arom:
            continue
        
        # Base quantum score
        q_score = collapse_prob[idx_H]
        
        # === CHEMICAL KNOWLEDGE MULTIPLIERS ===
        # These represent the ENERGY BARRIER modifications
        
        # Alpha-heteroatom (lone pair lowers barrier)
        alpha_mult = 1.0
        for nbr in atom.GetNeighbors():
            nbr_z = nbr.GetAtomicNum()
            if nbr_z == 7:
                alpha_mult = max(alpha_mult, 1.84)
            elif nbr_z == 8:
                alpha_mult = max(alpha_mult, 1.82)
            elif nbr_z == 16:
                alpha_mult = max(alpha_mult, 1.54)
        
        # Benzylic (resonance stabilization of TS)
        benz_mult = 1.0
        if not is_arom and n_H > 0:
            for nbr in atom.GetNeighbors():
                if nbr.GetIsAromatic():
                    benz_mult = 1.73
                    break
        
        # Tertiary (more stable radical)
        n_C_nbrs = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6)
        tert_mult = 1.0 + 0.22 * (n_C_nbrs - 1) if n_C_nbrs > 1 else 1.0
        
        # H-factor
        h_factor = (1 + 0.13 * n_H) if n_H > 0 else 0.3
        
        scores[idx_noH] = q_score * alpha_mult * benz_mult * tert_mult * h_factor
    
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
        
        scores = quantum_wave_som(smiles)
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
    
    print(f"\n=== QUANTUM WAVE INTERFERENCE MODEL ===")
    print(f"Top-1: {top1}/{total} = {top1/total*100:.1f}%")
    print(f"Top-3: {top3}/{total} = {top3/total*100:.1f}%\n")
    
    for src, s in sorted(by_source.items(), key=lambda x: -x[1]['n']):
        if s['n'] >= 5:
            print(f"  {src:20s}: Top-1={s['t1']/s['n']*100:5.1f}%, Top-3={s['t3']/s['n']*100:5.1f}% (n={s['n']})")
    
    return top1/total if total > 0 else 0


if __name__ == '__main__':
    evaluate('/home/claude/enzyme_Software/data/curated/merged_cyp3a4_extended.json')
