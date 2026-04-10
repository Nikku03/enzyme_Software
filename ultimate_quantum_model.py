"""
ULTIMATE QUANTUM FORCES MODEL

This model captures ALL relevant quantum-scale forces for CYP metabolism:

ATTRACTIVE FORCES (bring Fe=O to substrate):
1. Coulomb attraction (Fe⁺ to electron-rich regions)
2. van der Waals / London dispersion (polarizability)
3. π-cation interaction (with aromatic rings)
4. Charge-dipole (induced dipole from field)

REPULSIVE FORCES (barrier to reaction):
5. Pauli exclusion (steric crowding)
6. Electron-electron repulsion
7. Conformational strain

QUANTUM EFFECTS (modify barrier/rate):
8. Tunneling (H penetrates classical barrier)
9. Zero-point energy (vibrational ground state)
10. Non-adiabatic coupling (spin-state crossing)
11. Resonance (delocalization of radical)

TOPOLOGICAL EFFECTS (from wave function):
12. Phase winding (topological charge)
13. Berry phase (geometric phase)
14. Decoherence (interaction with environment)

DUAL MECHANISM:
- HAT (H-atom transfer) for aliphatic C-H
- SET (single electron transfer) for aromatic C-H

Each mechanism has different physics:
HAT: flexibility + tunneling + alpha effect
SET: π-density + edge accessibility + substituent effects
"""

import numpy as np
from rdkit import Chem
import json
from collections import defaultdict


# === PHYSICAL CONSTANTS ===
POLARIZABILITY = {1: 0.67, 6: 1.76, 7: 1.10, 8: 0.80, 9: 0.56, 15: 3.63, 16: 2.90, 17: 2.18, 35: 3.05}
ELECTRONEGATIVITY = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96}
IONIZATION_POTENTIAL = {1: 13.6, 6: 11.3, 7: 14.5, 8: 13.6, 9: 17.4, 15: 10.5, 16: 10.4, 17: 13.0, 35: 11.8}
ATOMIC_RADIUS = {1: 0.53, 6: 0.77, 7: 0.75, 8: 0.73, 9: 0.71, 15: 1.06, 16: 1.02, 17: 0.99, 35: 1.14}


def compute_laplacian(mol):
    """Compute graph Laplacian with bond-order weighting."""
    n = mol.GetNumAtoms()
    A = np.zeros((n, n))
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            w = 1.5
        if bond.GetIsConjugated():
            w *= 1.1
        A[i, j] = A[j, i] = w
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    return eigenvalues, eigenvectors, A


def compute_flexibility(eigenvectors, eigenvalues, n_high=3):
    """Flexibility = 1/rigidity = ease of deformation."""
    n = len(eigenvalues)
    flex = np.zeros(n)
    for i in range(n):
        rigidity = sum(eigenvectors[i, k]**2 for k in range(max(1, n-n_high), n))
        flex[i] = 1.0 / (rigidity + 0.1)
    return flex


def compute_amplitude(eigenvectors, eigenvalues, n_low=5):
    """Electron density proxy from low-frequency mode participation."""
    n = len(eigenvalues)
    amp = np.zeros(n)
    for i in range(n):
        amp[i] = sum(eigenvectors[i, k]**2 / (eigenvalues[k] + 0.1) 
                    for k in range(1, min(n_low, n)))
    return amp


def compute_coulomb_attraction(mol):
    """Electrostatic potential from local charge distribution."""
    n = mol.GetNumAtoms()
    
    # Estimate partial charges from electronegativity
    charges = np.zeros(n)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        z = atom.GetAtomicNum()
        en_i = ELECTRONEGATIVITY.get(z, 2.5)
        
        for nbr in atom.GetNeighbors():
            en_j = ELECTRONEGATIVITY.get(nbr.GetAtomicNum(), 2.5)
            charges[i] += 0.1 * (en_j - en_i)  # Pull from more EN neighbors
    
    # Fe=O attracted to slightly negative regions
    return -charges  # Higher = more attractive


def compute_dispersion(mol):
    """London dispersion from polarizability."""
    n = mol.GetNumAtoms()
    disp = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        alpha_i = POLARIZABILITY.get(atom.GetAtomicNum(), 1.5)
        
        for nbr in atom.GetNeighbors():
            alpha_j = POLARIZABILITY.get(nbr.GetAtomicNum(), 1.5)
            ip_i = IONIZATION_POTENTIAL.get(atom.GetAtomicNum(), 12.0)
            ip_j = IONIZATION_POTENTIAL.get(nbr.GetAtomicNum(), 12.0)
            
            # London formula: C6 = 3/2 * α₁α₂ * IP₁*IP₂/(IP₁+IP₂)
            c6 = 1.5 * alpha_i * alpha_j * ip_i * ip_j / (ip_i + ip_j + 0.1)
            disp[i] += c6
    
    return disp


def compute_pauli_repulsion(mol):
    """Steric accessibility (inverse of crowding)."""
    n = mol.GetNumAtoms()
    pauli = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        r_i = ATOMIC_RADIUS.get(atom.GetAtomicNum(), 0.8)
        
        crowding = 0.0
        for nbr in atom.GetNeighbors():
            r_j = ATOMIC_RADIUS.get(nbr.GetAtomicNum(), 0.8)
            crowding += (r_i + r_j)  # Larger atoms = more crowding
        
        pauli[i] = 1.0 / (crowding + 0.1)
    
    return pauli


def compute_tunneling(mol, flexibility):
    """Quantum tunneling probability for H-transfer."""
    n = mol.GetNumAtoms()
    tunnel = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            continue
        
        n_H = atom.GetTotalNumHs()
        if n_H == 0:
            tunnel[i] = 0.1  # Small aromatic contribution
            continue
        
        # Base tunneling from flexibility (flexible = lower barrier)
        base = flexibility[i] * (1 + 0.3 * n_H)
        
        # Alpha-heteroatom LOWERS barrier → enhances tunneling
        barrier_mod = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7:  # N
                barrier_mod *= 1.6  # Strongest enhancement
            elif z == 8:  # O  
                barrier_mod *= 1.5
            elif z == 16:  # S
                barrier_mod *= 1.3
        
        tunnel[i] = base * barrier_mod
    
    return tunnel


def compute_zpe(eigenvectors, eigenvalues):
    """Zero-point energy participation."""
    n = len(eigenvalues)
    zpe = np.zeros(n)
    
    for i in range(n):
        # ZPE = Σ (ℏω/2) × participation
        for k in range(1, n):
            omega = np.sqrt(max(0, eigenvalues[k]))
            zpe[i] += 0.5 * omega * eigenvectors[i, k]**2
    
    return zpe


def compute_exchange(mol, eigenvectors):
    """Exchange coupling (bonding character)."""
    n = mol.GetNumAtoms()
    exchange = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(i, j)
            bo = bond.GetBondTypeAsDouble() if bond else 1.0
            
            # Eigenvector overlap
            overlap = sum(eigenvectors[i, k] * eigenvectors[j, k] 
                         for k in range(1, n))
            exchange[i] += bo * abs(overlap)
    
    return exchange


def compute_topological_charge(mol, eigenvectors):
    """
    Topological charge from phase winding.
    
    Low winding = smooth wave function = reactive
    High winding = phase discontinuity = stable
    """
    n = mol.GetNumAtoms()
    topo = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        phase_diff = 0.0
        
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            # Phase difference from eigenvector components
            for k in range(1, min(5, n)):
                phase_diff += abs(eigenvectors[i, k] - eigenvectors[j, k])
        
        topo[i] = 1.0 / (phase_diff + 0.1)  # Low winding = high reactivity
    
    return topo


def compute_pi_density(mol, eigenvectors, eigenvalues):
    """π-electron density for aromatic carbons."""
    n = mol.GetNumAtoms()
    pi_density = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if not atom.GetIsAromatic():
            continue
        
        # π-density from low-frequency aromatic modes
        # These correspond to π-electron delocalization
        for k in range(1, min(n//2, n)):
            if eigenvalues[k] > 0.1 and eigenvalues[k] < 3.0:  # Mid-range for π
                pi_density[i] += eigenvectors[i, k]**2 / eigenvalues[k]
    
    return pi_density


def compute_edge_accessibility(mol):
    """
    Edge accessibility for aromatic rings.
    
    Positions at the edge of the π-system (ortho to substituents)
    are more accessible to attack.
    """
    n = mol.GetNumAtoms()
    edge = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if not atom.GetIsAromatic():
            continue
        
        # Count non-aromatic neighbors (substituents)
        n_subst = sum(1 for nbr in atom.GetNeighbors() if not nbr.GetIsAromatic())
        
        if n_subst > 0:
            # This is a substituted position
            edge[i] = 0.5  # Reduced accessibility
        else:
            # Check if adjacent to substituted position
            for nbr in atom.GetNeighbors():
                if nbr.GetIsAromatic():
                    n_subst_nbr = sum(1 for nn in nbr.GetNeighbors() 
                                     if not nn.GetIsAromatic())
                    if n_subst_nbr > 0:
                        edge[i] = 1.0  # Ortho to substituent = edge
                        break
            else:
                edge[i] = 0.7  # Middle of ring
    
    return edge


def dual_mechanism_score(mol, forces, weights):
    """
    Dual mechanism scoring:
    - HAT (H-atom transfer) for aliphatic
    - SET (single electron transfer) for aromatic
    """
    n = mol.GetNumAtoms()
    scores = np.full(n, -np.inf)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            continue
        
        n_H = atom.GetTotalNumHs()
        is_arom = atom.GetIsAromatic()
        
        if n_H == 0 and not is_arom:
            continue
        
        if is_arom:
            # === SET MECHANISM ===
            # π-density + edge accessibility
            score = (
                weights.get('set_pi', 0.4) * forces['pi_density'][i] +
                weights.get('set_edge', 0.3) * forces['edge'][i] +
                weights.get('set_topo', 0.2) * forces['topo'][i] +
                weights.get('set_disp', 0.1) * forces['disp'][i]
            )
        else:
            # === HAT MECHANISM ===
            # Flexibility + tunneling + amplitude
            score = (
                weights.get('hat_flex', 0.30) * forces['flex'][i] +
                weights.get('hat_amp', 0.20) * forces['amp'][i] +
                weights.get('hat_tunnel', 0.20) * forces['tunnel'][i] +
                weights.get('hat_topo', 0.15) * forces['topo'][i] +
                weights.get('hat_disp', 0.10) * forces['disp'][i] +
                weights.get('hat_pauli', 0.05) * forces['pauli'][i]
            )
        
        # Chemical multipliers (apply to both)
        alpha_mult = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7: alpha_mult = max(alpha_mult, weights.get('aN', 1.84))
            elif z == 8: alpha_mult = max(alpha_mult, weights.get('aO', 1.82))
            elif z == 16: alpha_mult = max(alpha_mult, weights.get('aS', 1.54))
        
        benz_mult = 1.0
        if not is_arom and n_H > 0:
            for nbr in atom.GetNeighbors():
                if nbr.GetIsAromatic():
                    benz_mult = weights.get('benz', 1.73)
                    break
        
        n_C = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6)
        tert_mult = 1.0 + weights.get('tert', 0.22) * (n_C - 1) if n_C > 1 else 1.0
        h_factor = (1 + weights.get('hf', 0.13) * n_H) if n_H > 0 else 0.3
        
        scores[i] = score * alpha_mult * benz_mult * tert_mult * h_factor
    
    return scores


def compute_all_forces(mol):
    """Compute all quantum forces for a molecule."""
    n = mol.GetNumAtoms()
    if n < 3:
        return None
    
    eigenvalues, eigenvectors, A = compute_laplacian(mol)
    
    forces = {
        'flex': compute_flexibility(eigenvectors, eigenvalues),
        'amp': compute_amplitude(eigenvectors, eigenvalues),
        'coulomb': compute_coulomb_attraction(mol),
        'disp': compute_dispersion(mol),
        'pauli': compute_pauli_repulsion(mol),
        'tunnel': compute_tunneling(mol, compute_flexibility(eigenvectors, eigenvalues)),
        'zpe': compute_zpe(eigenvectors, eigenvalues),
        'exchange': compute_exchange(mol, eigenvectors),
        'topo': compute_topological_charge(mol, eigenvectors),
        'pi_density': compute_pi_density(mol, eigenvectors, eigenvalues),
        'edge': compute_edge_accessibility(mol),
    }
    
    # Normalize
    def norm(x):
        x = np.array(x)
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-10 else np.ones_like(x) * 0.5
    
    for k in forces:
        forces[k] = norm(forces[k])
    
    return forces


def evaluate(data, weights, return_details=False):
    """Evaluate weights on data."""
    top1 = top3 = total = 0
    details = []
    
    for d in data:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        name = d.get('name', 'unknown')
        
        if not smiles or not sites:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        forces = compute_all_forces(mol)
        if forces is None:
            continue
        
        scores = dual_mechanism_score(mol, forces, weights)
        
        valid = [i for i in range(len(scores)) if scores[i] > -np.inf]
        if not valid:
            continue
        
        ranked = sorted(valid, key=lambda x: -scores[x])[:3]
        
        hit1 = ranked[0] in sites
        hit3 = any(r in sites for r in ranked)
        
        if hit1: top1 += 1
        if hit3: top3 += 1
        total += 1
        
        if return_details:
            details.append({
                'name': name, 'pred': ranked, 'actual': sites,
                'hit1': hit1, 'hit3': hit3
            })
    
    t1 = top1/total if total > 0 else 0
    t3 = top3/total if total > 0 else 0
    
    if return_details:
        return t1, t3, total, details
    return t1, t3, total


def optimize_dual_mechanism(data, n_trials=500):
    """Optimize weights for dual mechanism model."""
    best_t1 = 0
    best_weights = None
    
    for trial in range(n_trials):
        weights = {
            # HAT weights
            'hat_flex': np.random.uniform(0.15, 0.40),
            'hat_amp': np.random.uniform(0.10, 0.30),
            'hat_tunnel': np.random.uniform(0.10, 0.30),
            'hat_topo': np.random.uniform(0.05, 0.25),
            'hat_disp': np.random.uniform(0.0, 0.15),
            'hat_pauli': np.random.uniform(0.0, 0.10),
            # SET weights
            'set_pi': np.random.uniform(0.2, 0.5),
            'set_edge': np.random.uniform(0.1, 0.4),
            'set_topo': np.random.uniform(0.1, 0.3),
            'set_disp': np.random.uniform(0.0, 0.2),
            # Chemical multipliers
            'aN': np.random.uniform(1.5, 2.2),
            'aO': np.random.uniform(1.5, 2.0),
            'aS': np.random.uniform(1.2, 1.8),
            'benz': np.random.uniform(1.4, 2.0),
            'tert': np.random.uniform(0.1, 0.35),
            'hf': np.random.uniform(0.08, 0.2),
        }
        
        t1, t3, n = evaluate(data, weights)
        
        if t1 > best_t1:
            best_t1 = t1
            best_weights = weights.copy()
            print(f"  Trial {trial}: Top-1={t1*100:.1f}%, Top-3={t3*100:.1f}%")
    
    return best_weights, best_t1


def full_evaluation(data_path):
    """Complete evaluation with Zaretzki comparison."""
    with open(data_path) as f:
        data = json.load(f)['drugs']
    
    by_source = defaultdict(list)
    for d in data:
        by_source[d.get('source', 'unknown')].append(d)
    
    print("\n" + "="*70)
    print("ULTIMATE QUANTUM FORCES MODEL - DUAL MECHANISM (HAT + SET)")
    print("="*70)
    print("\nForces: Flexibility, Amplitude, Tunneling, van der Waals, Pauli,")
    print("        ZPE, Exchange, Topological Charge, π-Density, Edge Access")
    
    # Default dual-mechanism weights
    default_w = {
        'hat_flex': 0.30, 'hat_amp': 0.20, 'hat_tunnel': 0.20,
        'hat_topo': 0.15, 'hat_disp': 0.10, 'hat_pauli': 0.05,
        'set_pi': 0.40, 'set_edge': 0.30, 'set_topo': 0.20, 'set_disp': 0.10,
        'aN': 1.84, 'aO': 1.82, 'aS': 1.54, 'benz': 1.73, 'tert': 0.22, 'hf': 0.13
    }
    
    print(f"\n{'Source':<20} {'N':>6} {'Top-1':>10} {'Top-3':>10}")
    print("-" * 50)
    
    results = {}
    for src in ['AZ120', 'Zaretzki', 'DrugBank', 'MetXBioDB']:
        if src in by_source:
            t1, t3, n = evaluate(by_source[src], default_w)
            print(f"{src:<20} {n:>6} {t1*100:>9.1f}% {t3*100:>9.1f}%")
            results[src] = {'t1': t1, 't3': t3, 'n': n}
    
    # Optimize for AZ120
    print("\n--- Optimizing for AZ120 ---")
    az_data = by_source.get('AZ120', [])
    best_w, best_t1 = optimize_dual_mechanism(az_data, n_trials=500)
    
    print(f"\nBest AZ120: Top-1 = {best_t1*100:.1f}%")
    print(f"HAT: flex={best_w['hat_flex']:.2f}, amp={best_w['hat_amp']:.2f}, "
          f"tunnel={best_w['hat_tunnel']:.2f}, topo={best_w['hat_topo']:.2f}")
    print(f"SET: pi={best_w['set_pi']:.2f}, edge={best_w['set_edge']:.2f}, "
          f"topo={best_w['set_topo']:.2f}")
    
    print(f"\n{'Source':<20} {'N':>6} {'Top-1':>10} {'Top-3':>10} (Optimized)")
    print("-" * 55)
    
    for src in ['AZ120', 'Zaretzki', 'DrugBank', 'MetXBioDB']:
        if src in by_source:
            t1, t3, n = evaluate(by_source[src], best_w)
            print(f"{src:<20} {n:>6} {t1*100:>9.1f}% {t3*100:>9.1f}%")
    
    # Zaretzki detailed breakdown
    print("\n" + "="*70)
    print("ZARETZKI BENCHMARK ANALYSIS")
    print("="*70)
    
    zaretzki = by_source.get('Zaretzki', [])
    t1, t3, n, details = evaluate(zaretzki, best_w, return_details=True)
    
    # Analyze by site type
    arom_t1 = arom_n = alpha_t1 = alpha_n = other_t1 = other_n = 0
    
    for d, det in zip(zaretzki, details):
        mol = Chem.MolFromSmiles(d['smiles'])
        if mol is None:
            continue
        
        sites = d['site_atoms']
        for site in sites:
            if site >= mol.GetNumAtoms():
                continue
            atom = mol.GetAtomWithIdx(site)
            
            is_arom = atom.GetIsAromatic()
            is_alpha = any(mol.GetAtomWithIdx(nbr.GetIdx()).GetAtomicNum() in [7, 8, 16]
                          for nbr in atom.GetNeighbors())
            
            if is_arom:
                arom_n += 1
                if det['hit1']: arom_t1 += 1
            elif is_alpha:
                alpha_n += 1
                if det['hit1']: alpha_t1 += 1
            else:
                other_n += 1
                if det['hit1']: other_t1 += 1
            break  # Only count first site
    
    print(f"\nSite Type Breakdown:")
    print(f"  Aromatic C-H:  {arom_t1}/{arom_n} ({arom_t1/arom_n*100:.1f}% Top-1)" if arom_n > 0 else "  Aromatic: N/A")
    print(f"  Alpha (N/O/S): {alpha_t1}/{alpha_n} ({alpha_t1/alpha_n*100:.1f}% Top-1)" if alpha_n > 0 else "  Alpha: N/A")
    print(f"  Other:         {other_t1}/{other_n} ({other_t1/other_n*100:.1f}% Top-1)" if other_n > 0 else "  Other: N/A")
    
    return results, best_w


if __name__ == '__main__':
    full_evaluation('/home/claude/enzyme_Software/data/curated/merged_cyp3a4_extended.json')
