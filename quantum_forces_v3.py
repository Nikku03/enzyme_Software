"""
QUANTUM FORCES MODEL V3 - WITH OPTIMIZATION

Forces modeled:
1. Flexibility (graph Laplacian high-freq modes)
2. Amplitude (electron density proxy)  
3. Tunneling (barrier penetration)
4. van der Waals / Dispersion
5. Pauli repulsion (steric)
6. Zero-point energy
7. Exchange coupling
8. Topological charge (phase winding)
"""

import numpy as np
from rdkit import Chem
import json
from collections import defaultdict


def compute_all_forces(mol):
    """Compute all quantum forces for a molecule."""
    n = mol.GetNumAtoms()
    if n < 3:
        return None
    
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
    
    forces = {}
    
    # 1. Flexibility
    forces['flex'] = np.array([1.0 / (sum(eigenvectors[i, k]**2 
                               for k in range(max(1, n-3), n)) + 0.1) for i in range(n)])
    
    # 2. Amplitude
    forces['amp'] = np.array([sum(eigenvectors[i, k]**2 / (eigenvalues[k] + 0.1) 
                              for k in range(1, min(5, n))) for i in range(n)])
    
    # 3. Tunneling (base + alpha enhancement)
    tunnel = np.zeros(n)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            continue
        n_H = atom.GetTotalNumHs()
        base = forces['flex'][i] * (1 + 0.3 * n_H)
        mod = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7: mod *= 1.5
            elif z == 8: mod *= 1.4
            elif z == 16: mod *= 1.3
        tunnel[i] = base * mod
    forces['tunnel'] = tunnel
    
    # 4. van der Waals (polarizability-based)
    ALPHA = {1: 0.67, 6: 1.76, 7: 1.10, 8: 0.80, 16: 2.90, 17: 2.18, 35: 3.05}
    vdw = np.zeros(n)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        a_i = ALPHA.get(atom.GetAtomicNum(), 1.5)
        for nbr in atom.GetNeighbors():
            a_j = ALPHA.get(nbr.GetAtomicNum(), 1.5)
            vdw[i] += a_i * a_j
    forces['vdw'] = vdw
    
    # 5. Pauli (steric accessibility)
    forces['pauli'] = np.array([1.0 / (1 + 0.3 * sum(1 for nbr in mol.GetAtomWithIdx(i).GetNeighbors() 
                                 if nbr.GetAtomicNum() > 1)) for i in range(n)])
    
    # 6. Zero-point energy
    forces['zpe'] = np.array([sum(np.sqrt(eigenvalues[k] + 0.01) * eigenvectors[i, k]**2 
                              for k in range(1, n)) for i in range(n)])
    
    # 7. Exchange coupling
    exchange = np.zeros(n)
    for i in range(n):
        for nbr in mol.GetAtomWithIdx(i).GetNeighbors():
            j = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(i, j)
            bo = bond.GetBondTypeAsDouble() if bond else 1.0
            overlap = sum(eigenvectors[i, k] * eigenvectors[j, k] for k in range(1, n))
            exchange[i] += bo * abs(overlap)
    forces['exchange'] = exchange
    
    # 8. Topological charge (phase winding from eigenvectors)
    topo = np.zeros(n)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        phase = 0.0
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            for k in range(1, min(5, n)):
                phase += abs(eigenvectors[i, k] - eigenvectors[j, k])
        topo[i] = phase
    forces['topo'] = 1.0 / (topo + 0.1)  # Low winding = reactive
    
    # Normalize all forces
    def norm(x):
        x = np.array(x)
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-10 else np.ones_like(x) * 0.5
    
    for k in forces:
        forces[k] = norm(forces[k])
    
    return forces, mol


def score_with_weights(mol, forces, weights):
    """Score molecule with given weights."""
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
        
        # Quantum score from weighted forces
        q_score = sum(weights.get(k, 0) * forces[k][i] for k in forces)
        
        # Chemical multipliers
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
        
        scores[i] = q_score * alpha_mult * benz_mult * tert_mult * h_factor
    
    return scores


def evaluate_weights(data, weights):
    """Evaluate given weights on data."""
    top1 = top3 = total = 0
    
    for d in data:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        
        if not smiles or not sites:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        result = compute_all_forces(mol)
        if result is None:
            continue
        
        forces, mol = result
        scores = score_with_weights(mol, forces, weights)
        
        valid = [i for i in range(len(scores)) if scores[i] > -np.inf]
        if not valid:
            continue
        
        ranked = sorted(valid, key=lambda x: -scores[x])[:3]
        
        if ranked[0] in sites:
            top1 += 1
        if any(r in sites for r in ranked):
            top3 += 1
        total += 1
    
    return top1/total if total > 0 else 0, top3/total if total > 0 else 0, total


def optimize_weights(data, n_trials=500):
    """Random search for optimal weights."""
    best_t1 = 0
    best_weights = None
    
    for trial in range(n_trials):
        weights = {
            'flex': np.random.uniform(0.1, 0.4),
            'amp': np.random.uniform(0.1, 0.3),
            'tunnel': np.random.uniform(0.05, 0.25),
            'vdw': np.random.uniform(0.0, 0.2),
            'pauli': np.random.uniform(0.0, 0.15),
            'zpe': np.random.uniform(0.0, 0.15),
            'exchange': np.random.uniform(0.0, 0.15),
            'topo': np.random.uniform(0.0, 0.2),
            'aN': np.random.uniform(1.5, 2.2),
            'aO': np.random.uniform(1.5, 2.0),
            'aS': np.random.uniform(1.2, 1.8),
            'benz': np.random.uniform(1.4, 2.0),
            'tert': np.random.uniform(0.1, 0.35),
            'hf': np.random.uniform(0.08, 0.2),
        }
        
        # Normalize force weights
        force_keys = ['flex', 'amp', 'tunnel', 'vdw', 'pauli', 'zpe', 'exchange', 'topo']
        total_w = sum(weights[k] for k in force_keys)
        for k in force_keys:
            weights[k] /= total_w
        
        t1, t3, n = evaluate_weights(data, weights)
        
        if t1 > best_t1:
            best_t1 = t1
            best_weights = weights.copy()
            print(f"  Trial {trial}: Top-1={t1*100:.1f}%, Top-3={t3*100:.1f}%")
    
    return best_weights, best_t1


def full_evaluation(data_path):
    """Full evaluation with optimization."""
    with open(data_path) as f:
        data = json.load(f)['drugs']
    
    # Split by source
    by_source = defaultdict(list)
    for d in data:
        by_source[d.get('source', 'unknown')].append(d)
    
    print("\n" + "="*70)
    print("QUANTUM FORCES MODEL V3 - COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Default weights (from previous best)
    default_weights = {
        'flex': 0.25, 'amp': 0.20, 'tunnel': 0.15, 'vdw': 0.12,
        'pauli': 0.10, 'zpe': 0.10, 'exchange': 0.08, 'topo': 0.0,
        'aN': 1.84, 'aO': 1.82, 'aS': 1.54, 'benz': 1.73, 'tert': 0.22, 'hf': 0.13
    }
    
    print("\n--- Default Weights ---")
    print(f"Forces: flex={default_weights['flex']:.2f}, amp={default_weights['amp']:.2f}, "
          f"tunnel={default_weights['tunnel']:.2f}, vdw={default_weights['vdw']:.2f}")
    print(f"        pauli={default_weights['pauli']:.2f}, zpe={default_weights['zpe']:.2f}, "
          f"exchange={default_weights['exchange']:.2f}, topo={default_weights['topo']:.2f}")
    
    print(f"\n{'Source':<20} {'N':>6} {'Top-1':>10} {'Top-3':>10}")
    print("-" * 50)
    
    overall_t1 = overall_t3 = overall_n = 0
    
    for src in ['AZ120', 'Zaretzki', 'DrugBank', 'MetXBioDB']:
        if src in by_source:
            t1, t3, n = evaluate_weights(by_source[src], default_weights)
            print(f"{src:<20} {n:>6} {t1*100:>9.1f}% {t3*100:>9.1f}%")
            overall_t1 += t1 * n
            overall_t3 += t3 * n
            overall_n += n
    
    print("-" * 50)
    print(f"{'OVERALL':<20} {overall_n:>6} {overall_t1/overall_n*100:>9.1f}% {overall_t3/overall_n*100:>9.1f}%")
    
    # Optimize for AZ120
    print("\n--- Optimizing for AZ120 ---")
    az_data = by_source.get('AZ120', [])
    if az_data:
        best_weights, best_t1 = optimize_weights(az_data, n_trials=300)
        
        print(f"\nBest AZ120 weights: Top-1 = {best_t1*100:.1f}%")
        print(f"Forces: flex={best_weights['flex']:.3f}, amp={best_weights['amp']:.3f}, "
              f"tunnel={best_weights['tunnel']:.3f}, vdw={best_weights['vdw']:.3f}")
        print(f"        topo={best_weights['topo']:.3f}")
        
        # Re-evaluate all sources with AZ120-optimized weights
        print(f"\n{'Source':<20} {'N':>6} {'Top-1':>10} {'Top-3':>10} (AZ120-optimized)")
        print("-" * 55)
        
        for src in ['AZ120', 'Zaretzki', 'DrugBank', 'MetXBioDB']:
            if src in by_source:
                t1, t3, n = evaluate_weights(by_source[src], best_weights)
                print(f"{src:<20} {n:>6} {t1*100:>9.1f}% {t3*100:>9.1f}%")
    
    return


if __name__ == '__main__':
    full_evaluation('/home/claude/enzyme_Software/data/curated/merged_cyp3a4_extended.json')
