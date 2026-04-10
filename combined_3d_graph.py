"""
ULTIMATE COMBINED MODEL

Combines:
1. 3D symmetry-breaking features (breaks aromatic equivalence)
2. Graph-based quantum forces (proven baseline)
3. Optimized weights from search

Key insight: The 3D features help WITHIN a class of equivalent atoms,
but the graph features determine which CLASS is most reactive.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import json
from collections import defaultdict


def get_3d_coords(mol):
    """Generate 3D structure."""
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    
    conf = mol.GetConformer()
    return mol, np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] 
                          for i in range(mol.GetNumAtoms())])


def compute_3d_accessibility(mol, coords, target_idx, n_samples=200):
    """
    Combined 3D accessibility metric:
    - Solid angle (how much is exposed)
    - Surface exposure (distance from center)
    - Directional coupling (approach alignment)
    """
    target_pos = coords[target_idx]
    center = coords.mean(axis=0)
    n = len(coords)
    
    # 1. Surface exposure
    surface = np.linalg.norm(target_pos - center)
    
    # 2. Solid angle sampling
    radii = {1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52, 16: 1.8}
    atom_radii = [radii.get(mol.GetAtomWithIdx(i).GetAtomicNum(), 1.7) for i in range(n)]
    
    accessible = 0
    aligned = 0
    
    for _ in range(n_samples):
        phi = np.random.uniform(0, 2*np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        direction = np.array([sin_theta * np.cos(phi),
                              sin_theta * np.sin(phi),
                              cos_theta])
        
        probe_pos = target_pos + 3.5 * direction
        
        clash = False
        for i in range(n):
            if i == target_idx:
                continue
            if np.linalg.norm(probe_pos - coords[i]) < atom_radii[i] + 1.5:
                clash = True
                break
        
        if not clash:
            accessible += 1
            # Check alignment with "approach from outside"
            inward = (center - target_pos)
            inward = inward / (np.linalg.norm(inward) + 1e-10)
            if np.dot(direction, inward) < 0:  # Approaching from outside
                aligned += 1
    
    return surface, accessible / n_samples, aligned / (accessible + 1)


def compute_electric_environment(mol, coords, target_idx):
    """Asymmetry in local electric field."""
    EN = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 16: 2.58}
    
    target_pos = coords[target_idx]
    E = np.zeros(3)
    
    for i in range(len(coords)):
        if i == target_idx:
            continue
        z = mol.GetAtomWithIdx(i).GetAtomicNum()
        q = (EN.get(z, 2.5) - 2.5) * 0.1
        r = coords[i] - target_pos
        r_mag = np.linalg.norm(r)
        if r_mag > 0.5:
            E += q * r / (r_mag**3)
    
    return np.linalg.norm(E)


def combined_som_score(smiles, weights=None):
    """
    Combined score using 3D + graph features.
    """
    if weights is None:
        weights = {
            # 3D features
            '3d_solid': 0.15,
            '3d_align': 0.10,
            '3d_surface': 0.10,
            '3d_electric': 0.05,
            # Graph features  
            'flex': 0.25,
            'topo': 0.15,
            'tunnel': 0.20,
            # Chemical
            'aN': 1.72, 'aO': 1.87, 'aS': 1.77,
            'benz': 1.57, 'tert': 0.10, 'hf': 0.10
        }
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get 3D
    mol_3d, coords_3d = get_3d_coords(mol)
    mol_noH = Chem.RemoveAllHs(mol)
    n = mol_noH.GetNumAtoms()
    
    if n < 3:
        return None
    
    # Map to heavy atoms
    heavy_idx = [i for i in range(mol_3d.GetNumAtoms()) 
                if mol_3d.GetAtomWithIdx(i).GetAtomicNum() > 1]
    coords = coords_3d[heavy_idx]
    
    # === 3D FEATURES ===
    surface = np.zeros(n)
    solid = np.zeros(n)
    align = np.zeros(n)
    electric = np.zeros(n)
    
    for i in range(n):
        surface[i], solid[i], align[i] = compute_3d_accessibility(mol_noH, coords, i)
        electric[i] = compute_electric_environment(mol_noH, coords, i)
    
    # === GRAPH FEATURES ===
    A = np.zeros((n, n))
    for bond in mol_noH.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic(): w = 1.5
        A[i, j] = A[j, i] = w
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    flex = np.array([1.0 / (sum(eigenvectors[i, k]**2 
                    for k in range(max(1, n-3), n)) + 0.1) for i in range(n)])
    
    topo = np.zeros(n)
    for i in range(n):
        phase = sum(abs(eigenvectors[i, k] - eigenvectors[j, k]) 
                   for nbr in mol_noH.GetAtomWithIdx(i).GetNeighbors() 
                   for j in [nbr.GetIdx()] for k in range(1, min(5, n)))
        topo[i] = 1.0 / (phase + 0.1)
    
    # Tunneling
    tunnel = np.zeros(n)
    for i in range(n):
        atom = mol_noH.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            continue
        n_H = atom.GetTotalNumHs()
        base = flex[i] * (1 + 0.3 * n_H)
        mod = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7: mod *= 1.5
            elif z == 8: mod *= 1.4
            elif z == 16: mod *= 1.3
        tunnel[i] = base * mod
    
    # Normalize
    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-10 else np.ones_like(x) * 0.5
    
    surface_n = norm(surface)
    solid_n = norm(solid)
    align_n = norm(align)
    electric_n = norm(electric)
    flex_n = norm(flex)
    topo_n = norm(topo)
    tunnel_n = norm(tunnel)
    
    # === FINAL SCORE ===
    scores = np.full(n, -np.inf)
    
    for i in range(n):
        atom = mol_noH.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            continue
        
        n_H = atom.GetTotalNumHs()
        is_arom = atom.GetIsAromatic()
        
        if n_H == 0 and not is_arom:
            continue
        
        # Combined score
        base = (
            weights['3d_solid'] * solid_n[i] +
            weights['3d_align'] * align_n[i] +
            weights['3d_surface'] * surface_n[i] +
            weights['3d_electric'] * electric_n[i] +
            weights['flex'] * flex_n[i] +
            weights['topo'] * topo_n[i] +
            weights['tunnel'] * tunnel_n[i]
        )
        
        # Chemical multipliers
        alpha_mult = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7: alpha_mult = max(alpha_mult, weights['aN'])
            elif z == 8: alpha_mult = max(alpha_mult, weights['aO'])
            elif z == 16: alpha_mult = max(alpha_mult, weights['aS'])
        
        benz_mult = 1.0
        if not is_arom and n_H > 0:
            if any(nbr.GetIsAromatic() for nbr in atom.GetNeighbors()):
                benz_mult = weights['benz']
        
        n_C = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6)
        tert_mult = 1.0 + weights['tert'] * (n_C - 1) if n_C > 1 else 1.0
        
        h_factor = (1 + weights['hf'] * n_H) if n_H > 0 else 0.5
        
        scores[i] = base * alpha_mult * benz_mult * tert_mult * h_factor
    
    return scores


def optimize_weights(data, n_trials=300):
    """Optimize weights on data."""
    best_t1 = 0
    best_w = None
    
    for trial in range(n_trials):
        w = {
            '3d_solid': np.random.uniform(0.05, 0.25),
            '3d_align': np.random.uniform(0.05, 0.20),
            '3d_surface': np.random.uniform(0.05, 0.20),
            '3d_electric': np.random.uniform(0.0, 0.15),
            'flex': np.random.uniform(0.15, 0.35),
            'topo': np.random.uniform(0.05, 0.25),
            'tunnel': np.random.uniform(0.10, 0.30),
            'aN': np.random.uniform(1.5, 2.0),
            'aO': np.random.uniform(1.5, 2.0),
            'aS': np.random.uniform(1.3, 1.9),
            'benz': np.random.uniform(1.3, 1.8),
            'tert': np.random.uniform(0.05, 0.20),
            'hf': np.random.uniform(0.05, 0.20),
        }
        
        # Normalize feature weights
        fkeys = ['3d_solid', '3d_align', '3d_surface', '3d_electric', 'flex', 'topo', 'tunnel']
        total = sum(w[k] for k in fkeys)
        for k in fkeys:
            w[k] /= total
        
        t1, t3, n = evaluate_weights(data, w)
        
        if t1 > best_t1:
            best_t1 = t1
            best_w = w.copy()
            print(f"  Trial {trial}: Top-1={t1*100:.1f}%, Top-3={t3*100:.1f}%")
    
    return best_w, best_t1


def evaluate_weights(data, w):
    """Evaluate with given weights."""
    t1 = t3 = n = 0
    
    for d in data:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        
        if not smiles or not sites:
            continue
        
        scores = combined_som_score(smiles, w)
        if scores is None:
            continue
        
        valid = [i for i in range(len(scores)) if scores[i] > -np.inf]
        if not valid:
            continue
        
        ranked = sorted(valid, key=lambda x: -scores[x])[:3]
        
        if ranked[0] in sites: t1 += 1
        if any(r in sites for r in ranked): t3 += 1
        n += 1
    
    return t1/n if n > 0 else 0, t3/n if n > 0 else 0, n


def evaluate(data_path, sources=None, limit=None):
    """Full evaluation."""
    with open(data_path) as f:
        data = json.load(f)['drugs']
    
    if sources:
        data = [d for d in data if d.get('source') in sources]
    if limit:
        data = data[:limit]
    
    by_source = defaultdict(lambda: {'t1': 0, 't3': 0, 'n': 0})
    
    for d in data:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        src = d.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        scores = combined_som_score(smiles)
        if scores is None:
            continue
        
        valid = [i for i in range(len(scores)) if scores[i] > -np.inf]
        if not valid:
            continue
        
        ranked = sorted(valid, key=lambda x: -scores[x])[:3]
        
        by_source[src]['n'] += 1
        if ranked[0] in sites:
            by_source[src]['t1'] += 1
        if any(r in sites for r in ranked):
            by_source[src]['t3'] += 1
    
    print("\n" + "="*70)
    print("COMBINED 3D + GRAPH MODEL")
    print("="*70)
    
    print(f"\n{'Source':<15} {'N':>6} {'Top-1':>10} {'Top-3':>10}")
    print("-" * 45)
    
    for src in ['AZ120', 'Zaretzki', 'DrugBank', 'MetXBioDB']:
        s = by_source.get(src, {'t1': 0, 't3': 0, 'n': 0})
        if s['n'] > 0:
            print(f"{src:<15} {s['n']:>6} {s['t1']/s['n']*100:>9.1f}% {s['t3']/s['n']*100:>9.1f}%")
    
    return by_source


if __name__ == '__main__':
    print("Testing Combined Model...")
    evaluate('/home/claude/enzyme_Software/data/curated/merged_cyp3a4_extended.json', 
             sources=['AZ120'])
