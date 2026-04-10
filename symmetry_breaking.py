"""
SYMMETRY BREAKING MODEL - FAST VERSION

The core insight: aromatic carbons that appear equivalent in 2D
are NOT equivalent in 3D when an enzyme approaches.

The enzyme breaks symmetry through:
1. GEOMETRIC COUPLING - distance and angle to Fe=O
2. FIELD GRADIENT - electric field from enzyme active site
3. STERIC ACCESSIBILITY - which carbons can be reached
4. CONFORMATIONAL PREFERENCE - substrate docks in specific orientations

We don't know the exact enzyme position, but we CAN compute
which carbons are more LIKELY to be accessible across an 
ensemble of possible orientations.

Key insight: The "accessible solid angle" differs for different carbons!
- Carbons on the edge of the ring have more open solid angle
- Carbons with substituents nearby are shielded
- The 3D shape creates asymmetry even for equivalent 2D carbons
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
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    except:
        AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
    
    conf = mol.GetConformer()
    coords = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] 
                       for i in range(mol.GetNumAtoms())])
    return mol, coords


def compute_accessible_solid_angle(mol, coords, target_idx, probe_radius=3.0, n_samples=500):
    """
    Compute the solid angle accessible to an enzyme probe around target atom.
    
    This measures how "exposed" an atom is to attack from different directions.
    
    Returns a value from 0 (completely shielded) to 4π (completely exposed).
    """
    target_pos = coords[target_idx]
    n_atoms = len(coords)
    
    # Get atomic radii for steric clashes
    radii = {1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52, 9: 1.47, 16: 1.8, 17: 1.75}
    atom_radii = [radii.get(mol.GetAtomWithIdx(i).GetAtomicNum(), 1.7) 
                  for i in range(n_atoms)]
    
    # Sample directions uniformly on sphere
    accessible = 0
    
    for _ in range(n_samples):
        # Random direction
        phi = np.random.uniform(0, 2*np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        direction = np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ])
        
        # Probe position
        probe_pos = target_pos + probe_radius * direction
        
        # Check for steric clashes with other atoms
        clash = False
        for i in range(n_atoms):
            if i == target_idx:
                continue
            
            dist = np.linalg.norm(probe_pos - coords[i])
            if dist < atom_radii[i] + 1.5:  # 1.5Å probe radius
                clash = True
                break
        
        if not clash:
            accessible += 1
    
    # Convert to solid angle (fraction of sphere accessible)
    return accessible / n_samples * 4 * np.pi


def compute_field_gradient_exposure(mol, coords, target_idx):
    """
    Compute how exposed an atom is to external field gradients.
    
    Atoms on the "outside" of the molecule experience stronger gradients.
    This creates asymmetry for attack direction.
    """
    target_pos = coords[target_idx]
    center = coords.mean(axis=0)
    
    # Vector from center to target
    r = target_pos - center
    r_mag = np.linalg.norm(r)
    
    if r_mag < 0.1:
        return 0.5  # Center of molecule
    
    r_hat = r / r_mag
    
    # Compute local curvature (are neighbors also "outside"?)
    atom = mol.GetAtomWithIdx(target_idx)
    neighbor_outness = []
    
    for nbr in atom.GetNeighbors():
        j = nbr.GetIdx()
        r_nbr = coords[j] - center
        # Dot product: positive if neighbor also points outward
        outness = np.dot(r_nbr / (np.linalg.norm(r_nbr) + 0.1), r_hat)
        neighbor_outness.append(outness)
    
    if neighbor_outness:
        # If neighbors also point outward, this is a convex region
        convexity = np.mean(neighbor_outness)
    else:
        convexity = 0
    
    # Exposure: distance from center * convexity factor
    return r_mag * (1 + 0.3 * convexity)


def compute_approach_vector_distribution(mol, coords, target_idx, n_samples=200):
    """
    For an approaching enzyme, compute the distribution of possible
    approach angles and their associated coupling strengths.
    
    Returns: mean coupling, variance of coupling (higher variance = more directional)
    """
    target_pos = coords[target_idx]
    target_atom = mol.GetAtomWithIdx(target_idx)
    
    # Get normal direction for this atom (if aromatic, use ring normal)
    if target_atom.GetIsAromatic():
        # Find ring neighbors to define plane
        arom_nbrs = [nbr.GetIdx() for nbr in target_atom.GetNeighbors() 
                    if nbr.GetIsAromatic()]
        if len(arom_nbrs) >= 2:
            v1 = coords[arom_nbrs[0]] - target_pos
            v2 = coords[arom_nbrs[1]] - target_pos
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
        else:
            normal = np.array([0, 0, 1])
    else:
        # For sp3, use average of bond directions
        nbr_dirs = [coords[nbr.GetIdx()] - target_pos 
                   for nbr in target_atom.GetNeighbors()]
        if nbr_dirs:
            avg_dir = np.mean(nbr_dirs, axis=0)
            normal = -avg_dir / (np.linalg.norm(avg_dir) + 1e-10)
        else:
            normal = np.array([0, 0, 1])
    
    # Sample approach directions and compute coupling
    couplings = []
    
    for _ in range(n_samples):
        # Random approach direction
        phi = np.random.uniform(0, 2*np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        approach_dir = np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ])
        
        # Coupling depends on angle to C-H bond / ring normal
        cos_angle = abs(np.dot(approach_dir, normal))
        
        # Maximum coupling when aligned with C-H or perpendicular to ring
        coupling = cos_angle
        
        # Penalize if approach is blocked
        probe_pos = target_pos + 3.0 * approach_dir
        blocked = False
        for i in range(len(coords)):
            if i == target_idx:
                continue
            if np.linalg.norm(probe_pos - coords[i]) < 2.0:
                blocked = True
                break
        
        if not blocked:
            couplings.append(coupling)
    
    if not couplings:
        return 0, 0
    
    return np.mean(couplings), np.std(couplings)


def compute_local_electric_asymmetry(mol, coords, target_idx):
    """
    Compute asymmetry in local electric environment.
    
    Even for "equivalent" aromatic carbons, the 3D positions of
    substituents create electric field asymmetries.
    """
    target_pos = coords[target_idx]
    n = len(coords)
    
    # Electronegativity-based charges
    EN = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 16: 2.58}
    
    # Electric field at target
    E_field = np.zeros(3)
    
    for i in range(n):
        if i == target_idx:
            continue
        
        z = mol.GetAtomWithIdx(i).GetAtomicNum()
        q = (EN.get(z, 2.5) - 2.5) * 0.1  # Approximate partial charge
        
        r = coords[i] - target_pos
        r_mag = np.linalg.norm(r)
        
        if r_mag > 0.5:
            E_field += q * r / (r_mag**3)
    
    # Field magnitude and asymmetry
    E_mag = np.linalg.norm(E_field)
    
    # Asymmetry: how non-uniform is the field?
    # Compare with what a symmetric environment would give
    # For perfect symmetry, E_field would be zero
    return E_mag


def symmetry_breaking_score(smiles):
    """
    Compute SoM score with symmetry-breaking features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol_3d, coords_3d = get_3d_coords(mol)
    mol_noH = Chem.RemoveAllHs(mol)
    
    n = mol_noH.GetNumAtoms()
    if n < 3:
        return None
    
    # Map to heavy-atom coords
    heavy_idx = [i for i in range(mol_3d.GetNumAtoms()) 
                if mol_3d.GetAtomWithIdx(i).GetAtomicNum() > 1]
    coords = coords_3d[heavy_idx]
    
    # === SYMMETRY-BREAKING FEATURES ===
    
    # 1. Accessible solid angle
    solid_angle = np.array([compute_accessible_solid_angle(mol_noH, coords, i) 
                           for i in range(n)])
    
    # 2. Field gradient exposure
    field_exposure = np.array([compute_field_gradient_exposure(mol_noH, coords, i) 
                               for i in range(n)])
    
    # 3. Approach distribution
    approach_coupling = np.zeros(n)
    approach_variance = np.zeros(n)
    for i in range(n):
        if mol_noH.GetAtomWithIdx(i).GetAtomicNum() == 6:
            mean_c, std_c = compute_approach_vector_distribution(mol_noH, coords, i)
            approach_coupling[i] = mean_c
            approach_variance[i] = std_c
    
    # 4. Electric asymmetry
    electric_asym = np.array([compute_local_electric_asymmetry(mol_noH, coords, i) 
                              for i in range(n)])
    
    # === GRAPH FEATURES (baseline) ===
    A = np.zeros((n, n))
    for bond in mol_noH.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = bond.GetBondTypeAsDouble()
    
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
    
    # Normalize
    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-10 else np.ones_like(x) * 0.5
    
    solid_n = norm(solid_angle)
    field_n = norm(field_exposure)
    approach_n = norm(approach_coupling)
    electric_n = norm(electric_asym)
    flex_n = norm(flex)
    topo_n = norm(topo)
    
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
        
        # For aromatic: symmetry-breaking features are key
        if is_arom:
            base = (0.30 * solid_n[i] +       # Solid angle accessibility
                    0.25 * approach_n[i] +     # Approach coupling
                    0.20 * field_n[i] +        # Field exposure
                    0.15 * electric_n[i] +     # Electric asymmetry
                    0.10 * topo_n[i])          # Topological
        else:
            # For aliphatic: mix of 3D and graph features
            base = (0.25 * solid_n[i] +
                    0.25 * flex_n[i] +
                    0.20 * approach_n[i] +
                    0.15 * field_n[i] +
                    0.15 * topo_n[i])
        
        # Chemical multipliers
        alpha_mult = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7: alpha_mult = max(alpha_mult, 1.8)
            elif z == 8: alpha_mult = max(alpha_mult, 1.75)
            elif z == 16: alpha_mult = max(alpha_mult, 1.6)
        
        benz_mult = 1.0
        if not is_arom and n_H > 0:
            if any(nbr.GetIsAromatic() for nbr in atom.GetNeighbors()):
                benz_mult = 1.6
        
        h_factor = (1 + 0.15 * n_H) if n_H > 0 else 0.5
        
        scores[i] = base * alpha_mult * benz_mult * h_factor
    
    return scores


def evaluate(data_path, limit=None, sources=None):
    """Evaluate the symmetry-breaking model."""
    with open(data_path) as f:
        data = json.load(f)['drugs']
    
    if sources:
        data = [d for d in data if d.get('source') in sources]
    if limit:
        data = data[:limit]
    
    by_source = defaultdict(lambda: {'t1': 0, 't3': 0, 'n': 0})
    arom_t1 = arom_n = 0
    
    for d in data:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        src = d.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        scores = symmetry_breaking_score(smiles)
        if scores is None:
            continue
        
        valid = [i for i in range(len(scores)) if scores[i] > -np.inf]
        if not valid:
            continue
        
        ranked = sorted(valid, key=lambda x: -scores[x])[:3]
        
        by_source[src]['n'] += 1
        hit1 = ranked[0] in sites
        if hit1:
            by_source[src]['t1'] += 1
        if any(r in sites for r in ranked):
            by_source[src]['t3'] += 1
        
        # Track aromatic specifically
        site = sites[0]
        if site < mol.GetNumAtoms():
            if mol.GetAtomWithIdx(site).GetIsAromatic():
                arom_n += 1
                if hit1:
                    arom_t1 += 1
    
    print("\n" + "="*70)
    print("SYMMETRY-BREAKING MODEL - 3D GEOMETRIC FEATURES")
    print("="*70)
    print("\nFeatures: Solid angle, approach coupling, field exposure, electric asymmetry")
    
    print(f"\n{'Source':<15} {'N':>6} {'Top-1':>10} {'Top-3':>10}")
    print("-" * 45)
    
    total_t1 = total_t3 = total_n = 0
    for src in ['AZ120', 'Zaretzki', 'DrugBank', 'MetXBioDB']:
        s = by_source.get(src, {'t1': 0, 't3': 0, 'n': 0})
        if s['n'] > 0:
            print(f"{src:<15} {s['n']:>6} {s['t1']/s['n']*100:>9.1f}% {s['t3']/s['n']*100:>9.1f}%")
            total_t1 += s['t1']
            total_t3 += s['t3']
            total_n += s['n']
    
    if arom_n > 0:
        print(f"\nAromatic sites: {arom_t1}/{arom_n} = {arom_t1/arom_n*100:.1f}% Top-1")
    
    return by_source


if __name__ == '__main__':
    print("Testing Symmetry-Breaking Model...")
    evaluate('/home/claude/enzyme_Software/data/curated/merged_cyp3a4_extended.json', limit=100)
