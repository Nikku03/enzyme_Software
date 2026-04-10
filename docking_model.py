"""
MOLECULAR DOCKING FOR 90% SOM PREDICTION

The Problem:
- Many molecules have chemically equivalent atoms (e.g., ortho positions in benzene)
- Without enzyme structure, theoretical max on Zaretzki is only 58.7%
- We MUST use enzyme geometry to break this symmetry

The Solution:
1. Get CYP3A4 crystal structure
2. Dock substrate into active site
3. Identify atoms within reaction distance of Fe=O
4. Score by accessibility + reactivity

CYP3A4 Active Site:
- Heme Fe at center
- Fe-O bond ~1.65Å (Compound I)
- Reactive distance: 3.5-5.0Å from Fe
- Key residues: Phe304, Ile369, Ala370, Thr309
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import docking tools
try:
    import prody
    PRODY_AVAILABLE = True
except:
    PRODY_AVAILABLE = False

try:
    from openbabel import openbabel
    OPENBABEL_AVAILABLE = True
except:
    OPENBABEL_AVAILABLE = False


# CYP3A4 active site geometry (from crystal structures)
# These are approximate coordinates based on 1TQN
CYP3A4_ACTIVE_SITE = {
    'fe_position': np.array([0.0, 0.0, 0.0]),  # Reference frame centered on Fe
    'reactive_radius_min': 3.5,  # Å
    'reactive_radius_max': 5.0,  # Å
    'cavity_volume': 1400,  # Å³
    # Key residue positions relative to Fe (approximate)
    'phe304': np.array([4.5, 2.0, 3.0]),
    'ile369': np.array([-3.5, 3.0, 2.5]),
    'ala370': np.array([-2.0, 4.0, 1.5]),
    'thr309': np.array([3.0, -2.0, 4.0]),
}


def simulate_docking(mol, n_poses=10):
    """
    Simulate molecular docking by generating multiple conformers
    and scoring their accessibility to Fe.
    
    In production, this would use AutoDock Vina or Glide.
    Here we use conformer generation + geometric scoring.
    """
    mol = Chem.AddHs(mol)
    
    # Generate conformers
    try:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_poses, 
                                           randomSeed=42,
                                           pruneRmsThresh=0.5)
        for cid in cids:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=200)
    except:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        cids = [0]
    
    return mol, list(cids)


def score_atom_accessibility(mol, atom_idx, conf_id, fe_position):
    """
    Score how accessible an atom is to the Fe=O center.
    
    Factors:
    1. Distance to Fe (optimal: 3.5-5.0 Å)
    2. Steric accessibility (not blocked by other atoms)
    3. H-atom orientation (if applicable)
    """
    conf = mol.GetConformer(conf_id)
    
    # Get atom position
    pos = conf.GetAtomPosition(atom_idx)
    atom_pos = np.array([pos.x, pos.y, pos.z])
    
    # Distance to Fe
    dist = np.linalg.norm(atom_pos - fe_position)
    
    # Distance score (optimal at 4.0 Å)
    dist_optimal = 4.0
    dist_score = np.exp(-((dist - dist_optimal) / 1.5)**2)
    
    # Steric accessibility - count atoms blocking the path to Fe
    n_blocking = 0
    for i in range(mol.GetNumAtoms()):
        if i == atom_idx:
            continue
        other_pos = conf.GetAtomPosition(i)
        other = np.array([other_pos.x, other_pos.y, other_pos.z])
        
        # Check if this atom is between target and Fe
        to_fe = fe_position - atom_pos
        to_other = other - atom_pos
        
        proj = np.dot(to_other, to_fe) / (np.linalg.norm(to_fe)**2 + 1e-10)
        
        if 0 < proj < 1:
            # This atom is in the path
            closest = atom_pos + proj * to_fe
            dist_to_path = np.linalg.norm(other - closest)
            if dist_to_path < 2.0:  # Within VdW radius
                n_blocking += 1
    
    steric_score = 1.0 / (1.0 + 0.3 * n_blocking)
    
    # H-atom factor
    atom = mol.GetAtomWithIdx(atom_idx)
    n_H = atom.GetTotalNumHs()
    h_factor = 1.0 + 0.2 * n_H
    
    return dist_score * steric_score * h_factor


def docking_som_score(smiles, n_poses=10):
    """
    Compute SoM scores using simulated docking.
    
    For each atom, average accessibility across all poses.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol_noH = mol
    n = mol_noH.GetNumAtoms()
    
    if n < 3:
        return None
    
    # Generate docked poses
    mol_3d, conf_ids = simulate_docking(mol_noH, n_poses)
    
    if not conf_ids:
        return None
    
    # Map from 3D mol (with H) to original (without H)
    heavy_map = []
    for i in range(mol_3d.GetNumAtoms()):
        if mol_3d.GetAtomWithIdx(i).GetAtomicNum() > 1:
            heavy_map.append(i)
    
    # Score each atom across poses
    scores = np.zeros(n)
    
    for pose_idx, conf_id in enumerate(conf_ids):
        # Simulate different binding orientations
        # by rotating the Fe position
        theta = pose_idx * 2 * np.pi / len(conf_ids)
        fe_pos = np.array([4.0 * np.cos(theta), 4.0 * np.sin(theta), 0.0])
        
        for i in range(n):
            if i >= len(heavy_map):
                continue
            
            atom = mol_noH.GetAtomWithIdx(i)
            if atom.GetAtomicNum() != 6:
                continue
            
            n_H = atom.GetTotalNumHs()
            if n_H == 0 and not atom.GetIsAromatic():
                continue
            
            heavy_idx = heavy_map[i]
            acc_score = score_atom_accessibility(mol_3d, heavy_idx, conf_id, fe_pos)
            scores[i] += acc_score
    
    # Average across poses
    scores /= len(conf_ids)
    
    # Apply chemical multipliers
    final_scores = np.full(n, -np.inf)
    
    for i in range(n):
        atom = mol_noH.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            continue
        
        n_H = atom.GetTotalNumHs()
        if n_H == 0 and not atom.GetIsAromatic():
            continue
        
        base = scores[i]
        
        # Alpha-heteroatom
        alpha_mult = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7: alpha_mult = max(alpha_mult, 1.8)
            elif z == 8: alpha_mult = max(alpha_mult, 1.75)
            elif z == 16: alpha_mult = max(alpha_mult, 1.65)
        
        # Benzylic
        benz_mult = 1.0
        if not atom.GetIsAromatic() and n_H > 0:
            if any(nbr.GetIsAromatic() for nbr in atom.GetNeighbors()):
                benz_mult = 1.6
        
        # Tertiary
        n_C = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6)
        tert_mult = 1.0 + 0.12 * (n_C - 1) if n_C > 1 else 1.0
        
        # H-count
        h_factor = (1 + 0.15 * n_H) if n_H > 0 else 0.3
        
        final_scores[i] = base * alpha_mult * benz_mult * tert_mult * h_factor
    
    return final_scores


def evaluate(data_path, sources=None, limit=None, n_poses=10):
    """Evaluate docking model."""
    with open(data_path) as f:
        data = json.load(f)['drugs']
    
    if sources:
        data = [d for d in data if d.get('source') in sources]
    if limit:
        data = data[:limit]
    
    by_source = defaultdict(lambda: {'t1': 0, 't3': 0, 'n': 0})
    
    for idx, d in enumerate(data):
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        src = d.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        scores = docking_som_score(smiles, n_poses=n_poses)
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
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(data)}...")
    
    print("\n" + "="*70)
    print("DOCKING MODEL (Simulated)")
    print("="*70)
    
    print(f"\n{'Source':<12} {'N':>5} {'Top-1':>8} {'Top-3':>8}")
    print("-" * 35)
    
    for src in ['AZ120', 'Zaretzki', 'DrugBank', 'MetXBioDB']:
        s = by_source.get(src, {'t1': 0, 't3': 0, 'n': 0})
        if s['n'] > 0:
            t1 = s['t1']/s['n']*100
            t3 = s['t3']/s['n']*100
            print(f"{src:<12} {s['n']:>5} {t1:>7.1f}% {t3:>7.1f}%")
    
    return by_source


if __name__ == '__main__':
    print("Testing Docking Model...")
    print(f"ProDy available: {PRODY_AVAILABLE}")
    print(f"OpenBabel available: {OPENBABEL_AVAILABLE}")
    
    evaluate('data/curated/merged_cyp3a4_extended.json', 
             sources=['AZ120'], n_poses=10)
