"""
DEEP SYMMETRY BREAKING ANALYSIS

The 55-45 split is REAL. Let's find WHY.

=== DYNAMIC FLUCTUATIONS ===

At any instant, the molecule is NOT symmetric:
- Thermal vibrations distort bond lengths by ±0.1Å
- Ring puckering breaks planarity  
- C-H stretches are uncorrelated
- Vibrational phase determines which C-H is "ready" to react

Key insight: The REACTION happens at a specific instant.
The enzyme doesn't see the time-averaged structure.
It sees ONE snapshot of a fluctuating molecule.

=== SUBSTITUENT EFFECTS (INDUCTIVE + RESONANCE) ===

Even for "equivalent" carbons, substituents create gradients:

Consider toluene: the methyl at position 1 creates:
- Inductive effect: +I (electron donating) through sigma bonds
- Hyperconjugation: C-H → π* donation
- These effects decay with distance but create ASYMMETRY

Ortho carbons (2,6) feel it most.
Meta carbons (3,5) feel it less.
Para carbon (4) feels intermediate.

For ANY substituted benzene, there are NO truly equivalent carbons!

=== JAHN-TELLER EFFECT ===

For transition states with orbital degeneracy:
- The symmetric configuration is UNSTABLE
- System spontaneously distorts to lower symmetry
- One carbon becomes more reactive than others

In CYP oxidation:
- The [Fe=O...C-H] complex can have degenerate orbitals
- Jahn-Teller distortion favors one pathway
- This is a QUANTUM MECHANICAL symmetry breaking

=== PSEUDO-JAHN-TELLER (SECOND ORDER) ===

Even without strict degeneracy:
- Near-degenerate states can mix
- Vibronic coupling creates asymmetric potentials
- Different carbons sit in different parts of the potential

Let's compute all of these.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import json
from collections import defaultdict


# === SUBSTITUENT EFFECT CALCULATION ===

# Hammett sigma constants (electronic effect of substituents)
SIGMA_PARA = {
    'CH3': -0.17,   # Methyl - electron donating
    'OCH3': -0.27,  # Methoxy - strong donor
    'OH': -0.37,    # Hydroxyl
    'NH2': -0.66,   # Amino - very strong donor
    'F': 0.06,      # Fluoro - weak withdrawing
    'Cl': 0.23,     # Chloro
    'Br': 0.23,     # Bromo
    'NO2': 0.78,    # Nitro - strong withdrawing
    'CF3': 0.54,    # Trifluoromethyl
    'CN': 0.66,     # Cyano
    'COOH': 0.45,   # Carboxylic acid
    'CHO': 0.42,    # Aldehyde
}

SIGMA_META = {
    'CH3': -0.07,
    'OCH3': 0.12,
    'OH': 0.12,
    'NH2': -0.16,
    'F': 0.34,
    'Cl': 0.37,
    'Br': 0.39,
    'NO2': 0.71,
    'CF3': 0.43,
    'CN': 0.56,
    'COOH': 0.37,
    'CHO': 0.35,
}


def compute_substituent_asymmetry(mol, aromatic_carbons):
    """
    Compute how substituents break symmetry of aromatic carbons.
    
    Even "equivalent" positions feel different electronic effects
    from substituents at different distances/angles.
    """
    n = mol.GetNumAtoms()
    asymmetry = np.zeros(n)
    
    # For each aromatic carbon, compute cumulative substituent effect
    for i in aromatic_carbons:
        atom = mol.GetAtomWithIdx(i)
        
        # Find all non-H neighbors that are NOT aromatic (substituents)
        for nbr in atom.GetNeighbors():
            if not nbr.GetIsAromatic():
                # This carbon is directly substituted
                # Effect decays to ortho > meta > para positions
                z = nbr.GetAtomicNum()
                
                # Approximate substituent effect
                if z == 6:  # Alkyl
                    sigma = -0.15  # Electron donating
                elif z == 7:  # Amino-like
                    sigma = -0.50
                elif z == 8:  # Oxy
                    sigma = -0.30
                elif z in [9, 17, 35]:  # Halogen
                    sigma = 0.25
                else:
                    sigma = 0.0
                
                # This carbon feels full effect
                asymmetry[i] += sigma
                
                # Neighboring aromatic carbons feel ortho effect
                for ring_nbr in atom.GetNeighbors():
                    if ring_nbr.GetIsAromatic():
                        j = ring_nbr.GetIdx()
                        asymmetry[j] += 0.7 * sigma  # Ortho
                        
                        # Their neighbors feel meta effect
                        for meta_nbr in ring_nbr.GetNeighbors():
                            if meta_nbr.GetIsAromatic() and meta_nbr.GetIdx() != i:
                                k = meta_nbr.GetIdx()
                                asymmetry[k] += 0.4 * sigma  # Meta
    
    return asymmetry


def compute_inductive_gradient(mol, aromatic_carbons):
    """
    Compute inductive effect gradient across aromatic ring.
    
    Inductive effects propagate through sigma bonds and decay
    with distance. This creates ASYMMETRY even in "equivalent" positions.
    """
    n = mol.GetNumAtoms()
    
    # Build distance matrix (bond distances)
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist[i, j] = dist[j, i] = 1
    
    # Floyd-Warshall for shortest paths
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
    
    # Electronegativity difference creates inductive source
    EN = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 16: 2.58, 17: 3.16}
    
    inductive = np.zeros(n)
    
    for i in aromatic_carbons:
        # Sum inductive contributions from all heteroatoms
        for j in range(n):
            z_j = mol.GetAtomWithIdx(j).GetAtomicNum()
            if z_j in [7, 8, 9, 16, 17]:  # Heteroatoms
                en_diff = EN.get(z_j, 2.5) - 2.55  # Relative to carbon
                d = dist[i, j]
                if d < np.inf:
                    # Inductive effect decays as 1/d²
                    inductive[i] += en_diff / (d**2 + 0.1)
    
    return inductive


def compute_hyperconjugation(mol, aromatic_carbons):
    """
    Compute hyperconjugation effects from adjacent C-H bonds.
    
    Alkyl groups can donate electron density through C-H → π* overlap.
    This stabilizes carbocations at ortho/para positions.
    """
    n = mol.GetNumAtoms()
    hyperconj = np.zeros(n)
    
    for i in aromatic_carbons:
        atom = mol.GetAtomWithIdx(i)
        
        for nbr in atom.GetNeighbors():
            if not nbr.GetIsAromatic() and nbr.GetAtomicNum() == 6:
                # Adjacent alkyl carbon
                n_H = nbr.GetTotalNumHs()
                # More C-H bonds = more hyperconjugation
                hyperconj[i] += 0.1 * n_H
                
                # Also affects ortho positions (through π system)
                for ring_nbr in atom.GetNeighbors():
                    if ring_nbr.GetIsAromatic():
                        hyperconj[ring_nbr.GetIdx()] += 0.05 * n_H
    
    return hyperconj


def compute_vibrational_asymmetry(mol, coords, aromatic_carbons):
    """
    Compute how thermal vibrations break instantaneous symmetry.
    
    At any given instant:
    - Bond lengths fluctuate by ~0.1Å (thermal motion)
    - Ring puckers out of plane
    - C-H stretches are uncorrelated
    
    The VARIANCE of these fluctuations differs by position!
    """
    if coords is None:
        return np.zeros(mol.GetNumAtoms())
    
    n = mol.GetNumAtoms()
    
    # Compute graph Laplacian for vibrational modes
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Vibrational amplitude at each atom
    # Higher frequency modes = smaller amplitude
    # Positions with more low-freq participation = more mobile
    vib_amplitude = np.zeros(n)
    
    for i in range(n):
        for k in range(1, n):  # Skip zero mode
            omega = np.sqrt(max(0, eigenvalues[k]))
            if omega > 0.01:
                # Amplitude ∝ 1/ω × participation
                vib_amplitude[i] += eigenvectors[i, k]**2 / omega
    
    return vib_amplitude


def compute_jahn_teller_susceptibility(mol, aromatic_carbons):
    """
    Compute susceptibility to Jahn-Teller distortion.
    
    Positions that would have degenerate orbitals in the transition
    state are susceptible to symmetry-breaking distortion.
    
    In practice, this correlates with:
    - Positions with high orbital density
    - Positions where small distortion greatly changes energy
    """
    n = mol.GetNumAtoms()
    
    # Graph Laplacian eigenstructure
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = 1.5 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble()
        A[i, j] = A[j, i] = w
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Find near-degenerate eigenvalues (|λ_i - λ_j| < threshold)
    threshold = 0.3
    jt_suscept = np.zeros(n)
    
    for k in range(1, n-1):
        for l in range(k+1, n):
            gap = abs(eigenvalues[k] - eigenvalues[l])
            if gap < threshold:
                # Near-degenerate pair
                # Atoms with participation in BOTH modes are JT-active
                for i in aromatic_carbons:
                    mixed_participation = eigenvectors[i, k]**2 * eigenvectors[i, l]**2
                    # Susceptibility ∝ participation / gap
                    jt_suscept[i] += mixed_participation / (gap + 0.01)
    
    return jt_suscept


def compute_pseudo_jahn_teller(mol, aromatic_carbons):
    """
    Pseudo-Jahn-Teller effect (second-order vibronic coupling).
    
    Even without strict degeneracy, states can mix through
    vibrational modes, creating asymmetric potentials.
    
    PJT is strongest when:
    - Energy gap is small
    - Vibronic coupling matrix element is large
    - The distortion mode has the right symmetry
    """
    n = mol.GetNumAtoms()
    
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = 1.5 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble()
        A[i, j] = A[j, i] = w
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    pjt = np.zeros(n)
    
    # Consider mixing between HOMO-like and LUMO-like orbitals
    # In our graph model, these are eigenvalues around the middle
    n_occ = n // 2  # Approximate "HOMO" index
    
    for i in aromatic_carbons:
        # Vibronic coupling ∝ ⟨HOMO|∂H/∂Q|LUMO⟩ × ⟨i|HOMO⟩⟨i|LUMO⟩
        if n_occ > 0 and n_occ < n - 1:
            homo_amp = eigenvectors[i, n_occ]
            lumo_amp = eigenvectors[i, n_occ + 1]
            gap = eigenvalues[n_occ + 1] - eigenvalues[n_occ]
            
            # PJT contribution
            pjt[i] = (homo_amp * lumo_amp)**2 / (gap + 0.1)
    
    return pjt


def deep_symmetry_score(smiles):
    """
    Compute SoM score with deep symmetry-breaking effects.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    n = mol.GetNumAtoms()
    if n < 3:
        return None
    
    # Get 3D coords if possible
    try:
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
        conf = mol_3d.GetConformer()
        heavy_idx = [i for i in range(mol_3d.GetNumAtoms()) 
                    if mol_3d.GetAtomWithIdx(i).GetAtomicNum() > 1]
        coords = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z] 
                          for i in heavy_idx])
    except:
        coords = None
    
    # Find aromatic carbons
    aromatic_carbons = [i for i in range(n) 
                       if mol.GetAtomWithIdx(i).GetIsAromatic() and
                       mol.GetAtomWithIdx(i).GetAtomicNum() == 6]
    
    # === COMPUTE SYMMETRY-BREAKING EFFECTS ===
    
    # 1. Substituent electronic effects
    subst_asym = compute_substituent_asymmetry(mol, aromatic_carbons)
    
    # 2. Inductive gradient
    inductive = compute_inductive_gradient(mol, aromatic_carbons)
    
    # 3. Hyperconjugation
    hyperconj = compute_hyperconjugation(mol, aromatic_carbons)
    
    # 4. Vibrational asymmetry
    vib_asym = compute_vibrational_asymmetry(mol, coords, aromatic_carbons)
    
    # 5. Jahn-Teller susceptibility
    jt_suscept = compute_jahn_teller_susceptibility(mol, aromatic_carbons)
    
    # 6. Pseudo-Jahn-Teller
    pjt = compute_pseudo_jahn_teller(mol, aromatic_carbons)
    
    # === BASELINE GRAPH FEATURES ===
    
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = 1.5 if bond.GetIsAromatic() else bond.GetBondTypeAsDouble()
        A[i, j] = A[j, i] = w
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    flex = np.array([1.0 / (sum(eigenvectors[i, k]**2 
                    for k in range(max(1, n-3), n)) + 0.1) for i in range(n)])
    
    topo = np.zeros(n)
    for i in range(n):
        phase = sum(abs(eigenvectors[i, k] - eigenvectors[j, k]) 
                   for nbr in mol.GetAtomWithIdx(i).GetNeighbors() 
                   for j in [nbr.GetIdx()] for k in range(1, min(5, n)))
        topo[i] = 1.0 / (phase + 0.1)
    
    tunnel = np.zeros(n)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
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
    
    # Normalize all features
    def norm(x):
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-10 else np.ones_like(x) * 0.5
    
    subst_n = norm(np.abs(subst_asym))  # Magnitude of substituent effect
    induct_n = norm(np.abs(inductive))
    hyper_n = norm(hyperconj)
    vib_n = norm(vib_asym)
    jt_n = norm(jt_suscept)
    pjt_n = norm(pjt)
    flex_n = norm(flex)
    topo_n = norm(topo)
    tunnel_n = norm(tunnel)
    
    # === FINAL SCORE ===
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
            # For aromatic: symmetry-breaking effects dominate
            base = (0.15 * subst_n[i] +      # Substituent asymmetry
                    0.15 * induct_n[i] +      # Inductive gradient
                    0.10 * hyper_n[i] +       # Hyperconjugation
                    0.10 * vib_n[i] +         # Vibrational asymmetry
                    0.10 * jt_n[i] +          # Jahn-Teller
                    0.10 * pjt_n[i] +         # Pseudo-Jahn-Teller
                    0.15 * topo_n[i] +        # Topological (proven)
                    0.15 * flex_n[i])         # Flexibility (proven)
        else:
            # For aliphatic: standard model
            base = (0.30 * tunnel_n[i] +
                    0.25 * flex_n[i] +
                    0.20 * topo_n[i] +
                    0.10 * induct_n[i] +
                    0.15 * vib_n[i])
        
        # Chemical multipliers
        alpha_mult = 1.0
        for nbr in atom.GetNeighbors():
            z = nbr.GetAtomicNum()
            if z == 7: alpha_mult = max(alpha_mult, 1.72)
            elif z == 8: alpha_mult = max(alpha_mult, 1.87)
            elif z == 16: alpha_mult = max(alpha_mult, 1.77)
        
        benz_mult = 1.0
        if not is_arom and n_H > 0:
            if any(nbr.GetIsAromatic() for nbr in atom.GetNeighbors()):
                benz_mult = 1.57
        
        n_C = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 6)
        tert_mult = 1.0 + 0.10 * (n_C - 1) if n_C > 1 else 1.0
        
        h_factor = (1 + 0.10 * n_H) if n_H > 0 else 0.5
        
        scores[i] = base * alpha_mult * benz_mult * tert_mult * h_factor
    
    return scores


def evaluate(data_path, sources=None, limit=None):
    """Evaluate the deep symmetry model."""
    with open(data_path) as f:
        data = json.load(f)['drugs']
    
    if sources:
        data = [d for d in data if d.get('source') in sources]
    if limit:
        data = data[:limit]
    
    by_source = defaultdict(lambda: {'t1': 0, 't3': 0, 'n': 0, 'arom_t1': 0, 'arom_n': 0})
    
    for d in data:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        src = d.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        scores = deep_symmetry_score(smiles)
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
        
        # Track aromatic
        site = sites[0]
        if site < mol.GetNumAtoms():
            if mol.GetAtomWithIdx(site).GetIsAromatic():
                by_source[src]['arom_n'] += 1
                if hit1:
                    by_source[src]['arom_t1'] += 1
    
    print("\n" + "="*70)
    print("DEEP SYMMETRY BREAKING MODEL")
    print("="*70)
    print("\nEffects: Substituent asymmetry, Inductive gradient, Hyperconjugation,")
    print("         Vibrational asymmetry, Jahn-Teller, Pseudo-Jahn-Teller")
    
    print(f"\n{'Source':<15} {'N':>6} {'Top-1':>10} {'Top-3':>10} {'Arom':>12}")
    print("-" * 55)
    
    for src in ['AZ120', 'Zaretzki', 'DrugBank', 'MetXBioDB']:
        s = by_source.get(src, {'t1': 0, 't3': 0, 'n': 0, 'arom_t1': 0, 'arom_n': 0})
        if s['n'] > 0:
            arom_str = f"{s['arom_t1']}/{s['arom_n']}" if s['arom_n'] > 0 else "-"
            print(f"{src:<15} {s['n']:>6} {s['t1']/s['n']*100:>9.1f}% {s['t3']/s['n']*100:>9.1f}% {arom_str:>12}")
    
    return by_source


if __name__ == '__main__':
    print("Testing Deep Symmetry Breaking Model...")
    evaluate('/home/claude/enzyme_Software/data/curated/merged_cyp3a4_extended.json', 
             sources=['AZ120'])
