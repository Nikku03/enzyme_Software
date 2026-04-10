"""
QUANTUM FORCES MODEL FOR SoM PREDICTION

At the quantum scale, we see FORCES, not atoms:

1. COULOMB FORCES - Electrostatic attraction/repulsion
   - Fe=O is highly polar (δ+ on Fe, δ- on O)
   - Attacks regions of opposite charge

2. VAN DER WAALS / DISPERSION FORCES
   - Temporary dipoles from electron fluctuations
   - Attracts Fe=O to polarizable regions
   - London dispersion ~ α₁α₂/r⁶

3. PAULI REPULSION
   - Electron clouds can't overlap (exclusion principle)
   - Creates "steric" barrier
   - Exponential decay with distance

4. EXCHANGE FORCES
   - Quantum mechanical electron exchange
   - Creates bonding/antibonding character
   - Depends on spin state

5. TUNNELING
   - H can tunnel through classically forbidden barriers
   - Rate ~ exp(-2γa) where γ = √(2mV)/ℏ
   - Favors light atoms (H vs D)

6. ZERO-POINT ENERGY
   - Even at T=0, bonds vibrate
   - ZPE = ℏω/2 for each mode
   - Lowers effective barrier

7. DECOHERENCE
   - Environment "measures" the quantum state
   - Collapses superposition
   - Rate depends on coupling to bath

The SoM is where these forces BALANCE to minimize the activation energy.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import json


# Physical constants (in atomic units where useful)
BOHR_TO_ANGSTROM = 0.529177
HARTREE_TO_KCAL = 627.509
MASS_H = 1.008  # amu
MASS_D = 2.014  # amu
HBAR = 1.0  # in atomic units


def get_3d_coords(mol):
    """Get 3D coordinates, generating if necessary."""
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except:
        AllChem.Compute2DCoords(mol)
    
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.zeros((n, 3))
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]
    
    return mol, coords


def compute_coulomb_potential(mol, coords):
    """
    Compute local electrostatic potential at each atom.
    
    Fe=O (δ+ on Fe, δ- on O) is attracted to:
    - Electron-rich regions (negative potential)
    - But not TOO negative (Pauli repulsion)
    
    Optimal: slightly negative, with polarizable neighbors
    """
    n = mol.GetNumAtoms()
    
    # Gasteiger partial charges as proxy
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = np.array([mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') 
                           for i in range(n)])
        charges = np.nan_to_num(charges, 0)
    except:
        # Fallback: electronegativity-based
        EN = {1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 
              15: 2.19, 16: 2.58, 17: 3.16, 35: 2.96}
        charges = np.array([EN.get(mol.GetAtomWithIdx(i).GetAtomicNum(), 2.5) - 2.5 
                           for i in range(n)])
    
    # Electrostatic potential at each site (from all other atoms)
    potential = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                r = np.linalg.norm(coords[i] - coords[j])
                if r > 0.5:  # Avoid singularity
                    potential[i] += charges[j] / r
    
    return potential, charges


def compute_dispersion_attraction(mol, coords):
    """
    London dispersion / van der Waals attraction.
    
    V_disp = -C₆/r⁶
    C₆ ∝ α₁ × α₂ × IP₁ × IP₂ / (IP₁ + IP₂)
    
    Polarizable atoms attract Fe=O more strongly.
    """
    n = mol.GetNumAtoms()
    
    # Atomic polarizabilities (Å³)
    ALPHA = {1: 0.67, 6: 1.76, 7: 1.10, 8: 0.80, 9: 0.56,
             15: 3.63, 16: 2.90, 17: 2.18, 35: 3.05}
    
    # Ionization potentials (eV)
    IP = {1: 13.6, 6: 11.3, 7: 14.5, 8: 13.6, 9: 17.4,
          15: 10.5, 16: 10.4, 17: 13.0, 35: 11.8}
    
    dispersion = np.zeros(n)
    
    for i in range(n):
        atom_i = mol.GetAtomWithIdx(i)
        z_i = atom_i.GetAtomicNum()
        alpha_i = ALPHA.get(z_i, 1.5)
        ip_i = IP.get(z_i, 12.0)
        
        # Sum dispersion interactions with neighbors
        disp = 0.0
        for nbr in atom_i.GetNeighbors():
            j = nbr.GetIdx()
            z_j = nbr.GetAtomicNum()
            alpha_j = ALPHA.get(z_j, 1.5)
            ip_j = IP.get(z_j, 12.0)
            
            r = np.linalg.norm(coords[i] - coords[j])
            if r > 0.5:
                # London formula
                C6 = 1.5 * alpha_i * alpha_j * ip_i * ip_j / (ip_i + ip_j)
                disp += C6 / r**6
        
        dispersion[i] = disp
    
    return dispersion


def compute_pauli_repulsion(mol, coords):
    """
    Pauli exclusion creates steric repulsion.
    
    V_Pauli ∝ exp(-β × r) × ρ₁ × ρ₂
    
    Crowded atoms have higher Pauli repulsion → less accessible to Fe=O.
    """
    n = mol.GetNumAtoms()
    
    # Atomic radii (Å) - for steric bulk
    RADII = {1: 0.53, 6: 0.77, 7: 0.75, 8: 0.73, 9: 0.71,
             15: 1.06, 16: 1.02, 17: 0.99, 35: 1.14}
    
    pauli = np.zeros(n)
    
    for i in range(n):
        atom_i = mol.GetAtomWithIdx(i)
        z_i = atom_i.GetAtomicNum()
        r_i = RADII.get(z_i, 0.8)
        
        # Sum repulsion from nearby atoms
        repulsion = 0.0
        for j in range(n):
            if i != j:
                z_j = mol.GetAtomWithIdx(j).GetAtomicNum()
                r_j = RADII.get(z_j, 0.8)
                
                d = np.linalg.norm(coords[i] - coords[j])
                if d > 0.5:
                    # Exponential repulsion
                    overlap = (r_i + r_j) / d
                    if overlap > 0.5:
                        repulsion += np.exp(-2.0 * (d - r_i - r_j))
        
        pauli[i] = repulsion
    
    return pauli


def compute_tunneling_rate(mol, coords, eigenvectors, eigenvalues):
    """
    Quantum tunneling through the activation barrier.
    
    Γ_tunnel ∝ exp(-2 × √(2m V) × a / ℏ)
    
    For H-abstraction:
    - Lighter H → more tunneling
    - Lower barrier V → more tunneling
    - Narrower barrier a → more tunneling
    
    We approximate:
    - V from flexibility (rigid = high barrier)
    - a from bond order (higher BO = narrower)
    """
    n = mol.GetNumAtoms()
    
    tunneling = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        z = atom.GetAtomicNum()
        
        if z != 6:
            continue
        
        n_H = atom.GetTotalNumHs()
        if n_H == 0:
            tunneling[i] = 0.1  # Aromatic has some tunneling too
            continue
        
        # Barrier height proxy: rigidity
        n_atoms = mol.GetNumAtoms()
        rigidity = sum(eigenvectors[i, k]**2 for k in range(max(1, n_atoms-3), n_atoms))
        V_barrier = 10.0 * rigidity  # kcal/mol scale
        
        # Barrier width proxy: inverse of flexibility
        # Flexible = narrow barrier (atoms can move toward each other)
        flexibility = 1.0 / (rigidity + 0.1)
        a_width = 1.0 / (flexibility + 0.1)  # Å scale
        
        # Alpha-heteroatom lowers AND narrows barrier
        for nbr in atom.GetNeighbors():
            nbr_z = nbr.GetAtomicNum()
            if nbr_z in [7, 8, 16]:
                V_barrier *= 0.7  # 30% lower
                a_width *= 0.8   # 20% narrower
        
        # Tunneling exponent (simplified)
        # Γ ∝ exp(-2 × √(2m V) × a / ℏ)
        # Using m = 1 amu for H, and approximate units
        gamma = np.sqrt(2 * MASS_H * V_barrier)
        tunnel_exp = np.exp(-0.5 * gamma * a_width)  # Scaled for numerical stability
        
        tunneling[i] = tunnel_exp * n_H  # More H = more tunneling paths
    
    return tunneling


def compute_zero_point_energy(mol, eigenvectors, eigenvalues):
    """
    Zero-point energy contribution.
    
    ZPE = Σ ℏω_k/2 for each mode
    
    Sites with high ZPE participation have more "quantum jitter"
    which helps overcome barriers.
    """
    n = mol.GetNumAtoms()
    
    zpe = np.zeros(n)
    
    for i in range(n):
        e = 0.0
        for k in range(1, n):
            if eigenvalues[k] > 1e-6:
                omega = np.sqrt(eigenvalues[k])
                zpe_k = 0.5 * omega  # ℏ = 1
                participation = eigenvectors[i, k]**2
                e += zpe_k * participation
        
        zpe[i] = e
    
    return zpe


def compute_decoherence_factor(mol, coords):
    """
    Decoherence rate from environment coupling.
    
    Strong coupling to environment → fast decoherence → classical behavior
    Weak coupling → slow decoherence → quantum coherence preserved
    
    For reactions, we want SOME coherence to enable tunneling,
    but not too much (need to collapse to product state).
    
    Solvent-exposed, flexible regions have optimal decoherence.
    """
    n = mol.GetNumAtoms()
    
    # Solvent accessibility proxy: distance from center of mass
    com = coords.mean(axis=0)
    dist_from_com = np.linalg.norm(coords - com, axis=1)
    
    # Normalize
    max_dist = dist_from_com.max() + 0.1
    accessibility = dist_from_com / max_dist
    
    # Number of neighbors affects decoherence
    n_neighbors = np.zeros(n)
    for i in range(n):
        n_neighbors[i] = len(mol.GetAtomWithIdx(i).GetNeighbors())
    
    # Decoherence factor: accessible and not too crowded
    decoherence = accessibility / (1 + 0.2 * n_neighbors)
    
    return decoherence


def compute_exchange_coupling(mol, eigenvectors, eigenvalues):
    """
    Exchange coupling between electrons.
    
    J = 2K - J_Coulomb (in Heisenberg model)
    
    Sites with strong exchange coupling to neighbors
    can facilitate spin state changes in the reaction.
    """
    n = mol.GetNumAtoms()
    
    exchange = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        # Exchange with bonded neighbors
        ex = 0.0
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            bond = mol.GetBondBetweenAtoms(i, j)
            bo = bond.GetBondTypeAsDouble() if bond else 1.0
            if bond and bond.GetIsAromatic():
                bo = 1.5
            
            # Exchange coupling proportional to bond order
            # and orbital overlap (from eigenvector similarity)
            overlap = sum(eigenvectors[i, k] * eigenvectors[j, k] 
                         for k in range(1, n))
            ex += bo * abs(overlap)
        
        exchange[i] = ex
    
    return exchange


def quantum_forces_som(smiles, debug=False):
    """
    Full quantum forces model for SoM prediction.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    n_orig = mol.GetNumAtoms()
    
    # Get 3D structure
    mol_3d, coords = get_3d_coords(mol)
    n = mol_3d.GetNumAtoms()
    
    # Graph Laplacian (on original heavy-atom structure)
    mol_noH = Chem.RemoveHs(mol_3d)
    n_noH = mol_noH.GetNumAtoms()
    
    A = np.zeros((n_noH, n_noH))
    for bond in mol_noH.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic():
            w = 1.5
        A[i, j] = A[j, i] = w
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Map heavy atom indices
    heavy_idx = [i for i in range(n) if mol_3d.GetAtomWithIdx(i).GetAtomicNum() > 1]
    
    # Compute all forces on heavy atom structure
    heavy_coords = coords[heavy_idx]
    
    potential, charges = compute_coulomb_potential(mol_noH, heavy_coords)
    dispersion = compute_dispersion_attraction(mol_noH, heavy_coords)
    pauli = compute_pauli_repulsion(mol_noH, heavy_coords)
    tunneling = compute_tunneling_rate(mol_noH, heavy_coords, eigenvectors, eigenvalues)
    zpe = compute_zero_point_energy(mol_noH, eigenvectors, eigenvalues)
    decoherence = compute_decoherence_factor(mol_noH, heavy_coords)
    exchange = compute_exchange_coupling(mol_noH, eigenvectors, eigenvalues)
    
    # Classic flexibility
    flexibility = np.zeros(n_noH)
    for i in range(n_noH):
        rigidity = sum(eigenvectors[i, k]**2 for k in range(max(1, n_noH-3), n_noH))
        flexibility[i] = 1.0 / (rigidity + 0.1)
    
    # Normalize each factor
    def norm(x):
        x = np.array(x)
        r = x.max() - x.min()
        if r > 1e-10:
            return (x - x.min()) / r
        return np.ones_like(x) * 0.5
    
    potential_n = norm(-potential)  # Negative potential is attractive
    dispersion_n = norm(dispersion)
    pauli_n = norm(-pauli)  # Low repulsion is good
    tunneling_n = norm(tunneling)
    zpe_n = norm(zpe)
    decoherence_n = norm(decoherence)
    exchange_n = norm(exchange)
    flexibility_n = norm(flexibility)
    
    if debug:
        print(f"  Coulomb potential: min={potential.min():.3f}, max={potential.max():.3f}")
        print(f"  Dispersion: min={dispersion.min():.3f}, max={dispersion.max():.3f}")
        print(f"  Pauli repulsion: min={pauli.min():.3f}, max={pauli.max():.3f}")
        print(f"  Tunneling: min={tunneling.min():.3f}, max={tunneling.max():.3f}")
    
    scores = np.full(n_noH, -np.inf)
    
    for i in range(n_noH):
        atom = mol_noH.GetAtomWithIdx(i)
        z = atom.GetAtomicNum()
        
        if z != 6:
            continue
        
        n_H = atom.GetTotalNumHs()
        is_arom = atom.GetIsAromatic()
        
        if n_H == 0 and not is_arom:
            continue
        
        # === QUANTUM FORCES SCORE ===
        # Each force contributes with a weight
        q_score = (
            0.20 * flexibility_n[i] +      # Proven to work
            0.15 * tunneling_n[i] +         # Quantum tunneling
            0.12 * potential_n[i] +         # Electrostatic attraction
            0.12 * dispersion_n[i] +        # van der Waals
            0.10 * pauli_n[i] +             # Low steric repulsion
            0.10 * zpe_n[i] +               # Zero-point energy
            0.08 * decoherence_n[i] +       # Optimal decoherence
            0.08 * exchange_n[i]            # Exchange coupling
        )
        
        # === CHEMICAL MULTIPLIERS ===
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


def evaluate(data_path, limit=None):
    """Evaluate on dataset with detailed source breakdown."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data['drugs']
    if limit:
        drugs = drugs[:limit]
    
    top1 = top3 = total = 0
    by_source = {}
    
    for d in drugs:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        src = d.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        scores = quantum_forces_som(smiles)
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
    
    print(f"\n{'='*60}")
    print(f"QUANTUM FORCES MODEL - SITE OF METABOLISM PREDICTION")
    print(f"{'='*60}")
    print(f"\nOVERALL: Top-1 = {top1}/{total} ({top1/total*100:.1f}%), Top-3 = {top3}/{total} ({top3/total*100:.1f}%)\n")
    
    print(f"{'Source':<20} {'N':>6} {'Top-1':>10} {'Top-3':>10}")
    print("-" * 50)
    for src, s in sorted(by_source.items(), key=lambda x: -x[1]['n']):
        if s['n'] >= 3:
            t1_pct = s['t1']/s['n']*100
            t3_pct = s['t3']/s['n']*100
            print(f"{src:<20} {s['n']:>6} {t1_pct:>9.1f}% {t3_pct:>9.1f}%")
    
    # Highlight Zaretzki for comparison
    if 'Zaretzki' in by_source:
        z = by_source['Zaretzki']
        print(f"\n{'='*60}")
        print(f"ZARETZKI BENCHMARK: Top-1 = {z['t1']/z['n']*100:.1f}%, Top-3 = {z['t3']/z['n']*100:.1f}%")
        print(f"{'='*60}")
    
    return top1/total if total > 0 else 0


def quantum_forces_som_fast(smiles):
    """
    Fast quantum forces model without 3D optimization.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
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
    
    # 1. Flexibility
    flexibility = np.zeros(n)
    for i in range(n):
        rigidity = sum(eigenvectors[i, k]**2 for k in range(max(1, n-3), n))
        flexibility[i] = 1.0 / (rigidity + 0.1)
    
    # 2. Amplitude (electron density)
    amplitude = np.zeros(n)
    for i in range(n):
        amplitude[i] = sum(eigenvectors[i, k]**2 / (eigenvalues[k] + 0.1) 
                          for k in range(1, min(5, n)))
    
    # 3. Tunneling
    tunneling = np.zeros(n)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            continue
        n_H = atom.GetTotalNumHs()
        base = flexibility[i] * (1 + 0.3 * n_H)
        barrier_mod = 1.0
        for nbr in atom.GetNeighbors():
            nbr_z = nbr.GetAtomicNum()
            if nbr_z == 7: barrier_mod *= 1.5
            elif nbr_z == 8: barrier_mod *= 1.4
            elif nbr_z == 16: barrier_mod *= 1.3
        tunneling[i] = base * barrier_mod
    
    # 4. van der Waals / Dispersion
    ALPHA = {1: 0.67, 6: 1.76, 7: 1.10, 8: 0.80, 16: 2.90}
    dispersion = np.zeros(n)
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        alpha_i = ALPHA.get(atom.GetAtomicNum(), 1.5)
        for nbr in atom.GetNeighbors():
            alpha_j = ALPHA.get(nbr.GetAtomicNum(), 1.5)
            dispersion[i] += alpha_i * alpha_j
    
    # 5. Pauli repulsion (steric)
    pauli = np.array([1.0 / (1 + 0.3 * sum(1 for nbr in mol.GetAtomWithIdx(i).GetNeighbors() 
                                           if nbr.GetAtomicNum() > 1)) for i in range(n)])
    
    # 6. Zero-point energy
    zpe = np.array([sum(np.sqrt(eigenvalues[k] + 0.01) * eigenvectors[i, k]**2 
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
    
    # Normalize
    def norm(x):
        x = np.array(x)
        r = x.max() - x.min()
        return (x - x.min()) / r if r > 1e-10 else np.ones_like(x) * 0.5
    
    flex_n, amp_n, tunnel_n = norm(flexibility), norm(amplitude), norm(tunneling)
    disp_n, pauli_n, zpe_n, exchange_n = norm(dispersion), norm(pauli), norm(zpe), norm(exchange)
    
    scores = np.full(n, -np.inf)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            continue
        
        n_H = atom.GetTotalNumHs()
        is_arom = atom.GetIsAromatic()
        
        if n_H == 0 and not is_arom:
            continue
        
        q_score = (0.25 * flex_n[i] + 0.20 * amp_n[i] + 0.15 * tunnel_n[i] +
                   0.12 * disp_n[i] + 0.10 * pauli_n[i] + 0.10 * zpe_n[i] + 0.08 * exchange_n[i])
        
        # Chemical multipliers
        alpha_mult = 1.0
        for nbr in atom.GetNeighbors():
            nbr_z = nbr.GetAtomicNum()
            if nbr_z == 7: alpha_mult = max(alpha_mult, 1.84)
            elif nbr_z == 8: alpha_mult = max(alpha_mult, 1.82)
            elif nbr_z == 16: alpha_mult = max(alpha_mult, 1.54)
        
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


def evaluate_fast(data_path, sources=None):
    """Fast evaluation with source filtering."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data['drugs']
    if sources:
        drugs = [d for d in drugs if d.get('source') in sources]
    
    top1 = top3 = total = 0
    by_source = {}
    
    for d in drugs:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        src = d.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        scores = quantum_forces_som_fast(smiles)
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
    
    print(f"\n{'='*60}")
    print(f"QUANTUM FORCES MODEL (FAST)")
    print(f"{'='*60}")
    print(f"OVERALL: Top-1 = {top1}/{total} ({top1/total*100:.1f}%), Top-3 = {top3}/{total} ({top3/total*100:.1f}%)\n")
    
    print(f"{'Source':<20} {'N':>6} {'Top-1':>10} {'Top-3':>10}")
    print("-" * 50)
    for src, s in sorted(by_source.items(), key=lambda x: -x[1]['n']):
        t1_pct = s['t1']/s['n']*100
        t3_pct = s['t3']/s['n']*100
        print(f"{src:<20} {s['n']:>6} {t1_pct:>9.1f}% {t3_pct:>9.1f}%")
    
    return by_source


if __name__ == '__main__':
    data_path = '/home/claude/enzyme_Software/data/curated/merged_cyp3a4_extended.json'
    print("Evaluating Quantum Forces Model (Fast) on full dataset...")
    evaluate_fast(data_path)
