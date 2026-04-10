"""
3D QUANTUM FIELD SIMULATION OF CYP-SUBSTRATE INTERACTION

The key insight: "Equivalent" atoms are only equivalent in ISOLATION.
When an enzyme approaches, it BREAKS THE SYMMETRY.

Consider a benzene ring:
- In isolation: 6 equivalent carbons (C6v symmetry)
- With enzyme approaching from one side: symmetry broken!
  - C1 faces Fe=O directly → strong coupling
  - C4 (para) faces away → weak coupling
  - C2, C6 (ortho) are at angle → intermediate coupling

WHAT CREATES THE 55-45 SPLIT?

1. GEOMETRIC COUPLING
   - Fe=O creates an electric field E(r) = dipole field
   - Each carbon at position r_i couples as: V_i = -μ·E(r_i)
   - Distance and angle to Fe=O differ for each carbon

2. VAN DER WAALS DIRECTIONALITY
   - vdW = -C6/r^6 is isotropic
   - But RETARDATION at larger distances: -C6/r^7 kicks in
   - Casimir-Polder effect creates directional preference

3. QUANTUM FLUCTUATION ASYMMETRY
   - Vacuum fluctuations couple electron motion
   - Fluctuations are correlated with Fe=O electron motion
   - Creates ENTANGLEMENT between specific C and Fe
   
4. PHOTON EXCHANGE GEOMETRY
   - Virtual photon exchange rate ∝ overlap integral
   - ⟨ψ_C|r|ψ_Fe⟩ depends on spatial orientation
   - Transition dipole has directional character

5. SPIN-ORBIT COUPLING ANISOTROPY
   - Fe has strong SOC (Z=26)
   - SOC creates preferred spin axis
   - Carbon coupling depends on angle to this axis

6. ENZYME POCKET ELECTROSTATICS
   - Active site isn't just Fe=O
   - Surrounding residues create electric field landscape
   - Field points toward specific carbons

Let's simulate all of this.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import json
from scipy.spatial.transform import Rotation
from collections import defaultdict

# Physical constants
BOHR = 0.529177  # Å
HARTREE = 627.509  # kcal/mol
ALPHA_FE = 8.4  # Polarizability of Fe (Å³)
ALPHA_O = 0.8  # Polarizability of O
C6_FE = 400.0  # C6 coefficient Fe (Hartree·Bohr^6)
MU_FEO = 2.5  # Dipole moment of Fe=O (Debye)

def get_3d_structure(mol):
    """Generate 3D structure with hydrogens."""
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except:
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            x, y = conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y
            conf.SetAtomPosition(i, (x, y, np.random.uniform(-0.5, 0.5)))
    
    return mol

def get_coords(mol):
    """Extract atomic coordinates."""
    conf = mol.GetConformer()
    coords = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] 
                       for i in range(mol.GetNumAtoms())])
    return coords

def find_aromatic_systems(mol):
    """Find aromatic ring systems and their centers/normals."""
    rings = mol.GetRingInfo().AtomRings()
    coords = get_coords(mol)
    
    aromatic_systems = []
    for ring in rings:
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            # Ring center
            ring_coords = coords[list(ring)]
            center = ring_coords.mean(axis=0)
            
            # Ring normal (cross product of two vectors in plane)
            v1 = ring_coords[1] - ring_coords[0]
            v2 = ring_coords[2] - ring_coords[0]
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            
            aromatic_systems.append({
                'atoms': list(ring),
                'center': center,
                'normal': normal,
                'coords': ring_coords
            })
    
    return aromatic_systems

def compute_enzyme_field(enzyme_pos, enzyme_orientation, grid_points):
    """
    Compute the electric field from Fe=O at grid points.
    
    Fe=O is modeled as:
    - A dipole (μ pointing from O to Fe)
    - An electrophilic center (δ+ on Fe)
    - A radical oxygen (unpaired electron on O)
    """
    # Dipole field: E = (3(p·r̂)r̂ - p) / r³
    r = grid_points - enzyme_pos
    r_mag = np.linalg.norm(r, axis=-1, keepdims=True)
    r_hat = r / (r_mag + 1e-10)
    
    # Dipole moment vector
    p = MU_FEO * enzyme_orientation
    
    # Dipole field
    p_dot_r = np.sum(p * r_hat, axis=-1, keepdims=True)
    E_dipole = (3 * p_dot_r * r_hat - p) / (r_mag**3 + 1e-6)
    
    # Charge field from δ+ on Fe (positioned along dipole direction)
    fe_pos = enzyme_pos + 0.5 * enzyme_orientation  # Fe is 0.5Å from center
    r_fe = grid_points - fe_pos
    r_fe_mag = np.linalg.norm(r_fe, axis=-1, keepdims=True)
    E_charge = 0.3 * r_fe / (r_fe_mag**3 + 1e-6)  # δ+ ≈ 0.3e
    
    return E_dipole + E_charge

def compute_vdw_potential(enzyme_pos, atom_coords, atom_types):
    """
    Compute van der Waals potential with retardation effects.
    
    At close range: V = -C6/r^6 (London dispersion)
    At long range: V = -C7/r^7 (Casimir-Polder retardation)
    
    The crossover creates DIRECTIONAL preference!
    """
    # C6 coefficients
    C6 = {6: 46.6, 7: 24.2, 8: 15.6, 1: 6.5}  # Hartree·Bohr^6
    
    r = atom_coords - enzyme_pos
    r_mag = np.linalg.norm(r, axis=1)
    
    vdw = np.zeros(len(atom_coords))
    for i, (coord, z) in enumerate(zip(atom_coords, atom_types)):
        c6 = np.sqrt(C6.get(z, 30.0) * C6_FE)
        r = r_mag[i]
        
        # Crossover distance (retardation kicks in)
        r_cross = 5.0  # Å
        
        if r < r_cross:
            # Close range: -C6/r^6
            vdw[i] = -c6 / (r**6 + 0.1)
        else:
            # Long range: -C7/r^7 (retarded)
            vdw[i] = -c6 * r_cross / (r**7 + 0.1)
    
    return vdw

def compute_photon_exchange_coupling(enzyme_pos, enzyme_orientation, atom_coords, atom_types):
    """
    Virtual photon exchange rate between Fe=O and substrate atoms.
    
    Rate ∝ |⟨ψ_final|r|ψ_initial⟩|² × overlap factor
    
    The transition dipole is DIRECTIONAL - depends on angle between
    the Fe-O axis and the Fe-C direction.
    """
    # Fe=O has a transition dipole along its axis
    coupling = np.zeros(len(atom_coords))
    
    for i, (coord, z) in enumerate(zip(atom_coords, atom_types)):
        if z != 6:  # Only carbons
            continue
        
        r = coord - enzyme_pos
        r_mag = np.linalg.norm(r)
        r_hat = r / (r_mag + 1e-10)
        
        # Angle between Fe-O axis and Fe-C direction
        cos_angle = np.dot(enzyme_orientation, r_hat)
        
        # Transition dipole coupling (maximum when aligned)
        # This is the key asymmetry factor!
        angular_factor = (1 + cos_angle) / 2  # 0 to 1
        
        # Distance decay
        distance_factor = 1 / (r_mag**2 + 1)
        
        coupling[i] = angular_factor * distance_factor
    
    return coupling

def compute_spin_orbit_coupling(enzyme_orientation, atom_coords, mol):
    """
    Spin-orbit coupling creates a preferred spin axis for Fe.
    
    Carbon atoms whose p-orbitals align with this axis
    couple more strongly to Fe spin states.
    
    This is the SOC anisotropy that breaks aromatic symmetry.
    """
    n = len(atom_coords)
    soc = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6 or not atom.GetIsAromatic():
            continue
        
        # Find ring normal (p-orbital direction for aromatic C)
        for nbr in atom.GetNeighbors():
            if nbr.GetIsAromatic():
                j = nbr.GetIdx()
                # Approximate p-orbital direction from neighbors
                r_ij = atom_coords[j] - atom_coords[i]
                break
        else:
            continue
        
        # Second neighbor for plane
        for nbr in atom.GetNeighbors():
            if nbr.GetIsAromatic() and nbr.GetIdx() != j:
                k = nbr.GetIdx()
                r_ik = atom_coords[k] - atom_coords[i]
                break
        else:
            continue
        
        # p-orbital is perpendicular to ring plane
        p_orbital = np.cross(r_ij, r_ik)
        p_orbital = p_orbital / (np.linalg.norm(p_orbital) + 1e-10)
        
        # SOC coupling depends on alignment with Fe spin axis
        # Fe spin axis is along Fe-O bond (enzyme_orientation)
        cos_soc = abs(np.dot(p_orbital, enzyme_orientation))
        
        soc[i] = cos_soc
    
    return soc

def compute_entanglement_correlation(enzyme_pos, atom_coords, mol, coords):
    """
    Quantum entanglement between electron fluctuations.
    
    When electrons in Fe=O fluctuate, they create correlated
    fluctuations in nearby atoms. The correlation strength
    depends on the electron cloud overlap.
    
    This creates position-dependent quantum correlations!
    """
    n = len(atom_coords)
    entanglement = np.zeros(n)
    
    # Graph Laplacian for electron correlations
    A = np.zeros((n, n))
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            continue
        
        r = atom_coords[i] - enzyme_pos
        r_mag = np.linalg.norm(r)
        
        # Distance-dependent entanglement decay
        # Entanglement ∝ exp(-r/ξ) where ξ is correlation length
        xi = 3.0  # Correlation length in Å
        dist_factor = np.exp(-r_mag / xi)
        
        # Delocalization factor (entangled electrons are delocalized)
        deloc = sum(eigenvectors[i, k]**2 / (eigenvalues[k] + 0.1) 
                   for k in range(1, min(5, n)))
        
        entanglement[i] = dist_factor * deloc
    
    return entanglement

def sample_enzyme_approaches(mol, n_samples=50):
    """
    Sample different enzyme approach geometries.
    
    The enzyme can approach from many directions. We need to
    find which carbons are favored across the ENSEMBLE of
    possible approaches.
    """
    coords = get_coords(mol)
    n = mol.GetNumAtoms()
    
    # Find reactive carbons (aromatic or with H)
    reactive = []
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() == 6:
            if atom.GetIsAromatic() or atom.GetTotalNumHs() > 0:
                reactive.append(i)
    
    if not reactive:
        return None
    
    # Center of reactive region
    reactive_coords = coords[reactive]
    center = reactive_coords.mean(axis=0)
    
    # Sample approach directions uniformly on sphere
    approaches = []
    for _ in range(n_samples):
        # Random direction on sphere
        phi = np.random.uniform(0, 2*np.pi)
        cos_theta = np.random.uniform(-1, 1)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        direction = np.array([
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta
        ])
        
        # Enzyme position (approach from 4-6 Å away)
        distance = np.random.uniform(4.0, 6.0)
        enzyme_pos = center + distance * direction
        
        # Enzyme points TOWARD the molecule
        enzyme_orientation = -direction
        
        approaches.append((enzyme_pos, enzyme_orientation))
    
    return approaches, coords, reactive

def compute_3d_quantum_score(mol):
    """
    Full 3D quantum field score for each atom.
    
    This aggregates:
    1. Electric field coupling from enzyme dipole
    2. van der Waals with retardation
    3. Photon exchange coupling
    4. Spin-orbit coupling anisotropy
    5. Quantum entanglement correlations
    6. Geometric accessibility
    
    Averaged over ensemble of approach directions.
    """
    result = sample_enzyme_approaches(mol)
    if result is None:
        return None
    
    approaches, coords, reactive = result
    n = mol.GetNumAtoms()
    atom_types = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)]
    
    # Aggregate scores over all approaches
    aggregate_scores = np.zeros(n)
    
    for enzyme_pos, enzyme_orientation in approaches:
        # 1. Electric field coupling
        E_field = compute_enzyme_field(enzyme_pos, enzyme_orientation, coords)
        E_magnitude = np.linalg.norm(E_field, axis=1)
        
        # 2. van der Waals
        vdw = compute_vdw_potential(enzyme_pos, coords, atom_types)
        
        # 3. Photon exchange
        photon = compute_photon_exchange_coupling(
            enzyme_pos, enzyme_orientation, coords, atom_types
        )
        
        # 4. Spin-orbit coupling
        soc = compute_spin_orbit_coupling(enzyme_orientation, coords, mol)
        
        # 5. Entanglement
        entangle = compute_entanglement_correlation(enzyme_pos, coords, mol, coords)
        
        # 6. Geometric accessibility (solid angle visible to enzyme)
        accessibility = np.zeros(n)
        for i in reactive:
            r = coords[i] - enzyme_pos
            r_hat = r / (np.linalg.norm(r) + 1e-10)
            
            # Check if this carbon faces the enzyme
            # (use angle between C position and enzyme approach)
            facing = np.dot(-r_hat, enzyme_orientation)
            accessibility[i] = max(0, facing)  # 0 if facing away
        
        # Combine all contributions
        approach_score = (
            0.20 * E_magnitude +
            0.15 * (-vdw) +  # vdw is negative, more negative = better
            0.25 * photon +
            0.15 * soc +
            0.15 * entangle +
            0.10 * accessibility
        )
        
        aggregate_scores += approach_score
    
    # Normalize by number of approaches
    aggregate_scores /= len(approaches)
    
    return aggregate_scores

def compute_final_som_score(smiles):
    """
    Final SoM prediction combining 3D quantum fields with chemical knowledge.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    mol = get_3d_structure(mol)
    mol_noH = Chem.RemoveAllHs(mol)
    
    # Map H-added indices to original
    n_orig = mol_noH.GetNumAtoms()
    
    # Get 3D quantum scores
    q3d_scores = compute_3d_quantum_score(mol_noH)
    if q3d_scores is None:
        return None
    
    # Normalize
    q3d_scores = (q3d_scores - q3d_scores.min()) / (q3d_scores.max() - q3d_scores.min() + 1e-10)
    
    # Graph Laplacian features (our proven baseline)
    n = n_orig
    A = np.zeros((n, n))
    for bond in mol_noH.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        w = bond.GetBondTypeAsDouble()
        if bond.GetIsAromatic(): w = 1.5
        A[i, j] = A[j, i] = w
    
    L = np.diag(A.sum(axis=1)) - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Flexibility
    flex = np.array([1.0 / (sum(eigenvectors[i, k]**2 
                    for k in range(max(1, n-3), n)) + 0.1) for i in range(n)])
    flex = (flex - flex.min()) / (flex.max() - flex.min() + 1e-10)
    
    # Topological charge
    topo = np.zeros(n)
    for i in range(n):
        phase = sum(abs(eigenvectors[i, k] - eigenvectors[j, k]) 
                   for nbr in mol_noH.GetAtomWithIdx(i).GetNeighbors() 
                   for j in [nbr.GetIdx()] for k in range(1, min(5, n)))
        topo[i] = 1.0 / (phase + 0.1)
    topo = (topo - topo.min()) / (topo.max() - topo.min() + 1e-10)
    
    # Final scores
    scores = np.full(n, -np.inf)
    
    for i in range(n):
        atom = mol_noH.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            continue
        
        n_H = atom.GetTotalNumHs()
        is_arom = atom.GetIsAromatic()
        
        if n_H == 0 and not is_arom:
            continue
        
        # Combine 3D quantum field with graph features
        if is_arom:
            # For aromatic: 3D field is MORE important (breaks symmetry!)
            base_score = 0.50 * q3d_scores[i] + 0.30 * topo[i] + 0.20 * flex[i]
        else:
            # For aliphatic: both matter
            base_score = 0.40 * q3d_scores[i] + 0.35 * flex[i] + 0.25 * topo[i]
        
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
        
        scores[i] = base_score * alpha_mult * benz_mult * h_factor
    
    return scores


def evaluate_3d_model(data_path, limit=None):
    """Evaluate the 3D quantum field model."""
    with open(data_path) as f:
        data = json.load(f)['drugs']
    
    if limit:
        data = data[:limit]
    
    by_source = defaultdict(lambda: {'t1': 0, 't3': 0, 'n': 0})
    
    for d in data:
        smiles = d.get('smiles', '')
        sites = d.get('site_atoms', [])
        src = d.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        scores = compute_final_som_score(smiles)
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
    print("3D QUANTUM FIELD MODEL - SYMMETRY BREAKING")
    print("="*70)
    print("\nFeatures: Electric field, vdW retardation, photon exchange,")
    print("          spin-orbit coupling, quantum entanglement, 3D accessibility")
    
    print(f"\n{'Source':<15} {'N':>6} {'Top-1':>10} {'Top-3':>10}")
    print("-" * 45)
    
    for src in ['AZ120', 'Zaretzki', 'DrugBank', 'MetXBioDB']:
        s = by_source.get(src, {'t1': 0, 't3': 0, 'n': 0})
        if s['n'] > 0:
            print(f"{src:<15} {s['n']:>6} {s['t1']/s['n']*100:>9.1f}% {s['t3']/s['n']*100:>9.1f}%")
    
    return by_source


if __name__ == '__main__':
    print("Testing 3D Quantum Field Model...")
    evaluate_3d_model('/home/claude/enzyme_Software/data/curated/merged_cyp3a4_extended.json', limit=100)
