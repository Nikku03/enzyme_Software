#!/usr/bin/env python3
"""
QUANTUM FIELD THEORY OF METABOLISM
===================================

Going to the deepest level.

A molecule is not atoms connected by bonds.
A molecule is a STABLE CONFIGURATION of a quantum field.

A chemical reaction is a PHASE TRANSITION - the field reorganizes
from one stable pattern to another.

The enzyme doesn't just "perturb" - it COUPLES TO THE FIELD and
creates new low-energy pathways between configurations.

The reaction happens where:
1. The field can transition with minimum action
2. The enzyme coupling is strongest
3. The new configuration (product) is accessible

This is path integral territory. The reaction rate is:

    k ∝ ∫ D[ψ] exp(-S[ψ]/ℏ)

Where S[ψ] is the action along the path ψ(t).

The dominant contribution comes from the classical path - 
the minimum action trajectory.

For our purposes:
- Find the "instanton" - the tunneling path
- The site where this path originates is the reaction site

This is equivalent to finding where the EFFECTIVE POTENTIAL
has the lowest barrier when the enzyme field is present.
"""

import numpy as np
from scipy import linalg, optimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("RDKit required")


# =============================================================================
# CONFIGURATION SPACE REPRESENTATION
# =============================================================================

class MolecularField:
    """
    Represents the molecule as a quantum field in configuration space.
    
    The field ψ(x) lives on a graph (the molecular skeleton).
    Each atom is a point in configuration space.
    The field value at each point represents electron density/phase.
    """
    
    def __init__(self, mol):
        self.mol = mol
        self.n_atoms = mol.GetNumAtoms()
        
        # Build the kinetic operator (Laplacian)
        self.T = self._build_kinetic()
        
        # Build the potential (from electronegativity)
        self.V = self._build_potential()
        
        # Full Hamiltonian
        self.H = self.T + np.diag(self.V)
        
        # Solve for eigenstates
        self.energies, self.states = np.linalg.eigh(self.H)
        
        # Ground state
        self.psi_0 = self.states[:, 0]
        self.E_0 = self.energies[0]
    
    def _build_kinetic(self) -> np.ndarray:
        """Kinetic energy = negative Laplacian."""
        n = self.n_atoms
        T = np.zeros((n, n))
        
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            w = bond.GetBondTypeAsDouble()
            if bond.GetIsAromatic():
                w = 1.5
            T[i, j] = T[j, i] = -w  # Hopping
            T[i, i] += w  # Diagonal
            T[j, j] += w
        
        return T
    
    def _build_potential(self) -> np.ndarray:
        """Potential from atomic electronegativities."""
        V = np.zeros(self.n_atoms)
        chi_map = {1: 2.2, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 
                   16: 2.58, 17: 3.16, 35: 2.96}
        
        for i in range(self.n_atoms):
            atom = self.mol.GetAtomWithIdx(i)
            V[i] = chi_map.get(atom.GetAtomicNum(), 2.5) - 2.55  # Center on carbon
        
        return V
    
    def get_density(self) -> np.ndarray:
        """Get electron density at each atom from occupied orbitals."""
        n_occ = self.n_atoms // 2 + 1
        density = np.zeros(self.n_atoms)
        for k in range(n_occ):
            density += 2 * self.states[:, k]**2
        return density
    
    def get_gradient(self) -> np.ndarray:
        """Get gradient of the field (indicates flow direction)."""
        # Gradient = T @ psi_0 (how field wants to flow)
        return self.T @ self.psi_0


# =============================================================================
# ENZYME FIELD
# =============================================================================

class EnzymeField:
    """
    The enzyme as a field that couples to the molecular field.
    
    The Fe=O creates an electric field and exchange interaction.
    This deforms the molecular potential energy surface.
    """
    
    def __init__(self, mol, approach_site: int):
        self.mol = mol
        self.n_atoms = mol.GetNumAtoms()
        self.approach_site = approach_site
        
        # Build enzyme coupling field
        self.coupling = self._build_coupling()
    
    def _build_coupling(self) -> np.ndarray:
        """Build the enzyme-molecule coupling operator."""
        n = self.n_atoms
        
        # Distance from approach site (graph distance)
        distances = self._graph_distances(self.approach_site)
        
        # Coupling decays with distance
        decay = 1.5  # Characteristic length
        coupling = np.exp(-distances / decay)
        
        # Normalize
        coupling = coupling / coupling.max()
        
        return coupling
    
    def _graph_distances(self, source: int) -> np.ndarray:
        """Compute graph distances from source atom."""
        n = self.n_atoms
        distances = np.full(n, np.inf)
        distances[source] = 0
        
        # BFS
        queue = [source]
        visited = {source}
        
        while queue:
            current = queue.pop(0)
            atom = self.mol.GetAtomWithIdx(current)
            for neighbor in atom.GetNeighbors():
                j = neighbor.GetIdx()
                if j not in visited:
                    distances[j] = distances[current] + 1
                    visited.add(j)
                    queue.append(j)
        
        return distances
    
    def perturb_hamiltonian(self, H: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Apply enzyme perturbation to molecular Hamiltonian."""
        # Add attractive potential at approach site
        H_pert = H.copy()
        
        # Direct coupling
        for i in range(self.n_atoms):
            H_pert[i, i] -= strength * self.coupling[i]
        
        # Also affects hopping (weakens bonds near enzyme)
        for i in range(self.n_atoms):
            for j in range(i+1, self.n_atoms):
                if H[i, j] != 0:  # Bond exists
                    weakening = 1 - 0.3 * (self.coupling[i] + self.coupling[j]) / 2
                    H_pert[i, j] *= weakening
                    H_pert[j, i] *= weakening
        
        return H_pert


# =============================================================================
# TRANSITION PATH ANALYSIS
# =============================================================================

class TransitionPath:
    """
    Find the minimum action path between configurations.
    
    The reaction proceeds along the path of least action.
    This is the "instanton" in quantum field theory language.
    """
    
    def __init__(self, mol_field: MolecularField, enzyme_field: EnzymeField):
        self.mol = mol_field
        self.enz = enzyme_field
        
        # Perturbed Hamiltonian
        self.H_pert = enzyme_field.perturb_hamiltonian(mol_field.H)
        
        # New ground state
        self.E_pert, self.states_pert = np.linalg.eigh(self.H_pert)
        self.psi_pert = self.states_pert[:, 0]
    
    def compute_barrier(self) -> float:
        """
        Compute the energy barrier for transition.
        
        Barrier = max energy along minimum energy path - initial energy.
        """
        # Simple approximation: barrier is related to energy difference
        # between ground states and their mixing
        
        # Overlap between original and perturbed ground states
        overlap = np.abs(np.dot(self.mol.psi_0, self.psi_pert))
        
        # Low overlap = must go through high-energy intermediate
        barrier = (1 - overlap) * (self.E_pert[1] - self.E_pert[0])
        
        return barrier
    
    def compute_instanton_action(self) -> float:
        """
        Compute the action along the instanton path.
        
        Lower action = faster tunneling = more likely reaction.
        
        S = ∫ dt [T + V] along the path
        """
        n = self.mol.n_atoms
        
        # The instanton connects psi_0 to psi_pert
        # Parameterize path as linear interpolation (crude approximation)
        
        n_steps = 20
        action = 0.0
        
        for step in range(n_steps):
            t = step / n_steps
            
            # Interpolated wave function
            psi_t = (1 - t) * self.mol.psi_0 + t * self.psi_pert
            psi_t = psi_t / np.linalg.norm(psi_t)
            
            # Kinetic energy
            T = np.dot(psi_t, self.mol.T @ psi_t)
            
            # Potential energy (interpolate Hamiltonian too)
            H_t = (1 - t) * self.mol.H + t * self.H_pert
            V = np.dot(psi_t, H_t @ psi_t) - T
            
            # Action contribution
            action += (T + V) / n_steps
        
        return action
    
    def get_transition_density(self) -> np.ndarray:
        """
        Where does the wave function change most during transition?
        
        This indicates where the reaction happens.
        """
        # Change in density
        rho_0 = self.mol.psi_0**2
        rho_pert = self.psi_pert**2
        
        delta_rho = np.abs(rho_pert - rho_0)
        
        return delta_rho


# =============================================================================
# EFFECTIVE POTENTIAL LANDSCAPE
# =============================================================================

def compute_effective_potential(mol, site: int) -> Dict:
    """
    Compute the effective potential landscape when enzyme approaches a site.
    
    Returns quantities that characterize the transition:
    - barrier: energy barrier
    - action: instanton action
    - transition_density: where density changes
    - coupling_strength: how strongly enzyme couples
    """
    mol_field = MolecularField(mol)
    enzyme_field = EnzymeField(mol, site)
    
    path = TransitionPath(mol_field, enzyme_field)
    
    return {
        'barrier': path.compute_barrier(),
        'action': path.compute_instanton_action(),
        'transition_density': path.get_transition_density()[site],
        'coupling': enzyme_field.coupling[site],
        'energy_lowering': mol_field.E_0 - path.E_pert[0],
    }


# =============================================================================
# MULTI-SCALE FIELD ANALYSIS
# =============================================================================

def compute_field_curvature(mol) -> np.ndarray:
    """
    Compute the curvature of the field at each point.
    
    High curvature = field is "bent" here = energy concentrated.
    In differential geometry: this is the Ricci curvature of the 
    configuration space.
    """
    n = mol.GetNumAtoms()
    field = MolecularField(mol)
    
    # Curvature from second derivative of potential
    # Approximate: curvature[i] = V[i] - average of neighbors
    
    curvature = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
        
        if neighbors:
            neighbor_avg = np.mean([field.V[j] for j in neighbors])
            curvature[i] = field.V[i] - neighbor_avg
    
    return curvature


def compute_field_torsion(mol) -> np.ndarray:
    """
    Compute the "torsion" of the field - how it twists in configuration space.
    
    This relates to chirality and stereochemistry of reaction.
    """
    n = mol.GetNumAtoms()
    field = MolecularField(mol)
    
    # Torsion from gradient of the wave function
    gradient = field.get_gradient()
    
    # Torsion = curl of gradient (measures rotation)
    # On a graph, approximate as circulation around triangles
    
    torsion = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
        
        if len(neighbors) >= 2:
            # Look at pairs of neighbors
            for j in range(len(neighbors)):
                for k in range(j+1, len(neighbors)):
                    n1, n2 = neighbors[j], neighbors[k]
                    # Circulation = g[i->n1] - g[n1->i] + g[n1->n2] - ...
                    # Simplified: just use gradient differences
                    circulation = (gradient[n1] - gradient[i]) - (gradient[n2] - gradient[i])
                    torsion[i] += np.abs(circulation)
    
    return torsion


def compute_field_divergence(mol) -> np.ndarray:
    """
    Compute divergence of the field - sources and sinks.
    
    Positive divergence = field flows OUT (electron donating)
    Negative divergence = field flows IN (electron accepting)
    """
    n = mol.GetNumAtoms()
    field = MolecularField(mol)
    
    # Divergence = Laplacian @ psi_0
    divergence = field.T @ field.psi_0
    
    return divergence


# =============================================================================
# TOPOLOGICAL ANALYSIS
# =============================================================================

def compute_topological_charge(mol) -> np.ndarray:
    """
    Compute topological charge at each atom.
    
    In gauge theory, topological charge measures "winding number" -
    how the phase of the field winds around.
    
    High topological charge = protected against small perturbations.
    Low charge = vulnerable to disruption = reactive.
    """
    n = mol.GetNumAtoms()
    field = MolecularField(mol)
    
    # Approximate topological charge from phase structure
    # Phase = angle of wave function (treat as complex)
    
    # Make wave function complex with position-dependent phase
    psi_complex = field.psi_0 * np.exp(1j * np.linspace(0, 2*np.pi, n))
    
    charge = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
        
        if neighbors:
            # Phase winding around neighbors
            phases = np.angle(psi_complex[neighbors])
            winding = np.sum(np.diff(np.sort(phases)))
            charge[i] = np.abs(winding / (2 * np.pi))
    
    return charge


def compute_berry_phase(mol, site: int) -> float:
    """
    Compute Berry phase accumulated by adiabatic evolution around a site.
    
    Non-zero Berry phase indicates geometric/topological protection.
    Zero Berry phase = trivial = easily disrupted.
    """
    field = MolecularField(mol)
    
    # Adiabatic loop: rotate perturbation around the site
    n_angles = 8
    berry_phase = 0.0
    
    psi_prev = field.psi_0.copy()
    
    for step in range(n_angles):
        angle = 2 * np.pi * step / n_angles
        
        # Perturbation at angle
        H_pert = field.H.copy()
        H_pert[site, site] += 0.1 * np.cos(angle)
        
        if site < field.n_atoms - 1:
            H_pert[site, site+1] += 0.05 * np.sin(angle)
            H_pert[site+1, site] += 0.05 * np.sin(angle)
        
        # Ground state of perturbed system
        E, V = np.linalg.eigh(H_pert)
        psi_curr = V[:, 0]
        
        # Ensure consistent phase
        if np.dot(psi_curr, psi_prev) < 0:
            psi_curr = -psi_curr
        
        # Berry connection: <psi|d/dθ|psi>
        connection = np.dot(psi_prev, psi_curr - psi_prev)
        berry_phase += np.imag(connection)
        
        psi_prev = psi_curr
    
    return np.abs(berry_phase)


# =============================================================================
# MAIN PREDICTOR
# =============================================================================

@dataclass
class FieldTheoryPrediction:
    """Prediction from field theory model."""
    smiles: str
    scores: np.ndarray
    top1: int
    top3: List[int]
    components: Dict[str, np.ndarray]


def predict_field_theory(smiles: str) -> Optional[FieldTheoryPrediction]:
    """
    Predict SoM using quantum field theory approach.
    
    Combines:
    1. Transition barrier (from enzyme coupling)
    2. Instanton action (tunneling likelihood)
    3. Field curvature (energy concentration)
    4. Field divergence (electron flow)
    5. Topological charge (protection level)
    6. Berry phase (geometric phase)
    7. Transition density (where change happens)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    
    n = mol.GetNumAtoms()
    
    # === COMPUTE ALL FIELD QUANTITIES ===
    
    # Field structure
    curvature = compute_field_curvature(mol)
    divergence = compute_field_divergence(mol)
    torsion = compute_field_torsion(mol)
    topo_charge = compute_topological_charge(mol)
    
    # Per-site transition analysis
    barriers = np.zeros(n)
    actions = np.zeros(n)
    trans_density = np.zeros(n)
    couplings = np.zeros(n)
    berry_phases = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            barriers[i] = np.inf
            actions[i] = np.inf
            continue
        
        # Effective potential analysis
        eff_pot = compute_effective_potential(mol, i)
        barriers[i] = eff_pot['barrier']
        actions[i] = eff_pot['action']
        trans_density[i] = eff_pot['transition_density']
        couplings[i] = eff_pot['coupling']
        
        # Berry phase
        berry_phases[i] = compute_berry_phase(mol, i)
    
    # === NORMALIZE ===
    def safe_norm(x, invert=False):
        valid = ~np.isinf(x)
        if valid.sum() == 0:
            return np.zeros_like(x)
        x_valid = x[valid]
        if invert:
            x_valid = -x_valid
        if x_valid.max() > x_valid.min():
            x_normed = (x_valid - x_valid.min()) / (x_valid.max() - x_valid.min())
        else:
            x_normed = np.ones_like(x_valid) * 0.5
        result = np.zeros_like(x)
        result[valid] = x_normed
        return result
    
    # Lower barrier = more reactive
    barrier_score = safe_norm(barriers, invert=True)
    
    # Lower action = more reactive
    action_score = safe_norm(actions, invert=True)
    
    # Higher transition density = where reaction happens
    trans_score = safe_norm(trans_density)
    
    # Higher coupling = better enzyme interaction
    coupling_score = safe_norm(couplings)
    
    # Lower topological charge = less protected = more reactive
    topo_score = safe_norm(topo_charge, invert=True)
    
    # Higher divergence = electron donating = more reactive
    div_score = safe_norm(divergence)
    
    # Lower Berry phase = topologically trivial = reactive
    berry_score = safe_norm(berry_phases, invert=True)
    
    # === COMBINE INTO FINAL SCORE ===
    scores = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        
        if atom.GetAtomicNum() != 6:
            scores[i] = -np.inf
            continue
        
        n_H = atom.GetTotalNumHs()
        if n_H == 0 and not atom.GetIsAromatic():
            scores[i] = -np.inf
            continue
        
        # Field theory score
        scores[i] = (
            0.20 * barrier_score[i] +      # Low barrier
            0.20 * action_score[i] +       # Low action (easy tunneling)
            0.15 * coupling_score[i] +     # Strong enzyme coupling
            0.15 * trans_score[i] +        # High transition density
            0.10 * topo_score[i] +         # Low topological protection
            0.10 * div_score[i] +          # Electron donating
            0.10 * berry_score[i]          # Trivial Berry phase
        )
        
        # Chemical adjustments
        if n_H > 0:
            scores[i] *= (1 + 0.12 * n_H)
        else:
            scores[i] *= 0.6
        
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() in [7, 8, 16]:
                scores[i] *= 1.35
                break
        
        if not atom.GetIsAromatic() and n_H > 0:
            for nbr in atom.GetNeighbors():
                if nbr.GetIsAromatic():
                    scores[i] *= 1.2
                    break
    
    # Rank
    valid = [i for i in range(n) if scores[i] > -np.inf]
    if not valid:
        return None
    
    ranked = sorted(valid, key=lambda x: -scores[x])
    
    return FieldTheoryPrediction(
        smiles=smiles,
        scores=scores,
        top1=ranked[0],
        top3=ranked[:3],
        components={
            'barrier': barrier_score,
            'action': action_score,
            'coupling': coupling_score,
            'transition_density': trans_score,
            'topological': topo_score,
            'divergence': div_score,
            'berry_phase': berry_score,
            'curvature': curvature,
            'torsion': torsion,
        }
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data_path: str) -> Dict:
    """Evaluate field theory model."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get('drugs', data)
    
    top1 = top3 = total = 0
    by_source = {}
    
    print(f"\nEvaluating FIELD THEORY MODEL on {len(drugs)} molecules...")
    print("-" * 60)
    
    for i, entry in enumerate(drugs):
        smiles = entry.get('smiles', '')
        sites = entry.get('site_atoms', [])
        source = entry.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        pred = predict_field_theory(smiles)
        if pred is None:
            continue
        
        if source not in by_source:
            by_source[source] = {'t1': 0, 't3': 0, 'n': 0}
        by_source[source]['n'] += 1
        
        if pred.top1 in sites:
            top1 += 1
            by_source[source]['t1'] += 1
        if any(p in sites for p in pred.top3):
            top3 += 1
            by_source[source]['t3'] += 1
        total += 1
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(drugs)}: Top-1={top1/total*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("FIELD THEORY MODEL - RESULTS")
    print("=" * 60)
    print(f"\nOVERALL: Top-1={top1/total*100:.1f}%, Top-3={top3/total*100:.1f}%")
    
    print("\nBY SOURCE:")
    for src, s in sorted(by_source.items(), key=lambda x: -x[1]['n']):
        if s['n'] >= 5:
            print(f"  {src:20s}: Top-1={s['t1']/s['n']*100:5.1f}%, Top-3={s['t3']/s['n']*100:5.1f}% (n={s['n']})")
    
    return {'top1': top1/total, 'top3': top3/total, 'by_source': by_source}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])
    else:
        print("Usage: python field_theory.py <data.json>")
