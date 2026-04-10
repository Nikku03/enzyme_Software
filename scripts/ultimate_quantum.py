#!/usr/bin/env python3
"""
ULTIMATE QUANTUM FIELD PREDICTOR
=================================

Combining all insights from our exploration:

1. FLEXIBILITY (0.15): Atoms not locked in local vibrations can be perturbed
2. ELECTRON FLOW (0.30): Sites that can DONATE electrons to electrophilic enzyme  
3. TRANSITION BARRIER (0.25): Low barrier = easy reaction
4. ENZYME COUPLING (0.20): How strongly the enzyme field couples to this site
5. TOPOLOGICAL CHARGE (0.10): Low charge = not protected = reactive

The key insight: it's not just about WHERE electrons ARE,
but about WHERE electrons can FLOW TO when the enzyme arrives.

The enzyme is an electron sink. Reactive sites are electron sources
with low barriers and good coupling.
"""

import json
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit required")


# =============================================================================
# QUANTUM FIELD COMPUTATIONS
# =============================================================================

class MolecularQuantumField:
    """Complete quantum field analysis of a molecule."""
    
    def __init__(self, mol):
        self.mol = mol
        self.n = mol.GetNumAtoms()
        self.H, self.E, self.psi = self._build_and_solve()
        
    def _build_and_solve(self):
        """Build Hamiltonian and solve."""
        n = self.n
        
        # Adjacency
        A = np.zeros((n, n))
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            w = bond.GetBondTypeAsDouble()
            if bond.GetIsAromatic():
                w = 1.5
            A[i, j] = A[j, i] = w
        
        # Laplacian + potential
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        chi_map = {1: 2.2, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 
                   16: 2.58, 17: 3.16, 35: 2.96}
        V = np.array([chi_map.get(self.mol.GetAtomWithIdx(i).GetAtomicNum(), 2.5) - 2.55 
                      for i in range(n)])
        
        H = L + np.diag(V)
        E, psi = np.linalg.eigh(H)
        
        return H, E, psi
    
    def flexibility(self) -> np.ndarray:
        """Inverse participation in high-frequency modes."""
        rigidity = np.zeros(self.n)
        for k in range(max(1, self.n-3), self.n):
            rigidity += self.psi[:, k]**2
        flex = 1.0 / (rigidity + 0.1)
        return self._normalize(flex)
    
    def electron_flow(self) -> np.ndarray:
        """Divergence of ground state = electron source/sink."""
        A = np.zeros((self.n, self.n))
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            A[i, j] = A[j, i] = 1
        
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        psi_0 = self.psi[:, 0]
        divergence = L @ psi_0
        
        # Boost from electron-rich neighbors
        for i in range(self.n):
            atom = self.mol.GetAtomWithIdx(i)
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() in [7, 8, 16]:
                    divergence[i] += 0.3 * abs(psi_0[nbr.GetIdx()])
        
        return self._normalize(divergence)
    
    def transition_barrier(self, site: int) -> float:
        """Energy barrier for reaction at this site."""
        # Perturbed Hamiltonian
        H_pert = self.H.copy()
        
        # Enzyme at site
        H_pert[site, site] -= 0.5
        
        # Weaken adjacent bonds
        atom = self.mol.GetAtomWithIdx(site)
        for nbr in atom.GetNeighbors():
            j = nbr.GetIdx()
            H_pert[site, j] *= 0.8
            H_pert[j, site] *= 0.8
        
        # Solve
        E_pert, psi_pert = np.linalg.eigh(H_pert)
        
        # Barrier = change in ground state overlap
        overlap = abs(np.dot(self.psi[:, 0], psi_pert[:, 0]))
        barrier = 1 - overlap
        
        return barrier
    
    def enzyme_coupling(self, site: int) -> float:
        """Coupling strength to enzyme field."""
        # BFS distances
        distances = np.full(self.n, np.inf)
        distances[site] = 0
        queue = [site]
        visited = {site}
        
        while queue:
            curr = queue.pop(0)
            atom = self.mol.GetAtomWithIdx(curr)
            for nbr in atom.GetNeighbors():
                j = nbr.GetIdx()
                if j not in visited:
                    distances[j] = distances[curr] + 1
                    visited.add(j)
                    queue.append(j)
        
        # Coupling from overlap with enzyme field
        enzyme_field = np.exp(-distances / 1.5)
        coupling = np.dot(enzyme_field, self.psi[:, 0]**2)
        
        return coupling
    
    def topological_charge(self) -> np.ndarray:
        """Topological protection at each site."""
        charge = np.zeros(self.n)
        psi_0 = self.psi[:, 0]
        
        for i in range(self.n):
            atom = self.mol.GetAtomWithIdx(i)
            neighbors = [nb.GetIdx() for nb in atom.GetNeighbors()]
            
            if len(neighbors) >= 2:
                # Phase winding
                phases = [np.sign(psi_0[j]) for j in neighbors]
                changes = sum(1 for k in range(len(phases)) 
                             if phases[k] != phases[(k+1) % len(phases)])
                charge[i] = changes / len(neighbors)
        
        return charge
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        if x.max() > x.min():
            return (x - x.min()) / (x.max() - x.min())
        return np.ones_like(x) * 0.5


# =============================================================================
# MAIN PREDICTOR
# =============================================================================

@dataclass
class UltimatePrediction:
    """Prediction result."""
    smiles: str
    scores: np.ndarray
    top1: int
    top3: List[int]
    components: Dict[str, np.ndarray]


# Optimal weights from grid search
WEIGHTS = {
    'flexibility': 0.15,
    'electron_flow': 0.30,
    'barrier': 0.25,
    'coupling': 0.20,
    'topological': 0.10,
}


def predict_ultimate(smiles: str) -> Optional[UltimatePrediction]:
    """Predict SoM using ultimate quantum model."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
        return None
    
    n = mol.GetNumAtoms()
    
    # Quantum field analysis
    field = MolecularQuantumField(mol)
    
    # Compute all components
    flex = field.flexibility()
    flow = field.electron_flow()
    topo = field.topological_charge()
    
    # Per-site quantities
    barriers = np.zeros(n)
    couplings = np.zeros(n)
    
    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != 6:
            barriers[i] = np.inf
            continue
        barriers[i] = field.transition_barrier(i)
        couplings[i] = field.enzyme_coupling(i)
    
    # Normalize
    def safe_norm(x, invert=False):
        valid = np.isfinite(x)
        if not valid.any():
            return np.zeros_like(x)
        x_v = x.copy()
        x_v[~valid] = np.nan
        if invert:
            x_v = -x_v
        min_v, max_v = np.nanmin(x_v), np.nanmax(x_v)
        if max_v > min_v:
            x_v = (x_v - min_v) / (max_v - min_v)
        else:
            x_v = np.ones_like(x_v) * 0.5
        x_v[~valid] = 0
        return x_v
    
    barrier_n = safe_norm(barriers, invert=True)  # Lower = better
    coupling_n = safe_norm(couplings)
    topo_n = safe_norm(topo, invert=True)  # Lower = more reactive
    
    # Combined score
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
        
        # Weighted combination
        scores[i] = (
            WEIGHTS['flexibility'] * flex[i] +
            WEIGHTS['electron_flow'] * flow[i] +
            WEIGHTS['barrier'] * barrier_n[i] +
            WEIGHTS['coupling'] * coupling_n[i] +
            WEIGHTS['topological'] * topo_n[i]
        )
        
        # Chemistry
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
    
    return UltimatePrediction(
        smiles=smiles,
        scores=scores,
        top1=ranked[0],
        top3=ranked[:3],
        components={
            'flexibility': flex,
            'electron_flow': flow,
            'barrier': barrier_n,
            'coupling': coupling_n,
            'topological': topo_n,
        }
    )


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(data_path: str) -> Dict:
    """Evaluate on dataset."""
    with open(data_path) as f:
        data = json.load(f)
    
    drugs = data.get('drugs', data)
    
    top1 = top3 = total = 0
    by_source = {}
    
    print(f"\nEvaluating ULTIMATE QUANTUM MODEL on {len(drugs)} molecules...")
    print(f"Weights: {WEIGHTS}")
    print("-" * 60)
    
    for i, entry in enumerate(drugs):
        smiles = entry.get('smiles', '')
        sites = entry.get('site_atoms', [])
        source = entry.get('source', 'unknown')
        
        if not smiles or not sites:
            continue
        
        pred = predict_ultimate(smiles)
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
        
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(drugs)}: Top-1={top1/total*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("ULTIMATE QUANTUM MODEL - RESULTS")
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
        print("Usage: python ultimate_quantum.py <data.json>")
