"""
Per-atom global topology features for large-molecule handling.

The base atom features (140-dim) encode local chemistry well but cannot encode
where an atom sits within a large molecule -- two atoms can look identical
locally while being in very different molecular contexts (scaffold core vs
dangling side chain).  These 5 features give the site arbiter global-position
context so it can correct the GNN's local-only predictions on large molecules.

Feature layout (TOPOLOGY_FEATURE_DIM = 5):
    [0] scaffold_member      1.0 if atom is in the Bemis-Murcko scaffold ring system
    [1] sidechain_member     1.0 if atom is in a dangling side chain (not scaffold)
    [2] closeness_centrality normalized graph closeness (1.0 = most central atom)
    [3] dist_to_carbonyl     1.0 = atom IS the carbonyl C; 0.0 = far away / no C=O
    [4] mol_size_context     min(1.0, n_atoms / 50.0) -- same for every atom in mol
"""
from __future__ import annotations

from typing import Optional

import numpy as np

TOPOLOGY_FEATURE_DIM = 5

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover
    Chem = None
    MurckoScaffold = None


def _bfs_distances(mol, start: int) -> np.ndarray:
    """BFS shortest-path bond distances from `start` to every atom."""
    n = mol.GetNumAtoms()
    INF = n * 2
    dist = np.full(n, INF, dtype=np.int32)
    dist[start] = 0
    queue = [start]
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        for nbr in mol.GetAtomWithIdx(curr).GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if dist[nbr_idx] == INF:
                dist[nbr_idx] = dist[curr] + 1
                queue.append(nbr_idx)
    return dist


def compute_atom_topology_features(smiles: str) -> Optional[np.ndarray]:
    """
    Return a (num_atoms, 5) float32 array of global topology features.

    Returns None if RDKit is unavailable or SMILES parsing fails.
    Safe to call with any SMILES — falls back to zeros on errors.
    """
    if Chem is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None

    n = mol.GetNumAtoms()
    if n == 0:
        return np.zeros((0, TOPOLOGY_FEATURE_DIM), dtype=np.float32)

    features = np.zeros((n, TOPOLOGY_FEATURE_DIM), dtype=np.float32)

    # ── [0] scaffold_member, [1] sidechain_member ────────────────────────────
    # Bemis-Murcko scaffold atoms = ring atoms + linker atoms between rings.
    # Side-chain atoms = everything else (not in scaffold, not a ring atom).
    ring_info = mol.GetRingInfo()
    ring_atom_set: set[int] = set()
    for ring in ring_info.AtomRings():
        ring_atom_set.update(int(idx) for idx in ring)

    scaffold_atom_set: set[int] = set(ring_atom_set)  # start with ring atoms
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is not None and scaffold.GetNumAtoms() > 0:
            match = mol.GetSubstructMatch(scaffold)
            scaffold_atom_set = set(int(idx) for idx in match)
    except Exception:
        pass  # fall back to ring_atom_set

    for atom_idx in range(n):
        if atom_idx in scaffold_atom_set:
            features[atom_idx, 0] = 1.0   # scaffold_member
        else:
            features[atom_idx, 1] = 1.0   # sidechain_member

    # ── [2] closeness_centrality (normalized) ────────────────────────────────
    # For each atom: closeness = (n-1) / sum_of_bond_distances_to_all_others
    # Normalize so the most central atom = 1.0.
    if n == 1:
        features[:, 2] = 1.0
    else:
        centrality = np.zeros(n, dtype=np.float32)
        for atom_idx in range(n):
            dist = _bfs_distances(mol, atom_idx)
            reachable = dist[dist < n * 2]
            total = float(reachable.sum()) - float(dist[atom_idx])  # exclude self
            if total > 0:
                centrality[atom_idx] = float(n - 1) / total
        max_c = centrality.max()
        if max_c > 0:
            features[:, 2] = centrality / max_c

    # ── [3] dist_to_carbonyl (inverted, normalized) ──────────────────────────
    # 1.0 = atom is a carbonyl carbon; decays with distance; 0.0 = no C=O.
    carbonyl_atoms: list[int] = []
    carbonyl_pat = Chem.MolFromSmarts("[CX3]=O") if Chem is not None else None
    if carbonyl_pat is not None:
        try:
            for match in mol.GetSubstructMatches(carbonyl_pat):
                carbonyl_atoms.append(int(match[0]))
        except Exception:
            pass

    if carbonyl_atoms:
        for atom_idx in range(n):
            dist = _bfs_distances(mol, atom_idx)
            min_d = min(int(dist[c]) for c in carbonyl_atoms if dist[c] < n * 2)
            # 1.0 at distance 0, decays linearly to 0.0 at distance >= n
            features[atom_idx, 3] = max(0.0, 1.0 - float(min_d) / max(1, n))

    # ── [4] mol_size_context ─────────────────────────────────────────────────
    # Same value for every atom in the molecule.  Lets the arbiter know it is
    # dealing with a large molecule where local features are less reliable.
    features[:, 4] = min(1.0, float(n) / 50.0)

    return features.astype(np.float32)
