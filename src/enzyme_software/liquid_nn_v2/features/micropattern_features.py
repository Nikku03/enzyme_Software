from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence

import numpy as np

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol


CHEMISTRY_PRIOR_PATTERN_DEFS = [
    # ── High-priority heteroatom sites ──────────────────────────────────────────
    ("s_oxidation",        "[S;X2;!$([S]=*)]",              0, 0.75),  # thioether S → sulfoxide (CYP3A4/2C19)
    ("thiophene_s",        "[s;r5]",                         0, 0.70),  # aromatic thiophene S → reactive S-oxide
    ("n_oxidation",        "[N;X3;H0;!$([N+]);!$([N]=*)]",  0, 0.72),  # tertiary amine N → N-oxide (CYP3A4)
    ("primary_aro_amine",  "[NH2;!R][c]",                   0, 0.60),  # aniline-type → N-hydroxylation
    ("ring_nitrogen_6",    "[N;r6;X3]",                      0, 0.70),  # piperidine/piperazine N (CYP substrate)
    # ── Classic C-H oxidation sites ─────────────────────────────────────────────
    ("benzylic",           "[CH2,CH3;!R][c]",                0, 0.90),
    ("allylic",            "[CH2,CH3][C]=[C]",               0, 0.80),
    ("alkene_epoxidation", "[C]=[C]",                        0, 0.65),  # alkene → epoxide (CYP3A4)
    ("alpha_to_oxygen",    "[CH2,CH3][O]",                   0, 0.78),
    ("alpha_to_nitrogen",  "[CH2,CH3][N]",                   0, 0.78),
    ("n_methyl",           "[CH3][N]",                       0, 0.82),
    ("o_methyl_aromatic",  "[CH3]O[c]",                      0, 0.95),
    ("carbonyl_alpha",     "[CH2,CH3][C]=O",                 0, 0.68),
    # ── Deactivated / low-reactivity sites ──────────────────────────────────────
    ("aromatic_ch",        "[cH]",                           0, 0.18),
    ("halogen_adjacent",   "[C][F,Cl,Br,I]",                 0, 0.30),
]
CHEMISTRY_PRIOR_BASIC_DIM = 9
CHEMISTRY_PRIOR_DIM = len(CHEMISTRY_PRIOR_PATTERN_DEFS) + CHEMISTRY_PRIOR_BASIC_DIM

if Chem is not None:
    _CHEMISTRY_PRIOR_PATTERNS = [
        (name, Chem.MolFromSmarts(smarts), focus_index, reactivity)
        for name, smarts, focus_index, reactivity in CHEMISTRY_PRIOR_PATTERN_DEFS
    ]
else:  # pragma: no cover
    _CHEMISTRY_PRIOR_PATTERNS = []


def _adjacency(edge_index: np.ndarray, num_atoms: int) -> List[List[int]]:
    adj = [[] for _ in range(num_atoms)]
    for src, dst in edge_index.T.tolist():
        if 0 <= src < num_atoms and 0 <= dst < num_atoms:
            adj[src].append(dst)
    return adj


def neighbors_within_radius(edge_index: np.ndarray, num_atoms: int, center: int, radius: int) -> Dict[int, List[int]]:
    adj = _adjacency(edge_index, num_atoms)
    visited = {center}
    frontier = deque([(center, 0)])
    shells: Dict[int, List[int]] = {0: [center]}
    while frontier:
        node, dist = frontier.popleft()
        if dist >= radius:
            continue
        next_dist = dist + 1
        for nbr in adj[node]:
            if nbr in visited:
                continue
            visited.add(nbr)
            shells.setdefault(next_dist, []).append(nbr)
            frontier.append((nbr, next_dist))
    for dist in range(radius + 1):
        shells.setdefault(dist, [])
    return shells


def ring_and_aromatic_arrays(smiles: str, num_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    if Chem is None:
        return np.zeros((num_atoms,), dtype=np.float32), np.zeros((num_atoms,), dtype=np.float32)
    prep = prepare_mol(smiles)
    if prep.mol is None:
        return np.zeros((num_atoms,), dtype=np.float32), np.zeros((num_atoms,), dtype=np.float32)
    mol = prep.mol
    ring = np.zeros((num_atoms,), dtype=np.float32)
    aromatic = np.zeros((num_atoms,), dtype=np.float32)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx >= num_atoms:
            continue
        ring[idx] = 1.0 if atom.IsInRing() else 0.0
        aromatic[idx] = 1.0 if atom.GetIsAromatic() else 0.0
    return ring, aromatic


def chemistry_prior_matrix(smiles: str, num_atoms: int) -> np.ndarray:
    features = np.zeros((num_atoms, CHEMISTRY_PRIOR_DIM), dtype=np.float32)
    if Chem is None:
        return features
    prep = prepare_mol(smiles)
    if prep.mol is None:
        return features
    mol = prep.mol
    pattern_width = len(CHEMISTRY_PRIOR_PATTERN_DEFS)
    reactivity = np.zeros((num_atoms,), dtype=np.float32)
    for column, (_, pattern, focus_index, base_reactivity) in enumerate(_CHEMISTRY_PRIOR_PATTERNS):
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            if focus_index >= len(match):
                continue
            atom_idx = int(match[focus_index])
            if atom_idx >= num_atoms:
                continue
            features[atom_idx, column] = 1.0
            reactivity[atom_idx] = max(reactivity[atom_idx], float(base_reactivity))
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if atom_idx >= num_atoms:
            continue
        hetero_neighbors = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() not in (1, 6))
        basic_offset = pattern_width
        features[atom_idx, basic_offset + 0] = 1.0 if atom.GetAtomicNum() == 6 else 0.0
        features[atom_idx, basic_offset + 1] = 1.0 if atom.GetIsAromatic() else 0.0
        features[atom_idx, basic_offset + 2] = 1.0 if atom.IsInRing() else 0.0
        features[atom_idx, basic_offset + 3] = 1.0 if atom.IsInRingSize(5) else 0.0
        features[atom_idx, basic_offset + 4] = 1.0 if atom.IsInRingSize(6) else 0.0
        features[atom_idx, basic_offset + 5] = 1.0 if atom.GetHybridization() == Chem.HybridizationType.SP3 else 0.0
        features[atom_idx, basic_offset + 6] = 1.0 if atom.GetHybridization() == Chem.HybridizationType.SP2 else 0.0
        features[atom_idx, basic_offset + 7] = float(min(4, atom.GetTotalNumHs())) / 4.0
        features[atom_idx, basic_offset + 8] = max(
            reactivity[atom_idx],
            float(min(4, hetero_neighbors)) / 4.0,
        )
    return features


def _mean_or_zeros(rows: np.ndarray, width: int) -> np.ndarray:
    if rows.size == 0:
        return np.zeros((width,), dtype=np.float32)
    return rows.mean(axis=0).astype(np.float32)


def build_candidate_local_descriptor(
    *,
    edge_index: np.ndarray,
    num_atoms: int,
    center_idx: int,
    radius: int,
    atom_embeddings: np.ndarray,
    manual_features: np.ndarray | None,
    xtb_features: np.ndarray | None,
    prior_features: np.ndarray | None,
    ring_flags: np.ndarray,
    aromatic_flags: np.ndarray,
    edge_attr: np.ndarray | None = None,
) -> np.ndarray:
    shells = neighbors_within_radius(edge_index, num_atoms, center_idx, radius)
    emb_dim = int(atom_embeddings.shape[-1])
    manual_dim = int(manual_features.shape[-1]) if manual_features is not None and manual_features.size else 0
    xtb_dim = int(xtb_features.shape[-1]) if xtb_features is not None and xtb_features.size else 0
    prior_dim = int(prior_features.shape[-1]) if prior_features is not None and prior_features.size else 0

    center_emb = atom_embeddings[center_idx]
    hop1 = np.asarray(shells.get(1, []), dtype=np.int64)
    hop2 = np.asarray(shells.get(2, []), dtype=np.int64)
    hop1_emb = _mean_or_zeros(atom_embeddings[hop1], emb_dim) if hop1.size else np.zeros((emb_dim,), dtype=np.float32)
    hop2_emb = _mean_or_zeros(atom_embeddings[hop2], emb_dim) if hop2.size else np.zeros((emb_dim,), dtype=np.float32)

    if manual_dim:
        center_manual = manual_features[center_idx]
        hop1_manual = _mean_or_zeros(manual_features[hop1], manual_dim) if hop1.size else np.zeros((manual_dim,), dtype=np.float32)
        hop2_manual = _mean_or_zeros(manual_features[hop2], manual_dim) if hop2.size else np.zeros((manual_dim,), dtype=np.float32)
    else:
        center_manual = hop1_manual = hop2_manual = np.zeros((0,), dtype=np.float32)

    if xtb_dim:
        center_xtb = xtb_features[center_idx]
        hop1_xtb = _mean_or_zeros(xtb_features[hop1], xtb_dim) if hop1.size else np.zeros((xtb_dim,), dtype=np.float32)
        hop2_xtb = _mean_or_zeros(xtb_features[hop2], xtb_dim) if hop2.size else np.zeros((xtb_dim,), dtype=np.float32)
    else:
        center_xtb = hop1_xtb = hop2_xtb = np.zeros((0,), dtype=np.float32)

    if prior_dim:
        center_prior = prior_features[center_idx]
        hop1_prior = _mean_or_zeros(prior_features[hop1], prior_dim) if hop1.size else np.zeros((prior_dim,), dtype=np.float32)
        hop2_prior = _mean_or_zeros(prior_features[hop2], prior_dim) if hop2.size else np.zeros((prior_dim,), dtype=np.float32)
    else:
        center_prior = hop1_prior = hop2_prior = np.zeros((0,), dtype=np.float32)

    scalars = np.asarray(
        [
            float(len(hop1)) / max(1.0, float(num_atoms)),
            float(len(hop2)) / max(1.0, float(num_atoms)),
            float(ring_flags[center_idx]),
            float(aromatic_flags[center_idx]),
            float(ring_flags[hop1].mean()) if hop1.size else 0.0,
            float(aromatic_flags[hop1].mean()) if hop1.size else 0.0,
            float(ring_flags[hop2].mean()) if hop2.size else 0.0,
            float(aromatic_flags[hop2].mean()) if hop2.size else 0.0,
        ],
        dtype=np.float32,
    )

    bond_counts = np.zeros((4,), dtype=np.float32)
    if edge_attr is not None and edge_attr.size:
        for edge_idx, (src, _) in enumerate(edge_index.T.tolist()):
            if src != center_idx:
                continue
            bond_counts += edge_attr[edge_idx, :4].astype(np.float32)
        bond_counts = np.clip(bond_counts / 4.0, 0.0, 1.0)

    return np.concatenate(
        [
            center_emb.astype(np.float32),
            hop1_emb,
            hop2_emb,
            center_manual,
            hop1_manual,
            hop2_manual,
            center_xtb,
            hop1_xtb,
            hop2_xtb,
            center_prior,
            hop1_prior,
            hop2_prior,
            bond_counts,
            scalars,
        ],
        axis=0,
    ).astype(np.float32)
