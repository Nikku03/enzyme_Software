from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import warnings

import numpy as np

try:
    from rdkit import Chem
except Exception:  # pragma: no cover - optional dependency
    Chem = None

from enzyme_software.liquid_nn_v2.data.bde_table import bde_to_tau_init
from enzyme_software.liquid_nn_v2.data.drug_database import CYP_CLASSES
from enzyme_software.liquid_nn_v2.features.atom_features import extract_atom_features
from enzyme_software.liquid_nn_v2.features.group_detector import detect_functional_groups
from enzyme_software.liquid_nn_v2.features.physics_features import compute_molecule_physics_features
from enzyme_software.liquid_nn_v2.features.steric_features import compute_atom_3d_features, compute_atom_3d_features_for_smiles
from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol
from enzyme_software.liquid_nn_v2.utils.mol_provenance import mol_provenance_context


@dataclass
class MoleculeGraph:
    smiles: str
    canonical_smiles: Optional[str]
    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    tau_init: np.ndarray
    physics_features: Dict[str, object]
    group_assignments: Dict[tuple, List[int]]
    group_membership: np.ndarray
    batch: np.ndarray
    site_labels: Optional[np.ndarray] = None
    site_supervision_mask: Optional[np.ndarray] = None
    cyp_label: Optional[int] = None
    manual_engine_atom_features: Optional[np.ndarray] = None
    manual_engine_mol_features: Optional[np.ndarray] = None
    manual_engine_atom_prior_logits: Optional[np.ndarray] = None
    manual_engine_cyp_prior_logits: Optional[np.ndarray] = None
    manual_engine_route_prior: Optional[np.ndarray] = None
    atom_3d_features: Optional[np.ndarray] = None
    parsing_status: str = "ok"
    repaired: bool = False
    aggressive_repair: bool = False
    num_atoms: int = 0


CYP_TO_INDEX = {name: idx for idx, name in enumerate(CYP_CLASSES)}


def _bond_to_edge_features(bond) -> List[float]:
    bond_type = bond.GetBondType()
    stereo = str(bond.GetStereo())
    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond.GetIsAromatic()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond_type == Chem.BondType.SINGLE and (not bond.IsInRing())),
        float(stereo == "STEREONONE"),
        float(stereo == "STEREOE"),
        float(stereo not in {"STEREONONE", "STEREOE"}),
    ]


def smiles_to_graph(
    smiles: str,
    *,
    cyp_label: Optional[str] = None,
    site_atoms: Optional[List[int]] = None,
    structure_mol=None,
    allow_partial_sanitize: bool = True,
    allow_aggressive_repair: bool = False,
) -> MoleculeGraph:
    if Chem is None:
        raise RuntimeError("RDKit is required for smiles_to_graph")
    with mol_provenance_context(module_triggered="graph builder", source_category="graph builder", parsed_smiles=smiles):
        prep = prepare_mol(
            smiles,
            allow_partial_sanitize=allow_partial_sanitize,
            allow_aggressive_repair=allow_aggressive_repair,
        )
    if prep.mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}; status={prep.status}; error={prep.error}")
    mol = prep.mol

    group_assignments = detect_functional_groups(mol)
    x, meta = extract_atom_features(mol, group_assignments)
    physics_features = compute_molecule_physics_features(mol, group_assignments)
    tau_init = np.asarray([bde_to_tau_init(v) for v in physics_features["bde_values"]], dtype=np.float32)

    edges: List[List[int]] = []
    edge_attr_rows: List[List[float]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        edges.append([begin, end])
        edges.append([end, begin])
        features = _bond_to_edge_features(bond)
        edge_attr_rows.append(features)
        edge_attr_rows.append(features)
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    edge_attr = np.asarray(edge_attr_rows, dtype=np.float32) if edge_attr_rows else np.zeros((0, 10), dtype=np.float32)

    mapped_groups = {(0, group_name): atoms for group_name, atoms in group_assignments.items()}
    group_membership = np.asarray(physics_features["functional_groups"], dtype=np.float32)
    labels = None
    site_supervision_mask = None
    if site_atoms is not None:
        labels = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
        site_supervision_mask = np.ones((mol.GetNumAtoms(), 1), dtype=np.float32)
        for idx in site_atoms:
            idx_int = int(idx)
            if 0 <= idx_int < mol.GetNumAtoms():
                labels[idx_int, 0] = 1.0
            else:
                warnings.warn(
                    f"site_atom index {idx_int} out of range for {smiles} (num_atoms={mol.GetNumAtoms()})",
                    RuntimeWarning,
                )
        if len(site_atoms) > 0 and float(labels.sum()) == 0.0:
            warnings.warn(
                f"No valid site labels set for {smiles}; site_atoms={site_atoms}, num_atoms={mol.GetNumAtoms()}",
                RuntimeWarning,
            )
    else:
        labels = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
        site_supervision_mask = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)

    cyp_index = CYP_TO_INDEX.get(cyp_label) if cyp_label else None
    atom_3d_features = compute_atom_3d_features(mol, structure_mol) if structure_mol is not None else None
    if atom_3d_features is None:
        atom_3d_features = compute_atom_3d_features_for_smiles(prep.canonical_smiles or smiles)
    return MoleculeGraph(
        smiles=smiles,
        canonical_smiles=prep.canonical_smiles,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        tau_init=tau_init,
        physics_features=physics_features,
        group_assignments=mapped_groups,
        group_membership=group_membership,
        batch=np.zeros(mol.GetNumAtoms(), dtype=np.int64),
        site_labels=labels,
        site_supervision_mask=site_supervision_mask,
        cyp_label=cyp_index,
        atom_3d_features=atom_3d_features,
        parsing_status=prep.status,
        repaired=prep.repaired,
        aggressive_repair=prep.aggressive_repair,
        num_atoms=mol.GetNumAtoms(),
    )
