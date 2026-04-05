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
from enzyme_software.liquid_nn_v2.features.anomaly import compute_anomaly_features
from enzyme_software.liquid_nn_v2.features.atom_features import extract_atom_features
from enzyme_software.liquid_nn_v2.features.charge_update import update_local_charges_eem
from enzyme_software.liquid_nn_v2.features.chemistry_prior import compute_etn_prior_scores
from enzyme_software.liquid_nn_v2.features.group_detector import detect_functional_groups
from enzyme_software.liquid_nn_v2.features.local_field import compute_local_field_features, resolve_atom_coordinates
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
    candidate_mask: Optional[np.ndarray] = None
    candidate_train_mask: Optional[np.ndarray] = None
    cyp_label: Optional[int] = None
    manual_engine_atom_features: Optional[np.ndarray] = None
    manual_engine_mol_features: Optional[np.ndarray] = None
    manual_engine_atom_prior_logits: Optional[np.ndarray] = None
    manual_engine_cyp_prior_logits: Optional[np.ndarray] = None
    manual_engine_route_prior: Optional[np.ndarray] = None
    manual_engine_status: Optional[np.ndarray] = None
    xtb_atom_features: Optional[np.ndarray] = None
    xtb_atom_valid_mask: Optional[np.ndarray] = None
    xtb_mol_valid: Optional[np.ndarray] = None
    xtb_feature_status: Optional[str] = None
    xtb_status_flags: Optional[np.ndarray] = None
    atom_3d_features: Optional[np.ndarray] = None
    atom_3d_valid_mask: Optional[np.ndarray] = None
    atom_3d_source: Optional[np.ndarray] = None
    topology_atom_features: Optional[np.ndarray] = None
    topology_atom_valid_mask: Optional[np.ndarray] = None
    topology_mol_valid: Optional[np.ndarray] = None
    atom_coordinates: Optional[np.ndarray] = None
    local_chem_features: Optional[np.ndarray] = None
    local_charge_updated: Optional[np.ndarray] = None
    local_charge_delta: Optional[np.ndarray] = None
    local_etn_prior: Optional[np.ndarray] = None
    local_etn_features: Optional[np.ndarray] = None
    local_anomaly_features: Optional[np.ndarray] = None
    local_anomaly_score: Optional[np.ndarray] = None
    local_anomaly_score_normalized: Optional[np.ndarray] = None
    local_anomaly_flag: Optional[np.ndarray] = None
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
        invalid_site_atoms: List[int] = []
        for idx in site_atoms:
            idx_int = int(idx)
            if 0 <= idx_int < mol.GetNumAtoms():
                labels[idx_int, 0] = 1.0
            else:
                invalid_site_atoms.append(idx_int)
        if invalid_site_atoms:
            raise ValueError(
                f"Invalid site_atom indices for {smiles}: {invalid_site_atoms} "
                f"(num_atoms={mol.GetNumAtoms()})"
            )
        if len(site_atoms) > 0 and float(labels.sum()) == 0.0:
            raise ValueError(
                f"No valid site labels set for {smiles}; site_atoms={site_atoms}, num_atoms={mol.GetNumAtoms()}"
            )
    else:
        labels = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
        site_supervision_mask = np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)

    cyp_index = CYP_TO_INDEX.get(cyp_label) if cyp_label else None
    atom_coordinates = resolve_atom_coordinates(mol, structure_mol=structure_mol)
    atomic_numbers = np.asarray([int(atom.GetAtomicNum()) for atom in mol.GetAtoms()], dtype=np.int64)
    formal_charges = np.asarray([float(atom.GetFormalCharge()) for atom in mol.GetAtoms()], dtype=np.float32).reshape(-1, 1)
    field_payload = compute_local_field_features(atom_coordinates, atomic_numbers, formal_charges)
    updated_charges = update_local_charges_eem(atom_coordinates, atomic_numbers, formal_charges)
    charge_delta = updated_charges.astype(np.float32) - formal_charges.astype(np.float32)
    abs_charge_delta = np.abs(charge_delta).astype(np.float32)
    etn_payload = compute_etn_prior_scores(
        atom_coordinates,
        updated_charges,
        field_payload["field_score"],
        field_payload["access_proxy"],
        field_payload["crowding"],
        np.asarray(physics_features["bde_values"], dtype=np.float32),
        edge_index,
    )
    anomaly_payload = compute_anomaly_features(mol, num_atoms=mol.GetNumAtoms())
    etn_features = np.concatenate(
        [
            etn_payload["yield"],
            etn_payload["rank"],
            etn_payload["top_gap"],
            etn_payload["zscore"],
        ],
        axis=1,
    ).astype(np.float32)
    local_chem_features = np.concatenate(
        [
            field_payload["steric_score"],
            field_payload["electro_score"],
            field_payload["field_score"],
            field_payload["access_proxy"],
            field_payload["crowding"],
            charge_delta,
            abs_charge_delta,
            etn_features,
        ],
        axis=1,
    ).astype(np.float32)
    atom_3d_features = None
    atom_3d_source = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
    if structure_mol is not None:
        atom_3d_features = compute_atom_3d_features(mol, structure_mol)
        if atom_3d_features is not None:
            atom_3d_source = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    if atom_3d_features is None:
        atom_3d_features = compute_atom_3d_features_for_smiles(prep.canonical_smiles or smiles)
        if atom_3d_features is not None:
            atom_3d_source = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    atom_3d_valid_mask = np.ones((mol.GetNumAtoms(), 1), dtype=np.float32) if atom_3d_features is not None else np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32)
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
        candidate_mask=np.ones((mol.GetNumAtoms(), 1), dtype=np.float32),
        candidate_train_mask=np.ones((mol.GetNumAtoms(), 1), dtype=np.float32),
        cyp_label=cyp_index,
        atom_3d_features=atom_3d_features,
        atom_3d_valid_mask=atom_3d_valid_mask,
        atom_3d_source=atom_3d_source,
        manual_engine_status=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        topology_atom_valid_mask=np.zeros((mol.GetNumAtoms(), 1), dtype=np.float32),
        topology_mol_valid=np.asarray([[0.0]], dtype=np.float32),
        atom_coordinates=atom_coordinates.astype(np.float32) if atom_coordinates is not None else np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32),
        local_chem_features=local_chem_features,
        local_charge_updated=updated_charges.astype(np.float32),
        local_charge_delta=charge_delta.astype(np.float32),
        local_etn_prior=etn_payload["prior_score"].astype(np.float32),
        local_etn_features=etn_features.astype(np.float32),
        local_anomaly_features=anomaly_payload["features"].astype(np.float32),
        local_anomaly_score=anomaly_payload["score"].astype(np.float32),
        local_anomaly_score_normalized=anomaly_payload["score_normalized"].astype(np.float32),
        local_anomaly_flag=anomaly_payload["flag"].astype(np.float32),
        parsing_status=prep.status,
        repaired=prep.repaired,
        aggressive_repair=prep.aggressive_repair,
        num_atoms=mol.GetNumAtoms(),
    )
