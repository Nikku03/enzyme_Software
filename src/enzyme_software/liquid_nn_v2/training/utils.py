from __future__ import annotations

from typing import Dict, Iterable, List

import os
import numpy as np

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE, require_torch, torch
from enzyme_software.liquid_nn_v2.data.smarts_patterns import FUNCTIONAL_GROUP_SMARTS


def _tensor(data, *, dtype=None):
    require_torch()
    return torch.as_tensor(data, dtype=dtype) if dtype is not None else torch.as_tensor(data)


def _cast_tensor_precision(value, dtype):
    if dtype is None or not hasattr(value, "dtype"):
        return value
    if torch.is_floating_point(value):
        return value.to(dtype=dtype)
    return value


def collate_molecule_graphs(graphs: Iterable) -> Dict[str, object]:
    require_torch()
    graphs = list(graphs)
    x_parts: List[torch.Tensor] = []
    edge_parts: List[torch.Tensor] = []
    edge_attr_parts: List[torch.Tensor] = []
    tau_parts: List[torch.Tensor] = []
    batch_parts: List[torch.Tensor] = []
    site_parts: List[torch.Tensor] = []
    site_mask_parts: List[torch.Tensor] = []
    cyp_labels: List[int] = []
    cyp_supervision_mask: List[float] = []
    group_assignments = {}
    group_memberships: List[object] = []
    manual_atom_parts: List[object] = []
    manual_mol_parts: List[object] = []
    manual_atom_prior_parts: List[object] = []
    manual_cyp_prior_parts: List[object] = []
    manual_route_prior_parts: List[object] = []
    xtb_atom_parts: List[object] = []
    xtb_atom_valid_parts: List[object] = []
    xtb_mol_valid_parts: List[object] = []
    xtb_statuses: List[str] = []
    atom_3d_parts: List[object] = []
    parsing_statuses: List[str] = []
    canonical_smiles: List[str] = []
    repaired_flags: List[bool] = []
    aggressive_repair_flags: List[bool] = []
    physics_keys = ["bde_values", "bond_classes", "electronegativity", "is_aromatic", "functional_groups", "radical_stability", "nucleophilicity", "electrophilicity", "heteroatom_distance"]
    physics_parts = {key: [] for key in physics_keys}
    offset = 0
    for mol_idx, graph in enumerate(graphs):
        x_parts.append(_tensor(graph.x, dtype=torch.float32))
        tau_parts.append(_tensor(graph.tau_init, dtype=torch.float32))
        batch_parts.append(torch.full((graph.num_atoms,), mol_idx, dtype=torch.long))
        if graph.edge_index.size:
            edge_parts.append(_tensor(graph.edge_index, dtype=torch.long) + offset)
        if getattr(graph, "edge_attr", None) is not None and len(graph.edge_attr):
            edge_attr_parts.append(_tensor(graph.edge_attr, dtype=torch.float32))
        if graph.site_labels is not None:
            site_parts.append(_tensor(graph.site_labels, dtype=torch.float32))
        if getattr(graph, "site_supervision_mask", None) is not None:
            site_mask_parts.append(_tensor(graph.site_supervision_mask, dtype=torch.float32))
        cyp_labels.append(int(getattr(graph, "cyp_label", 0) or 0))
        cyp_supervision_mask.append(1.0 if bool(getattr(graph, "has_cyp_supervision", True)) else 0.0)
        for key in physics_keys:
            physics_parts[key].append(_tensor(graph.physics_features[key], dtype=torch.float32))
        group_memberships.append(_tensor(graph.group_membership, dtype=torch.float32) if getattr(graph, "group_membership", None) is not None else None)
        for _, group_name in enumerate(FUNCTIONAL_GROUP_SMARTS.keys()):
            atom_ids = graph.group_assignments.get((0, group_name), [])
            group_assignments[(mol_idx, group_name)] = [offset + int(idx) for idx in atom_ids]
        manual_atom_parts.append(_tensor(graph.manual_engine_atom_features, dtype=torch.float32) if getattr(graph, "manual_engine_atom_features", None) is not None else None)
        manual_mol_parts.append(_tensor(graph.manual_engine_mol_features, dtype=torch.float32) if getattr(graph, "manual_engine_mol_features", None) is not None else None)
        manual_atom_prior_parts.append(_tensor(graph.manual_engine_atom_prior_logits, dtype=torch.float32) if getattr(graph, "manual_engine_atom_prior_logits", None) is not None else None)
        manual_cyp_prior_parts.append(_tensor(graph.manual_engine_cyp_prior_logits, dtype=torch.float32) if getattr(graph, "manual_engine_cyp_prior_logits", None) is not None else None)
        manual_route_prior_parts.append(_tensor(graph.manual_engine_route_prior, dtype=torch.float32) if getattr(graph, "manual_engine_route_prior", None) is not None else None)
        xtb_atom_parts.append(_tensor(graph.xtb_atom_features, dtype=torch.float32) if getattr(graph, "xtb_atom_features", None) is not None else None)
        xtb_atom_valid_parts.append(_tensor(graph.xtb_atom_valid_mask, dtype=torch.float32) if getattr(graph, "xtb_atom_valid_mask", None) is not None else None)
        xtb_mol_valid_parts.append(_tensor(graph.xtb_mol_valid, dtype=torch.float32) if getattr(graph, "xtb_mol_valid", None) is not None else None)
        xtb_statuses.append(str(getattr(graph, "xtb_feature_status", "missing")))
        atom_3d_parts.append(_tensor(graph.atom_3d_features, dtype=torch.float32) if getattr(graph, "atom_3d_features", None) is not None else None)
        parsing_statuses.append(str(getattr(graph, "parsing_status", "unknown")))
        canonical_smiles.append(str(getattr(graph, "canonical_smiles", graph.smiles)))
        repaired_flags.append(bool(getattr(graph, "repaired", False)))
        aggressive_repair_flags.append(bool(getattr(graph, "aggressive_repair", False)))
        offset += graph.num_atoms
    num_molecules = len(graphs)
    max_atoms = max((int(graph.num_atoms) for graph in graphs), default=0)
    num_groups = len(FUNCTIONAL_GROUP_SMARTS)
    if any(membership is not None for membership in group_memberships):
        group_membership = torch.zeros((num_molecules, max_atoms, num_groups), dtype=torch.float32)
        for mol_idx, membership in enumerate(group_memberships):
            if membership is not None:
                group_membership[mol_idx, : membership.shape[0], : membership.shape[1]] = membership
    else:
        group_membership = torch.zeros((num_molecules, max_atoms, num_groups), dtype=torch.float32)
    batch = {
        "x": torch.cat(x_parts, dim=0),
        "edge_index": torch.cat(edge_parts, dim=1) if edge_parts else torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.cat(edge_attr_parts, dim=0) if edge_attr_parts else torch.zeros((0, 10), dtype=torch.float32),
        "tau_init": torch.cat(tau_parts, dim=0),
        "batch": torch.cat(batch_parts, dim=0),
        "physics_features": {key: torch.cat(parts, dim=0) for key, parts in physics_parts.items()},
        "group_assignments": group_assignments,
        "group_membership": group_membership,
        "parsing_status": parsing_statuses,
        "canonical_smiles": canonical_smiles,
        "repaired": repaired_flags,
        "aggressive_repair": aggressive_repair_flags,
        "xtb_feature_status": xtb_statuses,
    }
    if site_parts:
        batch["site_labels"] = torch.cat(site_parts, dim=0)
    if site_mask_parts:
        batch["site_supervision_mask"] = torch.cat(site_mask_parts, dim=0)
    if cyp_labels:
        batch["cyp_labels"] = torch.as_tensor(cyp_labels, dtype=torch.long)
        batch["cyp_supervision_mask"] = torch.as_tensor(cyp_supervision_mask, dtype=torch.float32)
    def _stack_optional(per_graph_values, row_counts, *, keep_rows="atom"):
        present = [value for value in per_graph_values if value is not None]
        if not present:
            return None
        width = present[0].shape[-1] if present[0].ndim > 1 else 1
        filled = []
        for idx, value in enumerate(per_graph_values):
            if value is None:
                rows = row_counts[idx]
                filled.append(torch.zeros((rows, width), dtype=torch.float32))
            else:
                filled.append(value if value.ndim > 1 else value.unsqueeze(-1))
        return torch.cat(filled, dim=0)

    atom_counts = [int(graph.num_atoms) for graph in graphs]
    mol_counts = [1 for _ in graphs]
    manual_atom = _stack_optional(manual_atom_parts, atom_counts)
    manual_mol = _stack_optional(manual_mol_parts, mol_counts)
    manual_atom_prior = _stack_optional(manual_atom_prior_parts, atom_counts)
    manual_cyp_prior = _stack_optional(manual_cyp_prior_parts, mol_counts)
    manual_route_prior = _stack_optional(manual_route_prior_parts, mol_counts)
    xtb_atom = _stack_optional(xtb_atom_parts, atom_counts)
    xtb_atom_valid = _stack_optional(xtb_atom_valid_parts, atom_counts)
    xtb_mol_valid = _stack_optional(xtb_mol_valid_parts, mol_counts)
    atom_3d = _stack_optional(atom_3d_parts, atom_counts)
    if manual_atom is not None:
        batch["manual_engine_atom_features"] = manual_atom
    if manual_mol is not None:
        batch["manual_engine_mol_features"] = manual_mol
    if manual_atom_prior is not None:
        batch["manual_engine_atom_prior_logits"] = manual_atom_prior
    if manual_cyp_prior is not None:
        batch["manual_engine_cyp_prior_logits"] = manual_cyp_prior
    if manual_route_prior is not None:
        batch["manual_engine_route_prior"] = manual_route_prior
    if xtb_atom is not None:
        batch["xtb_atom_features"] = xtb_atom
    if xtb_atom_valid is not None:
        batch["xtb_atom_valid_mask"] = xtb_atom_valid
    if xtb_mol_valid is not None:
        batch["xtb_mol_valid"] = xtb_mol_valid
    if atom_3d is not None:
        batch["atom_3d_features"] = atom_3d
    if os.environ.get("LNN_DEBUG_COLLATE", "").strip().lower() in {"1", "true", "yes", "on"} and "site_labels" in batch:
        print(f"DEBUG collate: site_labels sum = {batch['site_labels'].sum().item()}")
    return batch


def move_to_device(batch: Dict[str, object], device, *, dtype=None):
    if not TORCH_AVAILABLE:
        return batch
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                k: _cast_tensor_precision(v.to(device), dtype) if hasattr(v, "to") else v
                for k, v in value.items()
            }
        elif hasattr(value, "to"):
            moved[key] = _cast_tensor_precision(value.to(device), dtype)
        else:
            moved[key] = value
    return moved


def create_dummy_batch(
    num_molecules: int = 2,
    num_atoms: int = 20,
    atom_input_dim: int = 140,
    *,
    include_manual_engine: bool = False,
    include_3d: bool = False,
):
    require_torch()
    atoms_per_mol = max(1, num_atoms // num_molecules)
    total_atoms = atoms_per_mol * num_molecules
    x = torch.rand(total_atoms, atom_input_dim)
    edges = []
    edge_attr_rows = []
    offset = 0
    for mol_idx in range(num_molecules):
        for idx in range(atoms_per_mol - 1):
            edges.append([offset + idx, offset + idx + 1])
            edges.append([offset + idx + 1, offset + idx])
            edge_attr_rows.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
            edge_attr_rows.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        offset += atoms_per_mol
    edge_index = torch.as_tensor(edges, dtype=torch.long).T if edges else torch.zeros((2, 0), dtype=torch.long)
    edge_attr = torch.as_tensor(edge_attr_rows, dtype=torch.float32) if edge_attr_rows else torch.zeros((0, 10), dtype=torch.float32)
    batch = torch.repeat_interleave(torch.arange(num_molecules), atoms_per_mol)
    tau_init = torch.rand(total_atoms) * 1.4 + 0.1
    physics_features = {
        "bde_values": torch.rand(total_atoms) * 120.0 + 360.0,
        "bond_classes": torch.rand(total_atoms, 12),
        "electronegativity": torch.rand(total_atoms),
        "is_aromatic": torch.randint(0, 2, (total_atoms,), dtype=torch.float32),
        "functional_groups": torch.rand(total_atoms, 16),
        "radical_stability": torch.rand(total_atoms),
        "nucleophilicity": torch.rand(total_atoms),
        "electrophilicity": torch.rand(total_atoms),
        "heteroatom_distance": torch.rand(total_atoms),
    }
    group_assignments = {}
    group_membership = torch.zeros((num_molecules, atoms_per_mol, len(FUNCTIONAL_GROUP_SMARTS)), dtype=torch.float32)
    for mol_idx in range(num_molecules):
        start = mol_idx * atoms_per_mol
        group_assignments[(mol_idx, "aromatic_ring")] = [start]
        group_assignments[(mol_idx, "aliphatic_chain")] = [start + 1] if atoms_per_mol > 1 else [start]
        group_membership[mol_idx, 0, list(FUNCTIONAL_GROUP_SMARTS.keys()).index("aromatic_ring")] = 1.0
        if atoms_per_mol > 1:
            group_membership[mol_idx, 1, list(FUNCTIONAL_GROUP_SMARTS.keys()).index("aliphatic_chain")] = 1.0
        for group_name in FUNCTIONAL_GROUP_SMARTS.keys():
            group_assignments.setdefault((mol_idx, group_name), [])
    result = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch,
        "tau_init": tau_init,
        "physics_features": physics_features,
        "group_assignments": group_assignments,
        "group_membership": group_membership,
        "site_labels": torch.randint(0, 2, (total_atoms, 1), dtype=torch.float32),
        "cyp_labels": torch.randint(0, 5, (num_molecules,), dtype=torch.long),
    }
    if include_manual_engine:
        result["manual_engine_atom_features"] = torch.rand(total_atoms, 8)
        result["manual_engine_mol_features"] = torch.rand(num_molecules, 8)
        result["manual_engine_atom_prior_logits"] = torch.rand(total_atoms, 1)
        result["manual_engine_cyp_prior_logits"] = torch.rand(num_molecules, 5)
    if include_3d:
        result["atom_3d_features"] = torch.rand(total_atoms, 8)
    return result
