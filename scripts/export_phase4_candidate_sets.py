import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
from enzyme_software.liquid_nn_v2._compat import torch
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.dataset import create_full_xtb_dataloaders_from_drugs, split_drugs
from enzyme_software.liquid_nn_v2.features.micropattern_features import CHEMISTRY_PRIOR_PATTERN_DEFS, chemistry_prior_matrix
from enzyme_software.liquid_nn_v2.training.utils import move_to_device
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.model_utils import load_full_xtb_warm_start

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


def _primary_cyp(row: Dict[str, object]) -> str:
    return str(row.get("cyp") or row.get("primary_cyp") or "").strip()


def _load_drugs(dataset_path: Path, *, target_cyp: str = "", site_labeled_only: bool = False) -> List[Dict[str, object]]:
    payload = json.loads(dataset_path.read_text())
    drugs = list(payload.get("drugs", payload))
    if target_cyp:
        drugs = [row for row in drugs if _primary_cyp(row) == target_cyp]
    if site_labeled_only:
        drugs = [row for row in drugs if bool(row.get("site_atoms") or row.get("site_atom_indices") or row.get("som") or row.get("metabolism_sites"))]
    return drugs


def _load_proposer(checkpoint_path: Path, device) -> HybridLNNModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_config_dict = dict(((payload.get("config") or {}).get("base_model") or {}))
    base_config = ModelConfig(**base_config_dict)
    base_config.use_topk_reranker = False
    base_model = LiquidMetabolismNetV2(base_config)
    model = HybridLNNModel(base_model)
    load_full_xtb_warm_start(
        model,
        checkpoint_path,
        device=device,
        new_manual_atom_dim=int(base_config.manual_atom_feature_dim),
        new_atom_input_dim=int(base_config.atom_input_dim),
    )
    model.to(device)
    model.eval()
    return model


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _local_edge_index(global_edge_index, start: int, end: int) -> np.ndarray:
    edge_arr = _to_numpy(global_edge_index)
    if edge_arr is None or edge_arr.size == 0:
        return np.zeros((2, 0), dtype=np.int64)
    edge_arr = np.asarray(edge_arr, dtype=np.int64)
    mask = (
        (edge_arr[0] >= int(start))
        & (edge_arr[0] < int(end))
        & (edge_arr[1] >= int(start))
        & (edge_arr[1] < int(end))
    )
    local = edge_arr[:, mask].copy()
    if local.size:
        local -= int(start)
    return local


def _adjacency(edge_index: np.ndarray, num_atoms: int) -> list[list[int]]:
    adj = [[] for _ in range(int(num_atoms))]
    if edge_index.size == 0:
        return adj
    for src, dst in edge_index.T.tolist():
        if 0 <= int(src) < int(num_atoms) and 0 <= int(dst) < int(num_atoms):
            adj[int(src)].append(int(dst))
    return adj


def _shortest_path_matrix(edge_index: np.ndarray, num_atoms: int) -> np.ndarray:
    num_atoms = int(num_atoms)
    dist = np.full((num_atoms, num_atoms), fill_value=99.0, dtype=np.float32)
    if num_atoms <= 0:
        return dist
    adj = _adjacency(edge_index, num_atoms)
    for src in range(num_atoms):
        dist[src, src] = 0.0
        queue = deque([src])
        while queue:
            node = queue.popleft()
            next_dist = dist[src, node] + 1.0
            for nbr in adj[node]:
                if next_dist < dist[src, nbr]:
                    dist[src, nbr] = next_dist
                    queue.append(nbr)
    return dist


def _ring_system_ids(smiles: str, num_atoms: int) -> np.ndarray:
    labels = np.full((int(num_atoms),), fill_value=-1, dtype=np.int64)
    if Chem is None:
        return labels
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None or int(mol.GetNumAtoms()) != int(num_atoms):
        return labels
    rings = [set(int(v) for v in ring) for ring in mol.GetRingInfo().AtomRings()]
    if not rings:
        return labels
    parent = list(range(len(rings)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if rings[i].intersection(rings[j]):
                union(i, j)

    root_to_id = {}
    next_id = 0
    for ring_idx, atoms in enumerate(rings):
        root = find(ring_idx)
        if root not in root_to_id:
            root_to_id[root] = next_id
            next_id += 1
        system_id = root_to_id[root]
        for atom_idx in atoms:
            labels[int(atom_idx)] = int(system_id)
    return labels


def _chem_family_ids(smiles: str, num_atoms: int) -> np.ndarray:
    labels = np.zeros((int(num_atoms),), dtype=np.int64)
    if int(num_atoms) <= 0:
        return labels
    prior = chemistry_prior_matrix(smiles, int(num_atoms))
    if prior.size == 0:
        return labels
    pattern_names = [name for name, *_rest in CHEMISTRY_PRIOR_PATTERN_DEFS]
    family_defs = [
        ("benzylic", {"benzylic"}),
        ("allylic", {"allylic"}),
        ("alpha_hetero", {"alpha_to_oxygen", "alpha_to_nitrogen", "n_methyl", "o_methyl_aromatic", "carbonyl_alpha"}),
        ("hetero_oxidation", {"s_oxidation", "thiophene_s", "n_oxidation", "primary_aro_amine", "ring_nitrogen_6"}),
    ]
    for family_idx, (_name, members) in enumerate(family_defs, start=1):
        cols = [idx for idx, name in enumerate(pattern_names) if name in members]
        if not cols:
            continue
        active = np.asarray(prior[:, cols].sum(axis=1) > 0.0, dtype=bool)
        labels[(labels == 0) & active] = int(family_idx)
    return labels


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(arr[finite].min())
    hi = float(arr[finite].max())
    if hi - lo < 1.0e-6:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - lo) / (hi - lo)
    out[~finite] = 0.0
    return out.astype(np.float32)


def _build_rival_payload(
    *,
    smiles: str,
    num_atoms: int,
    candidate_local: np.ndarray,
    candidate_logits: np.ndarray,
    local_edge_index: np.ndarray,
    atom_coordinates: np.ndarray | None,
    topology_atom_features: np.ndarray | None,
    access_values: np.ndarray | None,
    barrier_values: np.ndarray | None,
) -> Dict[str, np.ndarray]:
    num_atoms = int(num_atoms)
    cand = np.asarray(candidate_local, dtype=np.int64).reshape(-1)
    k = int(cand.shape[0])
    if k <= 0:
        return {
            "candidate_local_rival_mask": np.zeros((0, 0), dtype=np.float32),
            "candidate_graph_distance": np.zeros((0, 0), dtype=np.float32),
            "candidate_3d_distance": np.zeros((0, 0), dtype=np.float32),
            "candidate_same_ring_system": np.zeros((0, 0), dtype=np.float32),
            "candidate_same_topology_role": np.zeros((0, 0), dtype=np.float32),
            "candidate_same_chem_family": np.zeros((0, 0), dtype=np.float32),
            "candidate_branch_bulk": np.zeros((0,), dtype=np.float32),
            "candidate_exposed_span": np.zeros((0,), dtype=np.float32),
            "candidate_anti_score": np.zeros((0,), dtype=np.float32),
        }
    graph_dist_full = _shortest_path_matrix(local_edge_index, num_atoms)
    cand_graph = graph_dist_full[np.ix_(cand, cand)].astype(np.float32)

    if atom_coordinates is not None and int(np.asarray(atom_coordinates).size):
        coords = np.asarray(atom_coordinates, dtype=np.float32).reshape(num_atoms, 3)
        cand_coords = coords[cand]
        cand_3d = np.linalg.norm(cand_coords[:, None, :] - cand_coords[None, :, :], axis=-1).astype(np.float32)
        centroid = coords.mean(axis=0, keepdims=True)
        radial = np.linalg.norm(coords - centroid, axis=-1).astype(np.float32)
        exposed_span = _normalize(radial)[cand]
        neighbor_density = np.asarray([(cand_3d[row] < 2.5).sum() - 1 for row in range(k)], dtype=np.float32)
        crowding = _normalize(neighbor_density)
    else:
        cand_3d = np.full((k, k), fill_value=99.0, dtype=np.float32)
        exposed_span = np.zeros((k,), dtype=np.float32)
        crowding = np.zeros((k,), dtype=np.float32)

    ring_ids = _ring_system_ids(smiles, num_atoms)
    same_ring = ((ring_ids[cand][:, None] == ring_ids[cand][None, :]) & (ring_ids[cand][:, None] >= 0)).astype(np.float32)

    chem_family = _chem_family_ids(smiles, num_atoms)
    same_family = ((chem_family[cand][:, None] == chem_family[cand][None, :]) & (chem_family[cand][:, None] > 0)).astype(np.float32)

    topo = np.asarray(topology_atom_features, dtype=np.float32).reshape(num_atoms, -1) if topology_atom_features is not None and np.asarray(topology_atom_features).size else None
    if topo is not None and topo.shape[1] >= 3:
        topo_role = np.where(topo[:, 0] >= topo[:, 1], 1, 2)
        same_role = (topo_role[cand][:, None] == topo_role[cand][None, :]).astype(np.float32)
        closeness = topo[:, 2].astype(np.float32)
        one_hop_counts = np.asarray([(graph_dist_full[idx] == 1).sum() for idx in range(num_atoms)], dtype=np.float32)
        two_hop_counts = np.asarray([(graph_dist_full[idx] <= 2).sum() - 1 for idx in range(num_atoms)], dtype=np.float32)
        branch_bulk = _normalize(0.45 * one_hop_counts + 0.55 * two_hop_counts + 0.75 * topo[:, 1] + 0.40 * (1.0 - closeness))[cand]
    else:
        same_role = np.zeros((k, k), dtype=np.float32)
        one_hop_counts = np.asarray([(graph_dist_full[idx] == 1).sum() for idx in range(num_atoms)], dtype=np.float32)
        two_hop_counts = np.asarray([(graph_dist_full[idx] <= 2).sum() - 1 for idx in range(num_atoms)], dtype=np.float32)
        branch_bulk = _normalize(0.5 * one_hop_counts + 0.5 * two_hop_counts)[cand]

    access_block = 1.0 - _normalize(np.asarray(access_values, dtype=np.float32).reshape(-1)) if access_values is not None and np.asarray(access_values).size else np.zeros((num_atoms,), dtype=np.float32)
    barrier_norm = _normalize(np.asarray(barrier_values, dtype=np.float32).reshape(-1)) if barrier_values is not None and np.asarray(barrier_values).size else np.zeros((num_atoms,), dtype=np.float32)
    anti_components = np.stack(
        [
            access_block[cand],
            barrier_norm[cand],
            crowding,
            1.0 - exposed_span,
        ],
        axis=1,
    ).astype(np.float32)
    anti_score = anti_components.max(axis=1).astype(np.float32)

    initial_rival = (
        (cand_graph <= 4.0)
        | (cand_3d <= 5.0)
        | (same_ring > 0.5)
        | (same_family > 0.5)
        | ((same_role > 0.5) & (cand_graph <= 6.0))
    )
    np.fill_diagonal(initial_rival, False)

    rival_mask = initial_rival.astype(np.float32)
    for row in range(k):
        current = int(rival_mask[row].sum())
        target = min(max(3, 1), max(0, k - 1))
        if current >= target:
            continue
        ordering = sorted(
            [col for col in range(k) if col != row],
            key=lambda col: (
                float(cand_graph[row, col]),
                float(cand_3d[row, col]),
                abs(float(candidate_logits[row] - candidate_logits[col])),
            ),
        )
        for col in ordering[:target]:
            rival_mask[row, col] = 1.0
    return {
        "candidate_local_rival_mask": rival_mask.astype(np.float32),
        "candidate_graph_distance": cand_graph.astype(np.float32),
        "candidate_3d_distance": cand_3d.astype(np.float32),
        "candidate_same_ring_system": same_ring.astype(np.float32),
        "candidate_same_topology_role": same_role.astype(np.float32),
        "candidate_same_chem_family": same_family.astype(np.float32),
        "candidate_branch_bulk": branch_bulk.astype(np.float32),
        "candidate_exposed_span": exposed_span.astype(np.float32),
        "candidate_anti_score": anti_score.astype(np.float32),
    }


def _topk_sample_rows(batch, outputs, *, top_k: int) -> List[Dict[str, object]]:
    batch_index = batch["batch"]
    site_logits = outputs["site_logits"].view(-1)
    atom_features = outputs["atom_features"]
    local_chem = batch.get("local_chem_features")
    anomaly = batch.get("local_anomaly_score_normalized")
    phase2 = outputs.get("phase2_context_outputs") or {}
    event_strain = phase2.get("event_strain")
    access_score = phase2.get("access_score")
    barrier_score = phase2.get("barrier_score")
    bde_values = (batch.get("physics_features") or {}).get("bde_values")
    edge_index = batch.get("edge_index")
    atom_coordinates = batch.get("atom_coordinates")
    topology_atom_features = batch.get("topology_atom_features")
    rows = []
    offset = 0
    graph_num_atoms = [int(v) for v in list(batch.get("graph_num_atoms", []))]
    graph_metadata = list(batch.get("graph_metadata", []))
    canonical_smiles = list(batch.get("canonical_smiles", []))
    site_labels = batch["site_labels"].view(-1)
    for mol_idx, num_atoms in enumerate(graph_num_atoms):
        start = offset
        end = offset + num_atoms
        offset = end
        logits = site_logits[start:end]
        order = torch.argsort(logits, descending=True)
        k = min(int(top_k), int(num_atoms))
        candidate_local = order[:k]
        candidate_global = candidate_local + start
        candidate_logits = logits[candidate_local]
        rank_feature = torch.linspace(1.0, 0.0, steps=k, device=logits.device, dtype=logits.dtype).unsqueeze(-1)
        next_logits = torch.cat([candidate_logits[1:], candidate_logits[-1:]], dim=0)
        gap_feature = (candidate_logits - next_logits).unsqueeze(-1)
        gap_feature[-1] = 0.0
        feature_parts = [
            atom_features[candidate_global],
            candidate_logits.unsqueeze(-1),
            rank_feature,
            gap_feature,
        ]
        if local_chem is not None:
            feature_parts.append(local_chem[candidate_global])
        if anomaly is not None:
            feature_parts.append(anomaly[mol_idx].view(1, -1).expand(k, -1))
        if event_strain is not None:
            feature_parts.append(event_strain[candidate_global])
        if access_score is not None:
            feature_parts.append(access_score[candidate_global])
        if barrier_score is not None:
            feature_parts.append(barrier_score[candidate_global])
        if bde_values is not None:
            feature_parts.append(bde_values[candidate_global].view(-1, 1))
        candidate_features = torch.cat(feature_parts, dim=-1)
        target_mask = (site_labels[candidate_global] > 0.5).float()
        proposal_hit = bool(target_mask.any())
        mol_edge_index = _local_edge_index(edge_index, start, end)
        mol_coords = _to_numpy(atom_coordinates[start:end]) if atom_coordinates is not None else None
        mol_topology = _to_numpy(topology_atom_features[start:end]) if topology_atom_features is not None else None
        mol_access = _to_numpy(access_score[start:end]).reshape(-1) if access_score is not None else None
        mol_barrier = _to_numpy(barrier_score[start:end]).reshape(-1) if barrier_score is not None else None
        rival_payload = _build_rival_payload(
            smiles=str(canonical_smiles[mol_idx]),
            num_atoms=num_atoms,
            candidate_local=_to_numpy(candidate_local),
            candidate_logits=_to_numpy(candidate_logits),
            local_edge_index=mol_edge_index,
            atom_coordinates=mol_coords,
            topology_atom_features=mol_topology,
            access_values=mol_access,
            barrier_values=mol_barrier,
        )
        rows.append(
            {
                "molecule_id": str((graph_metadata[mol_idx] or {}).get("id", mol_idx)),
                "canonical_smiles": str(canonical_smiles[mol_idx]),
                "source": str((graph_metadata[mol_idx] or {}).get("source", "")),
                "primary_cyp": str((graph_metadata[mol_idx] or {}).get("primary_cyp", "")),
                "candidate_features": candidate_features.detach().cpu(),
                "candidate_mask": torch.ones((k,), dtype=torch.float32),
                "target_mask": target_mask.detach().cpu(),
                "candidate_atom_indices": candidate_local.detach().cpu(),
                "proposal_scores": candidate_logits.detach().cpu(),
                "proposal_top1_index": 0,
                "proposal_top1_is_true": bool(target_mask[0].item() > 0.5),
                "true_site_atoms": [
                    int(idx)
                    for idx in range(num_atoms)
                    if float(site_labels[start + idx].item()) > 0.5
                ],
                **{
                    key: torch.as_tensor(value)
                    for key, value in rival_payload.items()
                },
                "proposal_hit": proposal_hit,
            }
        )
    return rows


def _export_split(model, loader, device, *, top_k: int) -> Dict[str, object]:
    samples = []
    total_molecules = 0
    proposal_hit_molecules = 0
    feature_dim = None
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_to_device(raw_batch, device)
            outputs = model(batch)
            rows = _topk_sample_rows(batch, outputs, top_k=top_k)
            total_molecules += len(rows)
            for row in rows:
                proposal_hit_molecules += int(bool(row.pop("proposal_hit", False)))
                feature_dim = int(row["candidate_features"].shape[-1])
                if bool(row["target_mask"].sum().item() > 0.0):
                    samples.append(row)
    return {
        "samples": samples,
        "summary": {
            "total_molecules": int(total_molecules),
            "proposal_hit_molecules": int(proposal_hit_molecules),
            "proposal_molecule_recall_at_k": float(proposal_hit_molecules) / float(total_molecules) if total_molecules > 0 else 0.0,
            "conditional_sample_count": int(len(samples)),
            "feature_dim": int(feature_dim or 0),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 4 candidate-set data from a frozen proposer")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--xtb-cache-dir", default="cache/full_xtb")
    parser.add_argument("--manual-feature-cache-dir", default="cache/manual_engine_full")
    parser.add_argument("--target-cyp", default="CYP3A4")
    parser.add_argument("--split-mode", default="scaffold_source_size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--site-labeled-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drugs = _load_drugs(Path(args.dataset), target_cyp=str(args.target_cyp or "").strip(), site_labeled_only=bool(args.site_labeled_only))
    train_drugs, val_drugs, test_drugs = split_drugs(
        drugs,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        mode=str(args.split_mode),
    )
    try:
        loaders = create_full_xtb_dataloaders_from_drugs(
            train_drugs,
            val_drugs,
            test_drugs,
            batch_size=int(args.batch_size),
            structure_sdf=str(args.structure_sdf),
            use_manual_engine_features=True,
            manual_feature_cache_dir=str(args.manual_feature_cache_dir),
            full_xtb_cache_dir=str(args.xtb_cache_dir),
            compute_full_xtb_if_missing=False,
            use_candidate_mask=False,
            candidate_cyp=str(args.target_cyp),
            balance_train_sources=False,
            drop_failed=True,
        )
    except RuntimeError:
        loaders = create_full_xtb_dataloaders_from_drugs(
            train_drugs,
            val_drugs,
            test_drugs,
            batch_size=int(args.batch_size),
            structure_sdf=str(args.structure_sdf),
            use_manual_engine_features=False,
            manual_feature_cache_dir=str(args.manual_feature_cache_dir),
            full_xtb_cache_dir=str(args.xtb_cache_dir),
            compute_full_xtb_if_missing=False,
            use_candidate_mask=False,
            candidate_cyp=str(args.target_cyp),
            balance_train_sources=False,
            drop_failed=True,
        )
    model = _load_proposer(Path(args.checkpoint), device)
    split_payload = {}
    for split_name, loader in zip(("train", "val", "test"), loaders):
        split_payload[split_name] = _export_split(model, loader, device, top_k=int(args.top_k))
        print(f"{split_name}: {split_payload[split_name]['summary']}", flush=True)
    feature_dim = 0
    for name in ("train", "val", "test"):
        feature_dim = max(feature_dim, int((split_payload[name]["summary"] or {}).get("feature_dim", 0)))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": {
                "dataset": str(args.dataset),
                "checkpoint": str(args.checkpoint),
                "target_cyp": str(args.target_cyp),
                "split_mode": str(args.split_mode),
                "top_k": int(args.top_k),
                "feature_dim": int(feature_dim),
            },
            "splits": split_payload,
        },
        output_path,
    )
    print(f"saved={output_path}", flush=True)


if __name__ == "__main__":
    main()
