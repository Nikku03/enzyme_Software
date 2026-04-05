import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
from enzyme_software.liquid_nn_v2._compat import torch
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.dataset import create_full_xtb_dataloaders_from_drugs, split_drugs
from enzyme_software.liquid_nn_v2.training.utils import move_to_device
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.model_utils import load_full_xtb_warm_start


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
