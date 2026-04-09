from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(SRC), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import torch

import train_hybrid_full_xtb as train_script
from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
from enzyme_software.liquid_nn_v2.data.dataset_loader import _extract_site_atoms
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.dataset import split_drugs
from enzyme_software.liquid_nn_v2.model.two_head_shortlist_winner import WinnerHeadV2, winner_v2_feature_dim
from enzyme_software.liquid_nn_v2.training.hard_negative_mining import _bfs_shortest_paths, _build_local_adjacency
from enzyme_software.liquid_nn_v2.training.pairwise_probe import apply_candidate_mask_to_site_logits
from enzyme_software.liquid_nn_v2.training.two_head_shortlist_winner_v2_rebuild_hard_source_finetune_trainer import (
    TwoHeadShortlistWinnerV2RebuildHardSourceFinetuneTrainer,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit hard-source shortlist/winner failures for two-head CYP3A4 checkpoints.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a two_head_shortlist_winner_v2_rebuild_hard_source_finetune best checkpoint.",
    )
    parser.add_argument("--dataset", default="data/prepared_training/main8_cyp3a4_augmented.json")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--xtb-cache-dir", default="cache/full_xtb")
    parser.add_argument("--manual-target-bond", default=None)
    parser.add_argument("--manual-feature-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=0, help="0 means use the batch size stored in the checkpoint.")
    parser.add_argument("--seed", type=int, default=-1, help="-1 means use the seed stored in the checkpoint.")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--split-mode", default="", choices=("", "random", "scaffold_source", "scaffold_source_size"))
    parser.add_argument("--target-cyp", default="")
    parser.add_argument("--hard-sources", default="attnsom,cyp_dbs_external")
    parser.add_argument("--output-dir", default="artifacts/hard_source_audit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--winner-small-margin-threshold", type=float, default=0.12)
    parser.add_argument("--shortlist-small-margin-threshold", type=float, default=0.05)
    parser.add_argument("--near-cutoff-rank", type=int, default=12)
    parser.add_argument("--near-true-graph-distance", type=int, default=2)
    return parser.parse_args()


def _normalize_source(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_") or "unknown"


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except Exception:
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def _softmax_prob_gap(logits: torch.Tensor) -> tuple[float | None, list[float]]:
    if int(logits.numel()) <= 0:
        return None, []
    probs = torch.softmax(logits, dim=0)
    prob_values = [float(v) for v in probs.detach().cpu().tolist()]
    if int(probs.numel()) == 1:
        return 1.0, prob_values
    order = torch.argsort(probs, descending=True)
    gap = float(probs[order[0]].item() - probs[order[1]].item())
    return gap, prob_values


def _min_true_graph_distance(
    *,
    mol_atom_indices: torch.Tensor,
    edge_index: torch.Tensor | None,
    anchor_global_index: int,
    true_local_indices: list[int],
    device: torch.device,
) -> int | None:
    if not true_local_indices:
        return None
    if edge_index is None:
        return None
    adjacency = _build_local_adjacency(mol_atom_indices, edge_index)
    local_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(mol_atom_indices.tolist())}
    if int(anchor_global_index) not in local_lookup:
        return None
    anchor_local = int(local_lookup[int(anchor_global_index)])
    distances = _bfs_shortest_paths(
        adjacency,
        anchor_local,
        device=device,
        dtype=torch.float32,
    ).detach().cpu()
    finite = []
    for true_local in true_local_indices:
        if 0 <= int(true_local) < int(distances.numel()):
            value = float(distances[int(true_local)].item())
            if math.isfinite(value):
                finite.append(int(round(value)))
    return min(finite) if finite else None


def _load_checkpoint(path: Path, *, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a dict: {path}")
    return checkpoint


def _build_model_and_trainer(checkpoint: dict[str, Any], *, device: torch.device) -> tuple[Any, Any, Any, dict[str, Any]]:
    config_block = dict(checkpoint.get("config") or {})
    base_model_config = dict(config_block.get("base_model") or {})
    branch_config = dict(config_block.get("two_head_shortlist_winner_v2_rebuild_hard_source_finetune") or {})
    if not base_model_config or not branch_config:
        raise KeyError("Checkpoint does not look like a two_head_shortlist_winner_v2_rebuild_hard_source_finetune checkpoint")

    base_config = ModelConfig(**base_model_config)
    base_model = LiquidMetabolismNetV2(base_config)
    model = HybridLNNModel(base_model).to(device)
    model_result = model.load_state_dict(checkpoint.get("model_state_dict") or {}, strict=False)

    atom_dim = int(getattr(base_config, "som_branch_dim", getattr(base_config, "hidden_dim", 128)))
    winner_feature_dim = winner_v2_feature_dim(
        atom_dim,
        use_existing_candidate_features=True,
        use_score_gap_features=True,
        use_rank_features=True,
        use_pairwise_features=True,
        use_graph_local_features=True,
        use_3d_local_features=True,
    )
    winner_hidden_dim = branch_config.get("winner_v2_rebuild_hidden_dim", None)
    winner_head = WinnerHeadV2(
        winner_feature_dim,
        hidden_dim=(int(winner_hidden_dim) if winner_hidden_dim is not None and int(winner_hidden_dim) > 0 else None),
        dropout=float(branch_config.get("winner_v2_rebuild_dropout", 0.1)),
    ).to(device)
    winner_result = winner_head.load_state_dict(checkpoint.get("winner_head_state_dict") or {}, strict=True)

    trainer = TwoHeadShortlistWinnerV2RebuildHardSourceFinetuneTrainer(
        model=model,
        winner_head=winner_head,
        learning_rate=0.0,
        weight_decay=0.0,
        max_grad_norm=0.0,
        frozen_shortlist_topk=int(branch_config.get("frozen_shortlist_topk", 6)),
        winner_v2_rebuild_loss_weight=float(branch_config.get("winner_v2_rebuild_loss_weight", 1.0)),
        shortlist_checkpoint_path=str(branch_config.get("frozen_shortlist_checkpoint_path", "")),
        hard_source_names=str(branch_config.get("hard_source_names", "attnsom,cyp_dbs_external")),
        hard_source_finetune_require_hit=bool(branch_config.get("hard_source_finetune_require_hit", True)),
        hard_source_finetune_skip_non_hard_sources=bool(branch_config.get("hard_source_finetune_skip_non_hard_sources", True)),
        winner_finetune_init_checkpoint_path=str(branch_config.get("winner_finetune_init_checkpoint_path", "")),
        device=device,
    )
    trainer.model.eval()
    trainer.winner_head.eval()
    return model, winner_head, trainer, {
        "base_config": dict(getattr(base_config, "__dict__", {}) or {}),
        "branch_config": branch_config,
        "model_load_summary": {
            "missing_keys": list(getattr(model_result, "missing_keys", []) or []),
            "unexpected_keys": list(getattr(model_result, "unexpected_keys", []) or []),
        },
        "winner_load_summary": {
            "missing_keys": list(getattr(winner_result, "missing_keys", []) or []),
            "unexpected_keys": list(getattr(winner_result, "unexpected_keys", []) or []),
        },
    }


def _make_loader_args(args: argparse.Namespace, checkpoint: dict[str, Any]) -> argparse.Namespace:
    training_cfg = dict(checkpoint.get("training_config") or {})
    split_mode = str(args.split_mode or checkpoint.get("split_mode") or "scaffold_source_size")
    target_cyp = str(args.target_cyp or checkpoint.get("target_cyp") or "CYP3A4")
    seed = int(args.seed if int(args.seed) >= 0 else int(checkpoint.get("seed", 42) or 42))
    batch_size = int(args.batch_size if int(args.batch_size) > 0 else int(training_cfg.get("batch_size", 12) or 12))
    return argparse.Namespace(
        batch_size=batch_size,
        seed=seed,
        structure_sdf=str(args.structure_sdf),
        manual_target_bond=args.manual_target_bond,
        manual_feature_cache_dir=args.manual_feature_cache_dir,
        xtb_cache_dir=str(Path(args.xtb_cache_dir)),
        compute_xtb_if_missing=False,
        use_candidate_mask=False,
        target_cyp=target_cyp,
        balance_train_sources=False,
        split_mode=split_mode,
    )


def _load_split_drugs(args: argparse.Namespace, loader_args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    drugs = train_script._load_drugs(dataset_path)
    target_cyp = str(loader_args.target_cyp or "").strip()
    if target_cyp:
        drugs = [drug for drug in drugs if str(drug.get("cyp") or drug.get("primary_cyp") or "").strip() == target_cyp]
    drugs = [drug for drug in drugs if bool(_extract_site_atoms(drug))]
    train_drugs, val_drugs, test_drugs = split_drugs(
        drugs,
        seed=int(loader_args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        mode=str(loader_args.split_mode),
    )
    return train_drugs, val_drugs, test_drugs


def _winner_topk_lists(
    *,
    logits: torch.Tensor,
    selected_atom_indices_local: list[int],
    topk: int = 3,
) -> tuple[list[int], list[float], list[float]]:
    if int(logits.numel()) <= 0:
        return [], [], []
    order = torch.argsort(logits, descending=True)[: min(int(topk), int(logits.numel()))]
    atom_indices = [int(selected_atom_indices_local[int(idx.item())]) for idx in order]
    probs = torch.softmax(logits, dim=0)
    return (
        atom_indices,
        [float(logits[int(idx.item())].item()) for idx in order],
        [float(probs[int(idx.item())].item()) for idx in order],
    )


def _build_audit_rows_for_loader(
    *,
    trainer: TwoHeadShortlistWinnerV2RebuildHardSourceFinetuneTrainer,
    loader,
    split_name: str,
    hard_source_names: set[str],
    winner_small_margin_threshold: float,
    shortlist_small_margin_threshold: float,
    near_cutoff_rank: int,
    near_true_graph_distance: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = trainer.device
    trainer.model.eval()
    trainer.winner_head.eval()
    with torch.no_grad():
        for raw_batch in loader:
            if raw_batch is None:
                continue
            batch = trainer._prepare_batch(raw_batch)
            outputs = trainer._forward_shortlist_provider(batch)
            atom_features = outputs.get("atom_features")
            shortlist_logits = outputs.get("site_logits")
            if atom_features is None or shortlist_logits is None:
                raise RuntimeError("Expected shortlist provider outputs `atom_features` and `site_logits`")
            shortlist_logits = shortlist_logits.view(-1)
            config = getattr(trainer.model, "config", None)
            masked_shortlist_logits = apply_candidate_mask_to_site_logits(
                shortlist_logits,
                trainer._candidate_mask(batch),
                mask_mode=str(getattr(config, "candidate_mask_mode", "hard") or "hard"),
                logit_bias=float(getattr(config, "candidate_mask_logit_bias", 2.0)),
            )
            shortlist_scores = torch.sigmoid(masked_shortlist_logits)

            batch_index = batch["batch"].view(-1)
            site_labels = batch["site_labels"].view(-1) > 0.5
            supervision = (
                trainer._supervision_mask(batch).view(-1) > 0.5
                if trainer._supervision_mask(batch) is not None
                else torch.ones_like(site_labels, dtype=torch.bool)
            )
            ranking = (
                trainer._candidate_mask(batch).view(-1) > 0.5
                if trainer._candidate_mask(batch) is not None
                else torch.ones_like(site_labels, dtype=torch.bool)
            )
            valid = supervision & ranking
            metadata = list(batch.get("graph_metadata") or [])
            edge_index = batch.get("edge_index")
            atom_coordinates = batch.get("atom_coordinates")
            num_molecules = int(batch_index.max().item()) + 1 if batch_index.numel() else 0

            for mol_idx in range(num_molecules):
                meta = dict(metadata[mol_idx] or {}) if mol_idx < len(metadata) and isinstance(metadata[mol_idx], dict) else {}
                source = _normalize_source(meta.get("source") or meta.get("data_source"))
                if source not in hard_source_names:
                    continue
                mol_atom_indices = torch.nonzero(batch_index == mol_idx, as_tuple=False).view(-1)
                mol_valid = (batch_index == mol_idx) & valid
                if not bool(mol_valid.any()):
                    continue
                mol_indices = torch.nonzero(mol_valid, as_tuple=False).view(-1)
                mol_scores = shortlist_scores[mol_indices]
                mol_labels = site_labels[mol_indices]
                candidate_count = int(mol_indices.numel())
                full_order = torch.argsort(mol_scores, descending=True)
                full_ranked_indices = mol_indices[full_order]
                full_ranked_scores = mol_scores[full_order]
                full_ranked_labels = mol_labels[full_order]
                local_lookup = {int(global_idx): local_idx for local_idx, global_idx in enumerate(mol_atom_indices.tolist())}
                full_ranked_atom_indices = [int(local_lookup[int(idx.item())]) for idx in full_ranked_indices]

                hit_at_6 = bool(full_ranked_labels[: min(6, candidate_count)].any().item())
                hit_at_12 = bool(full_ranked_labels[: min(12, candidate_count)].any().item())
                true_rank = None
                if bool(full_ranked_labels.any()):
                    true_rank = int(torch.nonzero(full_ranked_labels, as_tuple=False).view(-1)[0].item()) + 1

                topk = min(int(trainer.frozen_shortlist_topk), candidate_count)
                selected_order = full_order[:topk]
                selected_indices = mol_indices[selected_order]
                selected_scores = mol_scores[selected_order]
                selected_labels = mol_labels[selected_order]
                selected_atom_indices_local = [int(local_lookup[int(idx.item())]) for idx in selected_indices]

                winner_features = trainer._build_winner_features(
                    atom_features=atom_features,
                    selected_indices=selected_indices,
                    selected_scores=selected_scores,
                    mol_atom_indices=mol_atom_indices,
                    edge_index=edge_index,
                    atom_coordinates=atom_coordinates,
                )
                winner_logits = trainer.winner_head(winner_features).view(-1)
                winner_prob_gap, winner_probs = _softmax_prob_gap(winner_logits)
                winner_order = torch.argsort(winner_logits, descending=True)

                true_local_indices = [int(v) for v in list(meta.get("site_atoms") or []) if isinstance(v, int)]
                true_rank_7_to_12 = bool(true_rank is not None and 7 <= int(true_rank) <= 12)
                hit_at_train_k = bool(selected_labels.any().item())

                deterministic_target_index = None
                deterministic_target_atom_index = None
                winner_true_rank = None
                if hit_at_train_k:
                    positive_local = torch.nonzero(selected_labels, as_tuple=False).view(-1)
                    best_true_local = positive_local[torch.argmax(selected_scores[positive_local])]
                    deterministic_target_index = int(best_true_local.item())
                    deterministic_target_atom_index = int(selected_atom_indices_local[deterministic_target_index])
                    winner_true_rank = int(torch.nonzero(winner_order == deterministic_target_index, as_tuple=False).view(-1)[0].item()) + 1

                predicted_candidate_index = int(winner_order[0].item()) if int(winner_order.numel()) > 0 else None
                predicted_atom_index = (
                    int(selected_atom_indices_local[predicted_candidate_index]) if predicted_candidate_index is not None else None
                )
                winner_correct = bool(hit_at_train_k and predicted_candidate_index == deterministic_target_index)

                top_shortlist_atom_index = int(selected_atom_indices_local[0]) if selected_atom_indices_local else None
                top_shortlist_score = float(selected_scores[0].item()) if int(selected_scores.numel()) > 0 else None
                top_candidate_is_true = bool(top_shortlist_atom_index in true_local_indices) if top_shortlist_atom_index is not None else False

                best_true_score_overall = None
                if bool(mol_labels.any()):
                    best_true_score_overall = float(mol_scores[mol_labels].max().item())
                shortlist_top_vs_true_gap = (
                    float(top_shortlist_score - best_true_score_overall)
                    if top_shortlist_score is not None and best_true_score_overall is not None
                    else None
                )
                shortlist_margin_small = bool(
                    shortlist_top_vs_true_gap is not None and float(shortlist_top_vs_true_gap) <= float(shortlist_small_margin_threshold)
                )

                predicted_global_index = int(selected_indices[predicted_candidate_index].item()) if predicted_candidate_index is not None else None
                top_global_index = int(selected_indices[0].item()) if int(selected_indices.numel()) > 0 else None
                top_candidate_graph_distance_to_true_min = _min_true_graph_distance(
                    mol_atom_indices=mol_atom_indices,
                    edge_index=edge_index,
                    anchor_global_index=int(top_global_index) if top_global_index is not None else -1,
                    true_local_indices=true_local_indices,
                    device=device,
                )
                winner_predicted_graph_distance_to_true_min = _min_true_graph_distance(
                    mol_atom_indices=mol_atom_indices,
                    edge_index=edge_index,
                    anchor_global_index=int(predicted_global_index) if predicted_global_index is not None else -1,
                    true_local_indices=true_local_indices,
                    device=device,
                )
                top_candidate_is_near_true = bool(
                    top_candidate_graph_distance_to_true_min is not None
                    and int(top_candidate_graph_distance_to_true_min) <= int(near_true_graph_distance)
                )

                if hit_at_train_k and winner_correct:
                    error_type = "correct"
                elif not hit_at_6:
                    error_type = "shortlist_miss"
                else:
                    error_type = "winner_miss"

                winner_margin_small = bool(
                    winner_prob_gap is not None and float(winner_prob_gap) <= float(winner_small_margin_threshold)
                )
                is_multi_site = len(true_local_indices) > 1
                manual_review_reasons: list[str] = []
                if is_multi_site:
                    manual_review_reasons.append("multi_site")
                if error_type == "winner_miss" and winner_margin_small:
                    manual_review_reasons.append("winner_margin_small")
                if error_type == "shortlist_miss" and true_rank_7_to_12:
                    manual_review_reasons.append("shortlist_true_rank_near_cutoff")
                if shortlist_margin_small:
                    manual_review_reasons.append("shortlist_margin_small")
                if top_candidate_is_near_true:
                    manual_review_reasons.append("top_candidate_near_true")
                if error_type != "correct" and source == "cyp_dbs_external":
                    manual_review_reasons.append("rare_hard_source")
                needs_manual_review = bool(manual_review_reasons)

                manual_review_priority = 0
                if split_name == "test" and error_type != "correct":
                    manual_review_priority += 100
                elif split_name == "val" and error_type != "correct":
                    manual_review_priority += 60
                if error_type == "shortlist_miss":
                    manual_review_priority += 30
                elif error_type == "winner_miss":
                    manual_review_priority += 20
                if is_multi_site:
                    manual_review_priority += 15
                if true_rank_7_to_12:
                    manual_review_priority += 10
                if winner_margin_small:
                    manual_review_priority += 5
                if shortlist_margin_small:
                    manual_review_priority += 5
                if source == "cyp_dbs_external":
                    manual_review_priority += 3

                winner_top3_atom_indices, winner_top3_logits, winner_top3_probs = _winner_topk_lists(
                    logits=winner_logits,
                    selected_atom_indices_local=selected_atom_indices_local,
                    topk=3,
                )
                row = {
                    "split": split_name,
                    "source": source,
                    "hard_source_name": source,
                    "molecule_id": str(meta.get("id", "")),
                    "molecule_key": int(meta.get("molecule_key", 0) or 0),
                    "molecule_name": str(meta.get("name", "")),
                    "smiles": str(meta.get("smiles", "")),
                    "primary_cyp": str(meta.get("primary_cyp", "")),
                    "site_source": str(meta.get("site_source", "")),
                    "true_site_indices": list(true_local_indices),
                    "true_site_count": int(len(true_local_indices)),
                    "is_multi_site": bool(is_multi_site),
                    "deterministic_target_candidate_index": deterministic_target_index,
                    "deterministic_target_atom_index": deterministic_target_atom_index,
                    "shortlist_candidate_count": int(candidate_count),
                    "shortlist_top6_candidate_indices": full_ranked_atom_indices[: min(6, len(full_ranked_atom_indices))],
                    "shortlist_top12_candidate_indices": full_ranked_atom_indices[: min(12, len(full_ranked_atom_indices))],
                    "shortlist_selected_candidate_indices": list(selected_atom_indices_local),
                    "shortlist_hit_at_6": bool(hit_at_6),
                    "shortlist_hit_at_12": bool(hit_at_12),
                    "shortlist_hit_at_train_k": bool(hit_at_train_k),
                    "shortlist_true_site_rank": true_rank,
                    "shortlist_top_candidate_index": top_shortlist_atom_index,
                    "shortlist_top_candidate_score": top_shortlist_score,
                    "best_true_shortlist_score": best_true_score_overall,
                    "shortlist_top_vs_true_gap": shortlist_top_vs_true_gap,
                    "shortlist_margin_small": bool(shortlist_margin_small),
                    "winner_predicted_candidate_index": predicted_candidate_index,
                    "winner_predicted_atom_index": predicted_atom_index,
                    "winner_true_candidate_rank": winner_true_rank,
                    "winner_correct": bool(winner_correct),
                    "winner_prob_gap": winner_prob_gap,
                    "winner_margin_small": bool(winner_margin_small),
                    "winner_top3_atom_indices": winner_top3_atom_indices,
                    "winner_top3_logits": winner_top3_logits,
                    "winner_top3_probs": winner_top3_probs,
                    "winner_probabilities_all": list(winner_probs),
                    "error_type": error_type,
                    "shortlist_fail": bool(error_type == "shortlist_miss"),
                    "winner_fail": bool(error_type == "winner_miss"),
                    "correct_top1": bool(error_type == "correct"),
                    "multi_positive_case": bool(is_multi_site),
                    "top_candidate_is_true": bool(top_candidate_is_true),
                    "top_candidate_graph_distance_to_true_min": top_candidate_graph_distance_to_true_min,
                    "top_candidate_is_near_true": bool(top_candidate_is_near_true),
                    "winner_predicted_graph_distance_to_true_min": winner_predicted_graph_distance_to_true_min,
                    "true_rank_7_to_12": bool(true_rank_7_to_12),
                    "needs_manual_review": bool(needs_manual_review),
                    "manual_review_reasons": manual_review_reasons,
                    "manual_review_priority": int(manual_review_priority),
                }
                rows.append(row)
    return rows


def _summarize_rows(rows: list[dict[str, Any]], *, hard_source_names: list[str], near_cutoff_rank: int) -> dict[str, Any]:
    def summarize_subset(subset: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        shortlist_miss = [row for row in subset if row["error_type"] == "shortlist_miss"]
        winner_miss = [row for row in subset if row["error_type"] == "winner_miss"]
        correct = [row for row in subset if row["error_type"] == "correct"]
        multi_site = [row for row in subset if row["is_multi_site"]]
        manual_review = [row for row in subset if row["needs_manual_review"]]
        near_cutoff = [row for row in shortlist_miss if row["shortlist_true_site_rank"] is not None and 7 <= int(row["shortlist_true_site_rank"]) <= int(near_cutoff_rank)]
        winner_true_top2 = [row for row in winner_miss if row["winner_true_candidate_rank"] is not None and int(row["winner_true_candidate_rank"]) <= 2]
        winner_true_top3 = [row for row in winner_miss if row["winner_true_candidate_rank"] is not None and int(row["winner_true_candidate_rank"]) <= 3]
        miss_true_ranks = [float(row["shortlist_true_site_rank"]) for row in subset if row["error_type"] != "correct" and row["shortlist_true_site_rank"] is not None]
        return {
            "total_examples": int(total),
            "correct_top1_count": int(len(correct)),
            "shortlist_miss_count": int(len(shortlist_miss)),
            "winner_miss_count": int(len(winner_miss)),
            "multi_site_count": int(len(multi_site)),
            "manual_review_count": int(len(manual_review)),
            "shortlist_true_rank_7_to_12_miss_count": int(len(near_cutoff)),
            "winner_miss_true_in_top2_count": int(len(winner_true_top2)),
            "winner_miss_true_in_top3_count": int(len(winner_true_top3)),
            "avg_true_rank_for_misses": (float(sum(miss_true_ranks) / len(miss_true_ranks)) if miss_true_ranks else None),
            "top1_rate": (float(len(correct)) / float(total) if total > 0 else 0.0),
        }

    hard_rows = [row for row in rows if row["source"] in set(hard_source_names)]
    per_source = {source: summarize_subset([row for row in hard_rows if row["source"] == source]) for source in hard_source_names}
    split_breakdown = {
        split_name: summarize_subset([row for row in hard_rows if row["split"] == split_name])
        for split_name in sorted({str(row["split"]) for row in hard_rows})
    }
    overall = summarize_subset(hard_rows)

    if overall["shortlist_miss_count"] > overall["winner_miss_count"] * 1.25:
        bottleneck = "shortlist"
    elif overall["winner_miss_count"] > overall["shortlist_miss_count"] * 1.25:
        bottleneck = "winner"
    elif overall["multi_site_count"] >= max(3, int(math.ceil(0.25 * max(1, overall["total_examples"])))):
        bottleneck = "labels"
    else:
        bottleneck = "mixed"

    if overall["multi_site_count"] >= max(3, int(math.ceil(0.20 * max(1, overall["total_examples"])))):
        recommendation = "Label cleanup first: many hard-source misses are multi-site or ambiguity-adjacent. Build a gold hard-source eval slice and relabel uncertain cases."
    elif bottleneck == "shortlist":
        recommendation = "Next day should target shortlist recall near the cutoff. Many hard-source failures miss the true site before winner scoring."
    elif bottleneck == "winner":
        recommendation = "Next day should target winner refinement. The shortlist is often adequate, but the winner still misses the true shortlisted atom."
    else:
        recommendation = "Next day should focus on selective hard-source data expansion plus label cleanup. Failures are mixed and concentrated in a small hard-source slice."

    source_error_totals = {
        source: per_source[source]["shortlist_miss_count"] + per_source[source]["winner_miss_count"]
        for source in per_source
    }
    worst_source = max(source_error_totals.items(), key=lambda item: item[1])[0] if source_error_totals else None

    return {
        "hard_source_names": list(hard_source_names),
        "overall": overall,
        "per_source": per_source,
        "split_breakdown": split_breakdown,
        "worst_source": worst_source,
        "dominant_bottleneck": bottleneck,
        "recommendation": recommendation,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = []
    for row in rows:
        normalized = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                normalized[key] = _json_cell(value)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    fieldnames = sorted({key for row in normalized_rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(row)


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = train_script._resolve_device(args.device)
    checkpoint = _load_checkpoint(checkpoint_path, device=device)
    seed = int(args.seed if int(args.seed) >= 0 else int(checkpoint.get("seed", 42) or 42))
    train_script._apply_reproducibility_lock(seed)
    model, winner_head, trainer, load_summary = _build_model_and_trainer(checkpoint, device=device)
    loader_args = _make_loader_args(args, checkpoint)
    train_drugs, val_drugs, test_drugs = _load_split_drugs(args, loader_args)
    (train_loader, val_loader, test_loader), manual_engine_enabled = train_script._build_loaders_with_fallback(
        train_drugs,
        val_drugs,
        test_drugs,
        args=loader_args,
    )
    hard_source_names = sorted(
        {
            _normalize_source(token)
            for token in str(args.hard_sources or "").split(",")
            if str(token).strip()
        }
    )
    if not hard_source_names:
        raise ValueError("No hard sources configured")

    rows = []
    rows.extend(
        _build_audit_rows_for_loader(
            trainer=trainer,
            loader=val_loader,
            split_name="val",
            hard_source_names=set(hard_source_names),
            winner_small_margin_threshold=float(args.winner_small_margin_threshold),
            shortlist_small_margin_threshold=float(args.shortlist_small_margin_threshold),
            near_cutoff_rank=int(args.near_cutoff_rank),
            near_true_graph_distance=int(args.near_true_graph_distance),
        )
    )
    rows.extend(
        _build_audit_rows_for_loader(
            trainer=trainer,
            loader=test_loader,
            split_name="test",
            hard_source_names=set(hard_source_names),
            winner_small_margin_threshold=float(args.winner_small_margin_threshold),
            shortlist_small_margin_threshold=float(args.shortlist_small_margin_threshold),
            near_cutoff_rank=int(args.near_cutoff_rank),
            near_true_graph_distance=int(args.near_true_graph_distance),
        )
    )
    rows.sort(key=lambda row: (str(row["split"]), str(row["source"]), int(row["molecule_key"])))

    summary = _summarize_rows(rows, hard_source_names=hard_source_names, near_cutoff_rank=int(args.near_cutoff_rank))
    summary.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "dataset": str(Path(args.dataset)),
            "structure_sdf": str(args.structure_sdf),
            "seed": int(seed),
            "split_mode": str(loader_args.split_mode),
            "target_cyp": str(loader_args.target_cyp),
            "batch_size": int(loader_args.batch_size),
            "manual_engine_enabled": bool(manual_engine_enabled),
            "winner_small_margin_threshold": float(args.winner_small_margin_threshold),
            "shortlist_small_margin_threshold": float(args.shortlist_small_margin_threshold),
            "near_cutoff_rank": int(args.near_cutoff_rank),
            "near_true_graph_distance": int(args.near_true_graph_distance),
            "load_summary": load_summary,
        }
    )

    gold_slice_candidates = [
        row
        for row in rows
        if row["error_type"] != "correct" and row["split"] in {"val", "test"}
    ]
    gold_slice_candidates.sort(
        key=lambda row: (
            -int(row["manual_review_priority"]),
            0 if row["split"] == "test" else 1,
            str(row["source"]),
            int(row["molecule_key"]),
        )
    )

    audit_csv = output_dir / "hard_source_audit_val_test.csv"
    gold_csv = output_dir / "hard_source_gold_slice_candidates.csv"
    summary_json = output_dir / "hard_source_audit_summary.json"

    _write_csv(audit_csv, rows)
    _write_csv(gold_csv, gold_slice_candidates)
    summary_json.write_text(json.dumps(summary, indent=2))

    overall = summary["overall"]
    print(
        f"Hard-source audit complete | examples={overall['total_examples']} | "
        f"correct={overall['correct_top1_count']} | "
        f"shortlist_miss={overall['shortlist_miss_count']} | "
        f"winner_miss={overall['winner_miss_count']} | "
        f"multi_site={overall['multi_site_count']} | "
        f"manual_review={overall['manual_review_count']}",
        flush=True,
    )
    print(
        f"Worst source={summary.get('worst_source')} | "
        f"dominant_bottleneck={summary.get('dominant_bottleneck')}",
        flush=True,
    )
    print(f"Audit table: {audit_csv}", flush=True)
    print(f"Gold-slice candidates: {gold_csv}", flush=True)
    print(f"Summary JSON: {summary_json}", flush=True)
    print(f"Recommendation: {summary.get('recommendation')}", flush=True)


if __name__ == "__main__":
    main()
