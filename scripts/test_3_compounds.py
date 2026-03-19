#!/usr/bin/env python
"""
Run 3 known compounds through Module -1, the full-xTB hybrid model, and the xTB reranker.

Usage:
    cd /Users/deepika/Desktop/books/enzyme_software
    PYTHONPATH=src python scripts/test_3_compounds.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from rdkit import Chem

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
from enzyme_software.liquid_nn_v2.data.dataset_loader import CYPMetabolismDataset, collate_fn
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.dataset import FullXTBHybridDataset
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.config import MicroPatternXTBConfig
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.model import (
    MicroPatternXTBHybridModel,
    load_base_hybrid_checkpoint,
)
from enzyme_software.liquid_nn_v2.features.steric_features import StructureLibrary
from enzyme_software.liquid_nn_v2.features.xtb_features import attach_xtb_features_to_graph
from enzyme_software.liquid_nn_v2.training.utils import move_to_device
TEST_COMPOUNDS: List[Dict[str, object]] = [
    {
        "name": "Ibuprofen",
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "primary_cyp": "CYP2C9",
        "site_atoms": [7],
        "site_type": "benzylic hydroxylation",
        "metabolite": "2-hydroxy-ibuprofen",
    },
    {
        "name": "Dextromethorphan",
        "smiles": "COc1ccc2c(c1)C3CC4C(C2)N(C)CC3C4",
        "primary_cyp": "CYP2D6",
        "site_atoms": [0],
        "site_type": "O-demethylation",
        "metabolite": "dextrorphan",
    },
    {
        "name": "Midazolam",
        "smiles": "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2",
        "primary_cyp": "CYP3A4",
        "site_atoms": [0],
        "site_type": "1-hydroxylation",
        "metabolite": "1-hydroxy-midazolam",
    },
]

CYP_LIST = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]


def _resolve_device(name: Optional[str]):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _format_float(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return "n/a"


def _resolve_checkpoint(user_path: str, fallback_paths: List[str]) -> Path:
    if user_path:
        path = Path(user_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    for raw in fallback_paths:
        path = Path(raw)
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint found in: {fallback_paths}")


def _load_structure_library(structure_sdf: Optional[str]) -> Optional[StructureLibrary]:
    if not structure_sdf:
        return None
    path = Path(structure_sdf)
    if not path.exists():
        print(f"WARNING: structure SDF not found: {path}. Continuing without 3D structure library.")
        return None
    return StructureLibrary.from_sdf(str(path))


def _base_drug_row(compound: Dict[str, object]) -> Dict[str, object]:
    return {
        "name": compound["name"],
        "smiles": compound["smiles"],
        "primary_cyp": compound["primary_cyp"],
        "site_atoms": list(compound.get("site_atoms") or []),
        "source": "manual_test",
        "confidence": "validated",
    }


def _single_item_batch(dataset) -> Dict[str, object]:
    item = dataset[0]
    batch = collate_fn([item])
    if batch is None:
        raise RuntimeError("Failed to build a valid batch for the test compound")
    return batch


def _build_full_xtb_batch(
    compound: Dict[str, object],
    *,
    structure_library: Optional[StructureLibrary],
    xtb_cache_dir: str,
    compute_if_missing: bool,
) -> Dict[str, object]:
    dataset = FullXTBHybridDataset(
        split="test",
        augment=False,
        drugs=[_base_drug_row(compound)],
        structure_library=structure_library,
        use_manual_engine_features=True,
        full_xtb_cache_dir=xtb_cache_dir,
        compute_full_xtb_if_missing=compute_if_missing,
        drop_failed=True,
    )
    return _single_item_batch(dataset)


def _build_reranker_batch(
    compound: Dict[str, object],
    *,
    structure_library: Optional[StructureLibrary],
    xtb_cache_dir: str,
    compute_if_missing: bool,
) -> Dict[str, object]:
    dataset = CYPMetabolismDataset(
        split="test",
        augment=False,
        drugs=[_base_drug_row(compound)],
        structure_library=structure_library,
        use_manual_engine_features=True,
        drop_failed=True,
    )
    item = dataset[0]
    graph = item.get("graph")
    if graph is None:
        raise RuntimeError("Failed to build reranker graph for the test compound")
    item = dict(item)
    item["graph"] = attach_xtb_features_to_graph(
        graph,
        cache_dir=xtb_cache_dir,
        compute_if_missing=compute_if_missing,
    )
    batch = collate_fn([item])
    if batch is None:
        raise RuntimeError("Failed to collate reranker batch for the test compound")
    return batch


def _load_full_xtb_hybrid_model(checkpoint_path: Path, device) -> HybridLNNModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_cfg = payload.get("config", {}).get("base_model") or ModelConfig.light_advanced(
        use_manual_engine_priors=True,
        use_3d_branch=True,
        return_intermediate_stats=True,
    ).__dict__
    base_model = LiquidMetabolismNetV2(ModelConfig(**base_cfg))
    model = HybridLNNModel(base_model)
    state_dict = payload.get("model_state_dict") or payload
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _load_reranker_model(checkpoint_path: Path, device) -> Tuple[MicroPatternXTBHybridModel, Dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_payload = payload.get("config") or {}
    config = MicroPatternXTBConfig.default()
    for key, value in cfg_payload.items():
        if hasattr(config, key):
            setattr(config, key, value)
    source_checkpoint = payload.get("source_checkpoint") or config.base_checkpoint
    config.base_checkpoint = str(source_checkpoint)
    base_model = load_base_hybrid_checkpoint(config.base_checkpoint, device=device)
    model = MicroPatternXTBHybridModel(base_model, config=config)
    model.load_state_dict(payload.get("model_state_dict") or payload, strict=False)
    model.to(device)
    model.eval()
    return model, payload


def _analyze_molecule(compound: Dict[str, object]) -> bool:
    name = str(compound["name"])
    smiles = str(compound["smiles"])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"{name}: invalid SMILES")
        return False
    print(f"Name: {name}")
    print(f"SMILES: {smiles}")
    print(f"Heavy atoms: {mol.GetNumHeavyAtoms()}")
    print(f"Expected CYP: {compound['primary_cyp']}")
    print(f"Expected site atoms: {compound.get('site_atoms')}")
    print(f"Expected transformation: {compound.get('site_type')}")
    return True


def _run_module_minus1(compound: Dict[str, object]) -> Optional[Dict[str, object]]:
    from enzyme_software.modules.module_minus1_reactivity_hub import run_module_minus1_reactivity_hub

    try:
        result = run_module_minus1_reactivity_hub(str(compound["smiles"]), "C-H", None, {})
    except Exception as exc:
        print(f"Module -1 failed: {exc}")
        return None
    if not isinstance(result, dict):
        print("Module -1: no result")
        return None
    print(f"Module -1 status: {result.get('status', 'unknown')}")
    candidates = result.get("candidate_sites") or []
    ranked = sorted(
        [cand for cand in candidates if isinstance(cand, dict)],
        key=lambda cand: float((cand.get("bde") or {}).get("corrected_kj_mol", cand.get("bde_kj_mol", 1.0e9))),
    )
    print("Top Module -1 candidates:")
    for rank, cand in enumerate(ranked[:5], start=1):
        atom_idx = cand.get("atom_index", cand.get("index", -1))
        bde = cand.get("bde") if isinstance(cand.get("bde"), dict) else cand
        corrected = bde.get("corrected_kj_mol", bde.get("bde_kj_mol"))
        vertical = bde.get("xtb_bde_vertical_kj_mol")
        adiabatic = bde.get("xtb_bde_adiabatic_kj_mol")
        relax = None
        if isinstance(vertical, (int, float)) and isinstance(adiabatic, (int, float)):
            relax = float(vertical) - float(adiabatic)
        print(
            f"  {rank}. atom={atom_idx} corrected_bde={_format_float(corrected)} "
            f"vertical={_format_float(vertical)} adiabatic={_format_float(adiabatic)} "
            f"relax={_format_float(relax)}"
        )
    return result


def _top_site_rows(site_scores: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
    ranking = torch.argsort(site_scores, descending=True).tolist()
    return [(int(idx), float(site_scores[idx].item())) for idx in ranking[:top_k]]


def _predict_cyp(cyp_logits: torch.Tensor) -> Tuple[str, List[Tuple[str, float]]]:
    probs = torch.softmax(cyp_logits, dim=-1).squeeze(0)
    rows = [(name, float(probs[idx].item())) for idx, name in enumerate(CYP_LIST)]
    pred_idx = int(torch.argmax(probs).item())
    return CYP_LIST[pred_idx], rows


def _run_full_xtb_hybrid(model: HybridLNNModel, batch: Dict[str, object], device) -> Dict[str, object]:
    with torch.no_grad():
        batch = move_to_device(batch, device)
        outputs = model(batch)
    site_scores = outputs["site_scores"].squeeze(-1).detach().cpu()
    cyp_name, cyp_rows = _predict_cyp(outputs["cyp_logits"].detach().cpu())
    return {
        "site_scores": site_scores,
        "top_sites": _top_site_rows(site_scores),
        "cyp_prediction": cyp_name,
        "cyp_rows": cyp_rows,
        "xtb_valid": float(batch.get("xtb_mol_valid", torch.zeros((1, 1), device=device)).float().mean().item()),
    }


def _run_reranker(model: MicroPatternXTBHybridModel, batch: Dict[str, object], device) -> Dict[str, object]:
    with torch.no_grad():
        batch = move_to_device(batch, device)
        outputs = model(batch)
    base_logits = outputs["base_site_logits"].squeeze(-1).detach().cpu()
    reranked_logits = outputs["reranked_site_logits"].squeeze(-1).detach().cpu()
    base_scores = torch.sigmoid(base_logits)
    reranked_scores = torch.sigmoid(reranked_logits)
    cyp_name, cyp_rows = _predict_cyp(outputs["base_outputs"]["cyp_logits"].detach().cpu())
    return {
        "base_site_scores": base_scores,
        "reranked_site_scores": reranked_scores,
        "base_top_sites": _top_site_rows(base_scores),
        "reranked_top_sites": _top_site_rows(reranked_scores),
        "cyp_prediction": cyp_name,
        "cyp_rows": cyp_rows,
        "stats": outputs.get("stats", {}),
    }


def _evaluate(compound: Dict[str, object], ranking: List[Tuple[int, float]], pred_cyp: str) -> Dict[str, bool]:
    known_sites = [int(v) for v in compound.get("site_atoms") or []]
    top1 = ranking[0][0] if ranking else None
    top3 = [idx for idx, _ in ranking[:3]]
    return {
        "top1_correct": bool(top1 in known_sites),
        "top3_correct": bool(any(idx in known_sites for idx in top3)),
        "cyp_correct": str(pred_cyp) == str(compound["primary_cyp"]),
    }


def _print_prediction_block(title: str, ranking: List[Tuple[int, float]], pred_cyp: str, cyp_rows: List[Tuple[str, float]]) -> None:
    print(title)
    print("  Top site candidates:")
    for rank, (atom_idx, score) in enumerate(ranking[:5], start=1):
        print(f"    {rank}. atom={atom_idx} score={score:.3f}")
    print(f"  Predicted CYP: {pred_cyp}")
    print("  CYP probabilities:")
    for cyp_name, prob in cyp_rows:
        print(f"    {cyp_name}: {prob:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test 3 compounds with the full-xTB hybrid model and xTB reranker")
    parser.add_argument("--hybrid-checkpoint", default="")
    parser.add_argument("--reranker-checkpoint", default="")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--full-xtb-cache-dir", default="cache/full_xtb")
    parser.add_argument("--reranker-xtb-cache-dir", default="cache/full_xtb")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-full-xtb-if-missing", dest="compute_full_xtb_if_missing", action="store_true", default=True)
    parser.add_argument("--no-compute-full-xtb", dest="compute_full_xtb_if_missing", action="store_false")
    parser.add_argument("--compute-reranker-xtb-if-missing", dest="compute_reranker_xtb_if_missing", action="store_true", default=True)
    parser.add_argument("--no-compute-reranker-xtb", dest="compute_reranker_xtb_if_missing", action="store_false")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    hybrid_checkpoint = _resolve_checkpoint(
        args.hybrid_checkpoint,
        [
            "checkpoints/hybrid_full_xtb_391/hybrid_full_xtb_latest.pt",
            "checkpoints/hybrid_full_xtb/hybrid_full_xtb_latest.pt",
        ],
    )
    reranker_checkpoint = _resolve_checkpoint(
        args.reranker_checkpoint,
        [
            "checkpoints/micropattern_xtb_100/micropattern_xtb_latest.pt",
            "checkpoints/micropattern_xtb/micropattern_xtb_latest.pt",
        ],
    )

    structure_library = _load_structure_library(args.structure_sdf)

    _print_header("3-COMPOUND XTB TEST")
    print(f"Device: {device}")
    print(f"Hybrid checkpoint: {hybrid_checkpoint}")
    print(f"Reranker checkpoint: {reranker_checkpoint}")

    hybrid_model = _load_full_xtb_hybrid_model(hybrid_checkpoint, device)
    reranker_model, reranker_payload = _load_reranker_model(reranker_checkpoint, device)
    print(f"Reranker source checkpoint: {reranker_payload.get('source_checkpoint')}")

    summary = []
    for compound in TEST_COMPOUNDS:
        _print_header(f"TESTING {compound['name']}")
        if not _analyze_molecule(compound):
            summary.append(
                {
                    "name": compound["name"],
                    "error": "invalid_smiles",
                }
            )
            continue
        _run_module_minus1(compound)
        try:
            hybrid_batch = _build_full_xtb_batch(
                compound,
                structure_library=structure_library,
                xtb_cache_dir=args.full_xtb_cache_dir,
                compute_if_missing=bool(args.compute_full_xtb_if_missing),
            )
            hybrid_pred = _run_full_xtb_hybrid(hybrid_model, hybrid_batch, device)
            _print_prediction_block(
                "Full-xTB hybrid model:",
                hybrid_pred["top_sites"],
                hybrid_pred["cyp_prediction"],
                hybrid_pred["cyp_rows"],
            )
            print(f"  Full-xTB valid flag: {hybrid_pred['xtb_valid']:.3f}")

            reranker_batch = _build_reranker_batch(
                compound,
                structure_library=structure_library,
                xtb_cache_dir=args.reranker_xtb_cache_dir,
                compute_if_missing=bool(args.compute_reranker_xtb_if_missing),
            )
            reranker_pred = _run_reranker(reranker_model, reranker_batch, device)
            _print_prediction_block(
                "Micropattern xTB reranker:",
                reranker_pred["reranked_top_sites"],
                reranker_pred["cyp_prediction"],
                reranker_pred["cyp_rows"],
            )
            print("  Reranker stats:")
            for key in sorted(reranker_pred["stats"].keys()):
                value = reranker_pred["stats"][key]
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.3f}")
                else:
                    print(f"    {key}: {value}")

            hybrid_eval = _evaluate(compound, hybrid_pred["top_sites"], hybrid_pred["cyp_prediction"])
            reranker_eval = _evaluate(compound, reranker_pred["reranked_top_sites"], reranker_pred["cyp_prediction"])
            print("Evaluation:")
            print(f"  Hybrid top1={hybrid_eval['top1_correct']} top3={hybrid_eval['top3_correct']} cyp={hybrid_eval['cyp_correct']}")
            print(f"  Reranker top1={reranker_eval['top1_correct']} top3={reranker_eval['top3_correct']} cyp={reranker_eval['cyp_correct']}")
            summary.append(
                {
                    "name": compound["name"],
                    "hybrid": hybrid_eval,
                    "reranker": reranker_eval,
                }
            )
        except Exception as exc:
            print(f"ERROR while testing {compound['name']}: {exc}")
            summary.append(
                {
                    "name": compound["name"],
                    "error": str(exc),
                }
            )

    _print_header("SUMMARY")
    completed = [row for row in summary if "hybrid" in row and "reranker" in row]
    total = max(1, len(completed))
    for label in ("hybrid", "reranker"):
        top1 = sum(1 for row in completed if row[label]["top1_correct"])
        top3 = sum(1 for row in completed if row[label]["top3_correct"])
        cyp = sum(1 for row in completed if row[label]["cyp_correct"])
        print(
            f"{label}: top1={top1}/{total} ({top1/total:.1%}) | "
            f"top3={top3}/{total} ({top3/total:.1%}) | "
            f"cyp={cyp}/{total} ({cyp/total:.1%})"
        )
    print("\nPer compound:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
