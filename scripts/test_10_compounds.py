#!/usr/bin/env python
"""
Test 10 well-characterized compounds with the base ensemble and multi-head meta learner.

Usage:
    cd /Users/deepika/Desktop/books/enzyme_software
    PYTHONPATH=src python scripts/test_10_compounds.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from enzyme_software.cahml import CAHML
from enzyme_software.cahml.components.chemistry_encoder import ChemistryFeatureExtractor
from enzyme_software.cahml.config import CAHMLConfig
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES
from enzyme_software.meta_learner import FeatureStacker, MultiModelPredictor, load_default_model_specs
from enzyme_software.meta_learner.multi_head_meta_model import MultiHeadMetaLearner


TEST_COMPOUNDS: List[Dict[str, object]] = [
    {
        "name": "Warfarin",
        "smiles": "CC(=O)CC(c1ccccc1)c2c(O)c3ccccc3oc2=O",
        "primary_cyp": "CYP2C9",
        "site_atoms": [6, 7],
        "site_type": "aromatic hydroxylation",
        "clinical_notes": "Narrow therapeutic index, CYP2C9 polymorphisms critical",
    },
    {
        "name": "Omeprazole",
        "smiles": "COc1ccc2nc(Cc3ncc(C)c(OC)c3C)[nH]c2c1S(=O)C",
        "primary_cyp": "CYP2C19",
        "site_atoms": [19],
        "site_type": "sulfoxidation",
        "clinical_notes": "Proton pump inhibitor, CYP2C19 poor metabolizers",
    },
    {
        "name": "Codeine",
        "smiles": "COc1ccc2C3CC4=CC=C(O)C5Oc1c2C35CCN4C",
        "primary_cyp": "CYP2D6",
        "site_atoms": [0],
        "site_type": "O-demethylation -> morphine",
        "clinical_notes": "Prodrug, CYP2D6 ultra-rapid metabolizers at risk",
    },
    {
        "name": "Caffeine",
        "smiles": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
        "primary_cyp": "CYP1A2",
        "site_atoms": [0, 7, 12],
        "site_type": "N-demethylation",
        "clinical_notes": "CYP1A2 probe substrate, smoking induces metabolism",
    },
    {
        "name": "Diazepam",
        "smiles": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
        "primary_cyp": "CYP3A4",
        "site_atoms": [0, 1],
        "site_type": "N-demethylation -> nordiazepam",
        "clinical_notes": "Benzodiazepine, also CYP2C19 substrate",
    },
    {
        "name": "Tolbutamide",
        "smiles": "Cc1ccc(cc1)S(=O)(=O)NC(=O)NCCC",
        "primary_cyp": "CYP2C9",
        "site_atoms": [0],
        "site_type": "methyl hydroxylation",
        "clinical_notes": "CYP2C9 probe substrate, sulfonylurea",
    },
    {
        "name": "Atomoxetine",
        "smiles": "Cc1ccccc1OC(CCNC)c2ccccc2",
        "primary_cyp": "CYP2D6",
        "site_atoms": [0, 8],
        "site_type": "aromatic hydroxylation",
        "clinical_notes": "ADHD medication, CYP2D6 polymorphisms affect exposure",
    },
    {
        "name": "Nifedipine",
        "smiles": "COC(=O)C1=C(C)NC(C)=C(C1c2ccccc2[N+](=O)[O-])C(=O)OC",
        "primary_cyp": "CYP3A4",
        "site_atoms": [11],
        "site_type": "oxidation to pyridine",
        "clinical_notes": "Calcium channel blocker, photosensitive",
    },
    {
        "name": "Phenytoin",
        "smiles": "O=C1NC(=O)C(N1)(c2ccccc2)c3ccccc3",
        "primary_cyp": "CYP2C9",
        "site_atoms": [10, 16],
        "site_type": "aromatic hydroxylation (para)",
        "clinical_notes": "Anticonvulsant, non-linear kinetics",
    },
    {
        "name": "Propranolol",
        "smiles": "CC(C)NCC(O)COc1cccc2ccccc12",
        "primary_cyp": "CYP2D6",
        "site_atoms": [10, 11, 12, 13],
        "site_type": "aromatic hydroxylation",
        "clinical_notes": "Beta-blocker, also CYP1A2 substrate",
    },
]

CYP_LIST = list(MAJOR_CYP_CLASSES)


def _resolve_device(name: str | None):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _top_rows(scores: torch.Tensor, top_k: int = 5) -> List[Tuple[int, float]]:
    ranking = torch.argsort(scores.view(-1), descending=True).tolist()
    flat = scores.view(-1)
    return [(int(idx), float(flat[idx].item())) for idx in ranking[:top_k]]


def _predict_cyp_from_probs(cyp_probs: torch.Tensor) -> Tuple[str, List[Tuple[str, float]]]:
    probs = cyp_probs.view(-1)
    pred_idx = int(torch.argmax(probs).item())
    return CYP_LIST[pred_idx], [(name, float(probs[idx].item())) for idx, name in enumerate(CYP_LIST)]


def _evaluate(compound: Dict[str, object], ranking: List[Tuple[int, float]], pred_cyp: str) -> Dict[str, bool]:
    known_sites = [int(v) for v in compound.get("site_atoms") or []]
    top1 = ranking[0][0] if ranking else None
    top3 = [idx for idx, _ in ranking[:3]]
    return {
        "top1_correct": bool(top1 in known_sites),
        "top3_correct": bool(any(idx in known_sites for idx in top3)),
        "cyp_correct": str(pred_cyp) == str(compound["primary_cyp"]),
    }


def _load_meta_model(checkpoint_path: Path, device, *, n_models: int, atom_feature_dim: int, global_feature_dim: int) -> MultiHeadMetaLearner:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    model = MultiHeadMetaLearner(
        n_models=n_models,
        n_cyp=5,
        atom_feature_dim=atom_feature_dim,
        global_feature_dim=global_feature_dim,
        hidden_dim=int(cfg.get("hidden_dim", 32)),
    )
    state_dict = payload.get("model_state_dict") or payload
    current = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in current and tuple(v.shape) == tuple(current[k].shape)}
    skipped = len(state_dict) - len(compatible)
    if skipped:
        print(f"  [meta] skipped {skipped} incompatible checkpoint tensors (architecture updated)", flush=True)
    model.load_state_dict(compatible, strict=False)
    model.to(device)
    model.eval()
    return model


def _load_cahml_model(checkpoint_path: Path, device) -> CAHML:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = payload.get("config") or {}
    allowed = set(CAHMLConfig().__dict__.keys())
    config = CAHMLConfig(**{key: value for key, value in cfg.items() if key in allowed})
    model = CAHML(config)
    model.load_state_dict(payload.get("model_state_dict") or payload, strict=False)
    model.to(device)
    model.eval()
    return model


def _print_model_block(title: str, ranking: List[Tuple[int, float]], pred_cyp: str, cyp_rows: List[Tuple[str, float]]) -> None:
    print(title)
    print("  Top site candidates:")
    for rank, (atom_idx, score) in enumerate(ranking[:5], start=1):
        print(f"    {rank}. atom={atom_idx} score={score:.3f}")
    print(f"  Predicted CYP: {pred_cyp}")
    print("  CYP probabilities:")
    for cyp_name, prob in cyp_rows:
        print(f"    {cyp_name}: {prob:.3f}")


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Test 10 compounds with the base ensemble and multi-head meta learner")
    parser.add_argument("--meta-checkpoint", default="")
    parser.add_argument("--cahml-checkpoint", default="")
    parser.add_argument("--stack-cahml-under-multihead", action="store_true", help="Append CAHML as a fourth expert before running the multi-head checkpoint")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default="data/test_10_compounds.json")
    args = parser.parse_args()

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"drugs": TEST_COMPOUNDS}, indent=2))

    device = _resolve_device(args.device)
    _print_header("10-COMPOUND TEST SET")
    print(f"Device: {device}")
    print(f"Saved panel JSON: {output_json}")
    print("Resolving meta checkpoint...", flush=True)
    meta_checkpoint = _resolve_checkpoint(
        args.meta_checkpoint,
        [
            "checkpoints/meta_learner_multihead_391_from116/multihead_meta_learner_latest.pt",
            "checkpoints/meta_learner_multihead_391/multihead_meta_learner_latest.pt",
            "checkpoints/meta_learner_multihead_116/multihead_meta_learner_latest.pt",
            "checkpoints/meta_learner_multihead/multihead_meta_learner_latest.pt",
        ],
    )
    try:
        cahml_checkpoint = _resolve_checkpoint(
            args.cahml_checkpoint,
            [
                "checkpoints/cahml/cahml_latest.pt",
                "checkpoints/cahml_391/cahml_latest.pt",
                "checkpoints/cahml_116/cahml_latest.pt",
            ],
        )
    except FileNotFoundError:
        cahml_checkpoint = None
    print(f"Meta checkpoint: {meta_checkpoint}")
    if cahml_checkpoint is not None:
        print(f"CAHML checkpoint: {cahml_checkpoint}")
    print("Loading base model wrappers...", flush=True)

    specs = load_default_model_specs(["hybrid_lnn", "hybrid_full_xtb", "micropattern_xtb"])
    predictor = MultiModelPredictor(specs, device=str(device), structure_sdf=args.structure_sdf)
    base_model_names = [name for name in ["hybrid_lnn", "hybrid_full_xtb", "micropattern_xtb"] if name in predictor.wrappers]
    base_stacker = FeatureStacker(base_model_names)
    final_model_names = list(base_model_names)
    if args.stack_cahml_under_multihead:
        if cahml_checkpoint is None:
            raise FileNotFoundError("CAHML checkpoint required when --stack-cahml-under-multihead is enabled")
        final_model_names.append("cahml")
    stacker = FeatureStacker(final_model_names)
    print(f"Loaded wrappers: {list(predictor.wrappers.keys())}")
    print("Loading multi-head meta model...", flush=True)
    meta_model = _load_meta_model(
        meta_checkpoint,
        device,
        n_models=len(final_model_names),
        atom_feature_dim=stacker.atom_feature_dim,
        global_feature_dim=stacker.global_feature_dim,
    )
    cahml_model = _load_cahml_model(cahml_checkpoint, device) if cahml_checkpoint is not None else None
    chemistry_extractor = ChemistryFeatureExtractor() if cahml_model is not None else None
    print("Starting compound evaluation...", flush=True)

    summary: List[Dict[str, object]] = []

    for compound in TEST_COMPOUNDS:
        _print_header(f"TESTING {compound['name']}")
        print(f"CYP: {compound['primary_cyp']}")
        print(f"Known sites: {compound['site_atoms']} ({compound['site_type']})")
        print(f"Notes: {compound['clinical_notes']}")

        raw_predictions = predictor.predict_all(compound)
        base_stacked = None
        if args.stack_cahml_under_multihead:
            if cahml_model is None or chemistry_extractor is None:
                raise RuntimeError("CAHML model unavailable for stacked evaluation")
            chemistry = chemistry_extractor.extract(str(compound["smiles"]))
            if chemistry is None:
                raise RuntimeError("Failed to extract chemistry features for CAHML stacked evaluation")
            base_stacked = base_stacker.stack(raw_predictions)
            with torch.no_grad():
                cahml_stack_out = cahml_model(
                    chemistry.mol_features_raw.to(device),
                    chemistry.atom_features_raw.to(device),
                    chemistry.smarts_matches.to(device),
                    base_stacked["site_scores_raw"].to(device),
                    base_stacked["cyp_probs_raw"].to(device),
                )
            raw_predictions["cahml"] = {
                "site_scores": torch.sigmoid(cahml_stack_out["site_scores"]).detach().cpu(),
                "cyp_probs": torch.softmax(cahml_stack_out["cyp_logits"], dim=-1).detach().cpu(),
                "site_labels": base_stacked["site_labels"].detach().cpu(),
                "cyp_label": int(base_stacked["cyp_label"].item()),
                "num_atoms": int(base_stacked["num_atoms"].item()),
            }
        stacked = stacker.stack(raw_predictions)

        model_results: Dict[str, Dict[str, object]] = {}
        for model_name in ["hybrid_lnn", "hybrid_full_xtb", "micropattern_xtb"]:
            pred = raw_predictions.get(model_name)
            if pred is None:
                continue
            site_key = "site_scores"
            ranking = _top_rows(pred[site_key])
            pred_cyp, cyp_rows = _predict_cyp_from_probs(pred["cyp_probs"])
            model_results[model_name] = {
                "ranking": ranking,
                "pred_cyp": pred_cyp,
                "cyp_rows": cyp_rows,
                "eval": _evaluate(compound, ranking, pred_cyp),
            }

        with torch.no_grad():
            site_scores, cyp_logits, stats = meta_model(
                stacked["atom_features"].to(device),
                stacked["global_features"].to(device),
                stacked["site_scores_raw"].to(device),
                stacked["cyp_probs_raw"].to(device),
            )
        meta_ranking = _top_rows(site_scores.detach().cpu())
        meta_cyp_probs = torch.softmax(cyp_logits.detach().cpu(), dim=-1)
        meta_pred_cyp, meta_cyp_rows = _predict_cyp_from_probs(meta_cyp_probs)
        meta_eval = _evaluate(compound, meta_ranking, meta_pred_cyp)
        cahml_eval = None
        cahml_summary = None
        if cahml_model is not None and chemistry_extractor is not None:
            chemistry = chemistry_extractor.extract(str(compound["smiles"]))
            if chemistry is not None:
                cahml_inputs = base_stacked if base_stacked is not None else stacked
                with torch.no_grad():
                    cahml_out = cahml_model(
                        chemistry.mol_features_raw.to(device),
                        chemistry.atom_features_raw.to(device),
                        chemistry.smarts_matches.to(device),
                        cahml_inputs["site_scores_raw"].to(device),
                        cahml_inputs["cyp_probs_raw"].to(device),
                    )
                cahml_ranking = _top_rows(cahml_out["site_scores"].detach().cpu())
                cahml_pred_cyp = CYP_LIST[int(cahml_out["cyp_prediction"])]
                cahml_cyp_rows = _predict_cyp_from_probs(torch.softmax(cahml_out["cyp_logits"].detach().cpu(), dim=-1))[1]
                cahml_eval = _evaluate(compound, cahml_ranking, cahml_pred_cyp)
                cahml_summary = {
                    "ranking": cahml_ranking,
                    "pred_cyp": cahml_pred_cyp,
                    "cyp_rows": cahml_cyp_rows,
                    "reaction_type": cahml_out["reaction_type"],
                    "reaction_confidence": float(cahml_out["reaction_confidence"]),
                    "trusted_model": cahml_out["explanation"]["trusted_model"],
                    "confidence_level": cahml_out["explanation"]["confidence_level"],
                    "eval": cahml_eval,
                }

        for model_name in ["hybrid_lnn", "hybrid_full_xtb", "micropattern_xtb"]:
            if model_name in model_results:
                label = {
                    "hybrid_lnn": "Hybrid LNN:",
                    "hybrid_full_xtb": "Hybrid Full xTB:",
                    "micropattern_xtb": "Micropattern xTB Reranker:",
                }[model_name]
                _print_model_block(label, model_results[model_name]["ranking"], model_results[model_name]["pred_cyp"], model_results[model_name]["cyp_rows"])

        _print_model_block("Multi-head meta learner:", meta_ranking, meta_pred_cyp, meta_cyp_rows)
        print(
            "  Attention:"
            f" site={[round(float(v), 3) for v in stats['site_attention'].detach().cpu().tolist()]}"
            f" cyp={[round(float(v), 3) for v in stats['cyp_attention'].detach().cpu().tolist()]}"
        )
        if cahml_summary is not None:
            _print_model_block("CAHML:", cahml_summary["ranking"], cahml_summary["pred_cyp"], cahml_summary["cyp_rows"])
            print(
                f"  Reaction: {cahml_summary['reaction_type']} "
                f"(confidence={cahml_summary['reaction_confidence']:.3f})"
            )
            print(
                f"  Explanation: trusted_model={cahml_summary['trusted_model']} "
                f"confidence={cahml_summary['confidence_level']}"
            )

        print("Evaluation:")
        for model_name in ["hybrid_lnn", "hybrid_full_xtb", "micropattern_xtb"]:
            if model_name in model_results:
                ev = model_results[model_name]["eval"]
                print(f"  {model_name}: top1={ev['top1_correct']} top3={ev['top3_correct']} cyp={ev['cyp_correct']}")
        print(f"  multi_head: top1={meta_eval['top1_correct']} top3={meta_eval['top3_correct']} cyp={meta_eval['cyp_correct']}")
        if cahml_eval is not None:
            print(f"  cahml: top1={cahml_eval['top1_correct']} top3={cahml_eval['top3_correct']} cyp={cahml_eval['cyp_correct']}")

        summary.append(
            {
                "name": compound["name"],
                "hybrid_lnn": model_results.get("hybrid_lnn", {}).get("eval"),
                "hybrid_full_xtb": model_results.get("hybrid_full_xtb", {}).get("eval"),
                "micropattern_xtb": model_results.get("micropattern_xtb", {}).get("eval"),
                "multi_head": meta_eval,
                "cahml": cahml_eval,
                "site_attention": [float(v) for v in stats["site_attention"].detach().cpu().tolist()],
                "cyp_attention": [float(v) for v in stats["cyp_attention"].detach().cpu().tolist()],
            }
        )

    _print_header("SUMMARY")
    completed = [row for row in summary if row.get("multi_head") is not None]
    total = max(1, len(completed))
    for label in ["hybrid_lnn", "hybrid_full_xtb", "micropattern_xtb", "multi_head", "cahml"]:
        rows = [row[label] for row in completed if row.get(label) is not None]
        if not rows:
            continue
        top1 = sum(1 for row in rows if row["top1_correct"])
        top3 = sum(1 for row in rows if row["top3_correct"])
        cyp = sum(1 for row in rows if row["cyp_correct"])
        print(f"{label}: top1={top1}/{len(rows)} ({top1/max(1, len(rows)):.1%}) | top3={top3}/{len(rows)} ({top3/max(1, len(rows)):.1%}) | cyp={cyp}/{len(rows)} ({cyp/max(1, len(rows)):.1%})")

    print("\nPer compound:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
