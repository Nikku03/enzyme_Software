"""
generate_oof_cahml_predictions.py

Trains CAHML with K-fold cross-validation to generate out-of-fold (OOF)
predictions for every molecule in the base_predictions cache. These OOF
predictions are leak-free: when the multihead meta-learner trains on this
file, every CAHML prediction was made by a model that never saw that molecule.

The output file is a drop-in replacement for base_predictions.pt — the
multihead meta-learner reads atom_feature_dim/global_feature_dim from it
directly, so it automatically adapts to 4 experts instead of 3.

Molecules that CAHML could not predict (no site labels, chemistry failure,
not in any fold) keep their original 3-model stacking unchanged.

Usage:
    python scripts/generate_oof_cahml_predictions.py \\
        --predictions  cache/meta_learner/base_predictions.pt \\
        --dataset      data/prepared_training/main5_all_models_conservative.json \\
        --output       cache/meta_learner/base_predictions_cahml_oof.pt \\
        --n-folds      5 \\
        --epochs       100 \\
        --patience     15 \\
        --seed         42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from enzyme_software.cahml import CAHML, CAHMLConfig, CAHMLDataset, CAHMLTrainer
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.meta_learner import FeatureStacker


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_drugs(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(
        drug.get("som")
        or drug.get("site_atoms")
        or drug.get("site_atom_indices")
        or drug.get("metabolism_sites")
    )


def _resolve_device(name: Optional[str]):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_inference(
    model, dataset: CAHMLDataset, device
) -> Dict[str, dict]:
    """Run CAHML inference on every item in a dataset.

    Returns {smiles: prediction_dict} where site_scores are sigmoid probs
    (matching the format used by FeatureStacker).
    """
    model.eval()
    results: Dict[str, dict] = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        smiles: str = item["smiles"]
        with torch.no_grad():
            out = model(
                item["mol_features_raw"].to(device),
                item["atom_features_raw"].to(device),
                item["smarts_matches"].to(device),
                item["site_scores_raw"].to(device),
                item["cyp_probs_raw"].to(device),
            )
        results[smiles] = {
            "site_scores": torch.sigmoid(out["site_scores"]).detach().cpu(),
            "cyp_probs": torch.softmax(out["cyp_logits"], dim=-1).detach().cpu(),
            "num_atoms": int(item["num_atoms"].item()),
            "cyp_label": int(item["cyp_label"].item()),
            "site_labels": item["site_labels"].detach().cpu(),
            "reaction_type": out["reaction_type"],
            "trusted_model": out["explanation"]["trusted_model"],
        }
    return results


def _make_config(args, fold_idx: int, work_path: Path) -> CAHMLConfig:
    ckpt_dir = work_path / f"fold_{fold_idx}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return CAHMLConfig(
        checkpoint_dir=str(ckpt_dir),
        artifact_dir=str(ckpt_dir / "artifacts"),
        cache_dir=str(ckpt_dir / "cache"),
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        mirank_weight=args.mirank_weight,
        bce_weight=args.bce_weight,
        listmle_weight=args.listmle_weight,
        focal_weight=args.focal_weight,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    require_torch()

    parser = argparse.ArgumentParser(
        description="Generate OOF CAHML predictions (leak-free stacking)"
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to base_predictions.pt from extract_base_predictions.py",
    )
    parser.add_argument("--dataset", required=True, help="Training dataset JSON")
    parser.add_argument(
        "--output",
        default="cache/meta_learner/base_predictions_cahml_oof.pt",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mirank-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--listmle-weight", type=float, default=0.5)
    parser.add_argument("--focal-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--work-dir",
        default="cache/cahml_oof_folds",
        help="Directory for per-fold CAHML checkpoints",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    work_path = Path(args.work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    # ── load base predictions ────────────────────────────────────────────────
    payload = torch.load(args.predictions, map_location="cpu", weights_only=False)
    base_predictions: dict = payload.get("predictions") or payload
    base_model_names: List[str] = list(
        payload.get("model_names") or ["hybrid_lnn", "hybrid_full_xtb", "micropattern_xtb"]
    )
    print(f"Base predictions: {len(base_predictions)} molecules", flush=True)
    print(f"Base model names: {base_model_names}", flush=True)

    # ── filter drugs to those with site labels present in base_predictions ───
    all_drugs = _load_drugs(Path(args.dataset))
    smiles_in_preds = set(base_predictions.keys())
    eligible = [
        d for d in all_drugs
        if _has_site_labels(d) and str(d.get("smiles", "")).strip() in smiles_in_preds
    ]
    # Deduplicate by SMILES (keep first)
    seen: set = set()
    drugs: List[dict] = []
    for d in eligible:
        s = str(d.get("smiles", "")).strip()
        if s not in seen:
            seen.add(s)
            drugs.append(d)
    random.Random(args.seed).shuffle(drugs)
    print(f"Site-labeled drugs eligible for OOF: {len(drugs)}", flush=True)

    # ── K-fold split ─────────────────────────────────────────────────────────
    n_folds = args.n_folds
    folds: List[List[dict]] = [[] for _ in range(n_folds)]
    for i, d in enumerate(drugs):
        folds[i % n_folds].append(d)

    # ── per-fold training and inference ──────────────────────────────────────
    oof_cahml: Dict[str, dict] = {}

    for fold_idx in range(n_folds):
        print(f"\n{'=' * 55}", flush=True)
        print(f"FOLD {fold_idx + 1}/{n_folds}", flush=True)
        print(f"{'=' * 55}", flush=True)

        val_drugs = folds[fold_idx]
        train_all = [d for k, fold in enumerate(folds) for d in fold if k != fold_idx]

        # Split train_all into CAHML train / internal-val for early stopping
        # Use 85 % train, 15 % internal val (mirrors train_cahml defaults)
        n_internal_val = max(1, int(len(train_all) * 0.15))
        internal_val_drugs = train_all[-n_internal_val:]
        train_drugs = train_all[:-n_internal_val]

        print(
            f"CAHML train={len(train_drugs)}  internal-val={len(internal_val_drugs)}"
            f"  OOF-val={len(val_drugs)}",
            flush=True,
        )

        # CAHMLDataset loads the full predictions file but filters to its drug list
        train_dataset = CAHMLDataset(args.predictions, train_drugs)
        internal_val_dataset = CAHMLDataset(args.predictions, internal_val_drugs)
        oof_val_dataset = CAHMLDataset(args.predictions, val_drugs)

        print(
            f"Dataset rows after chemistry filter: "
            f"train={len(train_dataset)}  internal-val={len(internal_val_dataset)}"
            f"  oof-val={len(oof_val_dataset)}",
            flush=True,
        )

        if len(train_dataset) == 0:
            print(f"WARNING: fold {fold_idx + 1} has 0 training rows — skipping.", flush=True)
            continue

        config = _make_config(args, fold_idx, work_path)
        model = CAHML(config)
        trainer = CAHMLTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=internal_val_dataset,
            config=config,
            device=device,
        )
        trainer.train()

        # Load best checkpoint for inference
        best_path = Path(config.checkpoint_dir) / "cahml_best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt.get("model_state_dict") or ckpt, strict=False)
            print(f"Loaded best checkpoint: {best_path}", flush=True)
        model.to(device)

        fold_preds = _run_inference(model, oof_val_dataset, device)
        oof_cahml.update(fold_preds)
        print(
            f"Fold {fold_idx + 1} OOF predictions: {len(fold_preds)} molecules"
            f"  |  Total so far: {len(oof_cahml)}",
            flush=True,
        )

    print(
        f"\nOOF CAHML predictions collected: {len(oof_cahml)} / {len(drugs)} "
        f"eligible molecules",
        flush=True,
    )

    # ── rebuild stacked predictions with CAHML as 4th expert ─────────────────
    model_names_4 = base_model_names + ["cahml"]
    stacker_4 = FeatureStacker(model_names_4)
    stacker_3 = FeatureStacker(base_model_names)

    updated_predictions: dict = {}
    n_upgraded = 0

    for smiles, stacked in base_predictions.items():
        model_preds = stacked.get("model_predictions") or {}

        # Reconstruct per-model raw dicts for re-stacking
        raw: Dict[str, Optional[dict]] = {name: model_preds.get(name) for name in base_model_names}

        cahml_pred = oof_cahml.get(smiles)
        if cahml_pred is not None:
            raw["cahml"] = cahml_pred
            new_stacked = stacker_4.stack(raw)
            n_upgraded += 1
        else:
            # Molecule has no OOF prediction — keep 3-model stacking
            new_stacked = stacker_3.stack({k: v for k, v in raw.items() if k != "cahml"})

        # Preserve original site_labels (stacker derives them from model preds
        # but model_predictions dict doesn't carry them; restore from original)
        original_labels = stacked.get("site_labels")
        if original_labels is not None:
            new_stacked["site_labels"] = original_labels
            new_stacked["site_supervised"] = torch.tensor(
                bool(torch.any(original_labels.float() > 0.5)), dtype=torch.bool
            )

        # Store per-model prediction dicts (including cahml if available)
        new_stacked["model_predictions"] = {
            name: {
                key: value
                for key, value in (pred or {}).items()
                if key in {
                    "site_scores", "cyp_probs", "base_site_scores", "stats",
                    "num_atoms", "cyp_label", "reaction_type", "trusted_model",
                }
            }
            for name, pred in raw.items()
        }
        updated_predictions[smiles] = new_stacked

    print(f"\nMolecules upgraded to 4-expert stacking: {n_upgraded}", flush=True)
    print(
        f"Molecules kept at 3-expert stacking: {len(updated_predictions) - n_upgraded}",
        flush=True,
    )

    # Determine feature dims actually present in the output
    if updated_predictions:
        first = next(iter(updated_predictions.values()))
        print(
            f"atom_features shape: {tuple(first['atom_features'].shape)}  "
            f"global_features shape: {tuple(first['global_features'].shape)}",
            flush=True,
        )

    # ── save ─────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "predictions": updated_predictions,
            "failures": payload.get("failures", {}),
            "dataset": str(args.dataset),
            "model_names": model_names_4,
            "oof_folds": n_folds,
        },
        out_path,
    )
    print(f"\nSaved: {out_path}", flush=True)
    print(f"Total molecules: {len(updated_predictions)}", flush=True)


if __name__ == "__main__":
    main()
