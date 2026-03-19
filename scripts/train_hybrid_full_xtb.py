from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig, TrainingConfig
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb import (
    create_full_xtb_dataloaders_from_drugs,
    load_full_xtb_warm_start,
    split_drugs,
)
from enzyme_software.liquid_nn_v2.features.xtb_features import FULL_XTB_FEATURE_DIM
from enzyme_software.liquid_nn_v2.training.trainer import Trainer


def _initialized_state_dict(model) -> dict:
    state = {}
    uninitialized_type = getattr(torch.nn.parameter, "UninitializedParameter", ())
    for key, value in model.state_dict().items():
        if isinstance(value, uninitialized_type):
            continue
        state[key] = value.detach().cpu() if hasattr(value, "detach") else value
    return state


def _resolve_device(name: str | None):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices") or drug.get("metabolism_sites"))


def _count_xtb_valid(drugs: list[dict], cache_dir: Path) -> tuple[int, dict[str, int]]:
    from enzyme_software.liquid_nn_v2.features.xtb_features import load_or_compute_full_xtb_features

    valid = 0
    statuses: dict[str, int] = {}
    for drug in drugs:
        smiles = str(drug.get("smiles", "")).strip()
        if not smiles:
            continue
        payload = load_or_compute_full_xtb_features(smiles, cache_dir=cache_dir, compute_if_missing=False)
        if bool(payload.get("xtb_valid")):
            valid += 1
        status = str(payload.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    return valid, statuses


def _save_training_state(
    *,
    model,
    output_dir: Path,
    artifact_dir: Path,
    args,
    history,
    best_val_top1: float,
    best_state,
    base_config,
    xtb_cache_dir: Path,
    xtb_valid_count: int,
    xtb_statuses: dict[str, int],
    test_metrics=None,
    status: str = "running",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = output_dir / "hybrid_full_xtb_latest.pt"
    best_path = output_dir / "hybrid_full_xtb_best.pt"
    archive_path = output_dir / f"hybrid_full_xtb_{timestamp}.pt"
    report_path = artifact_dir / f"hybrid_full_xtb_report_{timestamp}.json"
    checkpoint = {
        "model_state_dict": _initialized_state_dict(model),
        "config": {
            "base_model": base_config.__dict__,
            "hybrid_wrapper": {"prior_weight": float(torch.sigmoid(model.prior_weight_logit).detach().item())},
        },
        "training_config": TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
        ).__dict__,
        "best_val_top1": best_val_top1,
        "test_metrics": test_metrics,
        "history": history,
        "xtb_feature_dim": FULL_XTB_FEATURE_DIM,
        "xtb_cache_dir": str(xtb_cache_dir),
        "status": status,
    }
    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, archive_path)
    if best_state is not None:
        best_checkpoint = dict(checkpoint)
        best_checkpoint["model_state_dict"] = best_state
        best_checkpoint["status"] = f"{status}_best"
        torch.save(best_checkpoint, best_path)
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "best_val_top1": best_val_top1,
                "test_metrics": test_metrics,
                "xtb_feature_dim": FULL_XTB_FEATURE_DIM,
                "xtb_valid_molecules": xtb_valid_count,
                "xtb_statuses": xtb_statuses,
                "history": history,
            },
            indent=2,
        )
    )
    return latest_path, best_path, archive_path, report_path


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Train hybrid model with full xTB manual priors")
    parser.add_argument("--dataset", default="data/training_dataset_drugbank.json")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--checkpoint", default="checkpoints/hybrid_lnn_latest.pt")
    parser.add_argument("--xtb-cache-dir", default="cache/full_xtb")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--manual-target-bond", default=None)
    parser.add_argument("--manual-feature-cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--output-dir", default="checkpoints/hybrid_full_xtb")
    parser.add_argument("--artifact-dir", default="artifacts/hybrid_full_xtb")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--site-labeled-only", action="store_true")
    parser.add_argument("--compute-xtb-if-missing", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    device = _resolve_device(args.device)
    output_dir = Path(args.output_dir)
    artifact_dir = Path(args.artifact_dir)
    xtb_cache_dir = Path(args.xtb_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    xtb_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("HYBRID LNN: FULL XTB MANUAL PRIORS", flush=True)
    print("=" * 60, flush=True)
    print(f"Using device: {device}", flush=True)

    drugs = _load_drugs(dataset_path)
    print(f"Loaded {len(drugs)} drugs", flush=True)
    if args.site_labeled_only:
        drugs = [drug for drug in drugs if _has_site_labels(drug)]
        print(f"Site-labeled: {len(drugs)}", flush=True)
    random.Random(args.seed).shuffle(drugs)
    if args.limit is not None:
        drugs = drugs[: int(args.limit)]
        print(f"Limited to: {len(drugs)}", flush=True)

    train_drugs, val_drugs, test_drugs = split_drugs(drugs, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    for split_name, split_items in (("train", train_drugs), ("val", val_drugs), ("test", test_drugs)):
        source_counts = Counter(str(d.get("source", "DrugBank")) for d in split_items)
        site_count = sum(1 for d in split_items if _has_site_labels(d))
        print(
            f"{split_name}: total={len(split_items)} | site_supervised={site_count} | sources={dict(source_counts)}",
            flush=True,
        )

    xtb_valid_count, xtb_statuses = _count_xtb_valid(drugs, xtb_cache_dir)
    print(f"xTB cache valid molecules: {xtb_valid_count}/{len(drugs)} | statuses={xtb_statuses}", flush=True)

    train_loader, val_loader, test_loader = create_full_xtb_dataloaders_from_drugs(
        train_drugs,
        val_drugs,
        test_drugs,
        batch_size=args.batch_size,
        structure_sdf=args.structure_sdf,
        use_manual_engine_features=True,
        manual_target_bond=args.manual_target_bond,
        manual_feature_cache_dir=args.manual_feature_cache_dir,
        full_xtb_cache_dir=str(xtb_cache_dir),
        compute_full_xtb_if_missing=args.compute_xtb_if_missing,
        drop_failed=True,
    )

    manual_atom_feature_dim = 32 + FULL_XTB_FEATURE_DIM
    # Step 1 atom_input_dim = 146 = 140 base graph features + 6 standard XTB dims.
    # Step 2 appends FULL_XTB_FEATURE_DIM (8) instead of 6, so atom_input_dim = 140 + 8 = 148.
    _BASE_GRAPH_ATOM_DIM = 140
    full_xtb_atom_input_dim = _BASE_GRAPH_ATOM_DIM + FULL_XTB_FEATURE_DIM
    base_config = ModelConfig.light_advanced(
        use_manual_engine_priors=True,
        use_3d_branch=True,
        return_intermediate_stats=True,
        manual_atom_feature_dim=manual_atom_feature_dim,
        atom_input_dim=full_xtb_atom_input_dim,
    )
    base_model = LiquidMetabolismNetV2(base_config)
    model = HybridLNNModel(base_model)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        load_full_xtb_warm_start(
            model,
            checkpoint_path,
            device=device,
            new_manual_atom_dim=manual_atom_feature_dim,
            new_atom_input_dim=full_xtb_atom_input_dim,
        )
        print(f"Loaded warm-start checkpoint: {checkpoint_path}", flush=True)
    else:
        raise FileNotFoundError(f"Warm-start checkpoint not found: {checkpoint_path}")

    trainer = Trainer(
        model=model,
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping_patience,
        ),
        device=device,
    )

    history = []
    best_val_top1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    train_start = time.perf_counter()

    try:
        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            train_stats = trainer.train_loader_epoch(train_loader)
            val_metrics = trainer.evaluate_loader(val_loader)
            epoch_seconds = time.perf_counter() - epoch_start
            elapsed_seconds = time.perf_counter() - train_start
            history.append({"epoch": epoch + 1, "train": train_stats, "val": val_metrics})

            val_top1 = float(val_metrics.get("site_top1_acc", 0.0))
            trainer.step_scheduler(val_top1)
            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_state = _initialized_state_dict(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % max(1, int(args.log_every)) == 0 or epoch == 0:
                avg_epoch_seconds = elapsed_seconds / float(epoch + 1)
                eta_seconds = avg_epoch_seconds * max(0, args.epochs - (epoch + 1))
                print(
                    f"Epoch {epoch + 1:3d} | loss={train_stats.get('total_loss', float('nan')):.4f} | "
                    f"site_loss={train_stats.get('site_loss', float('nan')):.4f} | "
                    f"cyp_loss={train_stats.get('cyp_loss', float('nan')):.4f} | "
                    f"site_top1={val_metrics.get('site_top1_acc', 0.0):.3f} | "
                    f"site_top3={val_metrics.get('site_top3_acc', 0.0):.3f} | "
                    f"cyp_acc={val_metrics.get('accuracy', 0.0):.3f} | "
                    f"cyp_f1={val_metrics.get('f1_macro', 0.0):.3f} | "
                    f"physics_gate={train_stats.get('physics_gate_mean', 0.0):.3f} | "
                    f"epoch_time={epoch_seconds:.1f}s | "
                    f"elapsed={elapsed_seconds / 60.0:.1f}m | "
                    f"eta={eta_seconds / 60.0:.1f}m",
                    flush=True,
                )

            latest_path, best_path, _, report_path = _save_training_state(
                model=model,
                output_dir=output_dir,
                artifact_dir=artifact_dir,
                args=args,
                history=history,
                best_val_top1=best_val_top1,
                best_state=best_state,
                base_config=base_config,
                xtb_cache_dir=xtb_cache_dir,
                xtb_valid_count=xtb_valid_count,
                xtb_statuses=xtb_statuses,
                test_metrics=None,
                status="running",
            )

            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping after epoch {epoch + 1}: no site_top1 improvement for "
                    f"{args.early_stopping_patience} epochs.",
                    flush=True,
                )
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Saving current hybrid_full_xtb progress...", flush=True)
        latest_path, best_path, _, report_path = _save_training_state(
            model=model,
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            args=args,
            history=history,
            best_val_top1=best_val_top1,
            best_state=best_state,
            base_config=base_config,
            xtb_cache_dir=xtb_cache_dir,
            xtb_valid_count=xtb_valid_count,
            xtb_statuses=xtb_statuses,
            test_metrics=None,
            status="interrupted",
        )
        print(f"Saved latest checkpoint: {latest_path}", flush=True)
        print(f"Saved best checkpoint: {best_path}", flush=True)
        print(f"Saved report: {report_path}", flush=True)
        return

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    print("\n" + "=" * 60, flush=True)
    print("TEST SET EVALUATION", flush=True)
    print("=" * 60, flush=True)
    test_metrics = trainer.evaluate_loader(test_loader)
    print(json.dumps(test_metrics, indent=2), flush=True)

    latest_path, best_path, archive_path, report_path = _save_training_state(
        model=model,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        args=args,
        history=history,
        best_val_top1=best_val_top1,
        best_state=best_state,
        base_config=base_config,
        xtb_cache_dir=xtb_cache_dir,
        xtb_valid_count=xtb_valid_count,
        xtb_statuses=xtb_statuses,
        test_metrics=test_metrics,
        status="completed",
    )
    print(f"\nSaved checkpoint: {archive_path}", flush=True)
    print(f"Saved latest checkpoint: {latest_path}", flush=True)
    print(f"Saved best checkpoint: {best_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
