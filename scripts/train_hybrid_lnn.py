from __future__ import annotations

import argparse
import json
import time
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig, TrainingConfig
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.data.dataset_loader import create_dataloaders, create_dataloaders_from_drugs
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
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices"))


def _split_drugs(drugs: list[dict], seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1):
    shuffled = list(drugs)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Train the manual-engine/LNN hybrid model")
    parser.add_argument("--dataset", default="data/training_dataset_580.json")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--manual-target-bond", default=None)
    parser.add_argument("--manual-feature-cache-dir", default=None)
    parser.add_argument("--supercyp-dataset", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("=" * 60, flush=True)
    print("HYBRID LNN: Manual Engine + LNN", flush=True)
    print("=" * 60, flush=True)

    device = _resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    if args.supercyp_dataset:
        supercyp_path = Path(args.supercyp_dataset)
        if not supercyp_path.exists():
            raise FileNotFoundError(f"SuperCYP dataset not found: {supercyp_path}")
        primary_drugs = _load_drugs(dataset_path)
        supercyp_drugs = _load_drugs(supercyp_path)
        merged_drugs = primary_drugs + supercyp_drugs
        train_drugs, val_drugs, test_drugs = _split_drugs(merged_drugs, seed=args.seed)
        for split_name, split_drugs in (("train", train_drugs), ("val", val_drugs), ("test", test_drugs)):
            source_counts = Counter(str(d.get("source", "unknown")) for d in split_drugs)
            site_count = sum(1 for d in split_drugs if _has_site_labels(d))
            print(
                f"{split_name}: total={len(split_drugs)} | site_supervised={site_count} | sources={dict(source_counts)}",
                flush=True,
            )
        train_loader, val_loader, test_loader = create_dataloaders_from_drugs(
            train_drugs,
            val_drugs,
            test_drugs,
            batch_size=args.batch_size,
            structure_sdf=args.structure_sdf,
            use_manual_engine_features=True,
            manual_target_bond=args.manual_target_bond,
            manual_feature_cache_dir=args.manual_feature_cache_dir,
            drop_failed=True,
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            str(dataset_path),
            batch_size=args.batch_size,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=args.seed,
            structure_sdf=args.structure_sdf,
            use_manual_engine_features=True,
            manual_target_bond=args.manual_target_bond,
            manual_feature_cache_dir=args.manual_feature_cache_dir,
        )

    base_model = LiquidMetabolismNetV2(
        ModelConfig.light_advanced(
            use_manual_engine_priors=True,
            use_3d_branch=True,
            return_intermediate_stats=True,
        )
    )
    model = HybridLNNModel(base_model)

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

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_stats = trainer.train_loader_epoch(train_loader)
        val_metrics = trainer.evaluate_loader(val_loader)
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed_seconds = time.perf_counter() - train_start
        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_metrics})

        val_top1 = float(val_metrics.get("site_top1_acc", 0.0))
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
                f"elapsed={elapsed_seconds/60.0:.1f}m | "
                f"eta={eta_seconds/60.0:.1f}m",
                flush=True,
            )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                f"Early stopping after epoch {epoch + 1}: no site_top1 improvement for "
                f"{args.early_stopping_patience} epochs.",
                flush=True,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    print("\n" + "=" * 60, flush=True)
    print("TEST SET EVALUATION", flush=True)
    print("=" * 60, flush=True)
    test_metrics = trainer.evaluate_loader(test_loader)
    print(json.dumps(test_metrics, indent=2), flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "hybrid_lnn_latest.pt"
    archive_path = output_dir / f"hybrid_lnn_{timestamp}.pt"
    report_path = output_dir / f"hybrid_lnn_report_{timestamp}.json"

    checkpoint = {
        "model_state_dict": _initialized_state_dict(model),
        "config": {
            "base_model": ModelConfig.light_advanced(
                use_manual_engine_priors=True,
                use_3d_branch=True,
                return_intermediate_stats=True,
            ).__dict__,
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
    }
    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, archive_path)
    report_path.write_text(json.dumps({"best_val_top1": best_val_top1, "test_metrics": test_metrics}, indent=2))

    print(f"\nSaved checkpoint: {archive_path}", flush=True)
    print(f"Saved latest checkpoint: {latest_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
