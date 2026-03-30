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
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES
from enzyme_software.liquid_nn_v2.training.loss import compute_cyp_weights
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


def _save_training_state(
    *,
    model,
    output_dir: Path,
    args,
    history,
    best_val_top1: float,
    best_state,
    base_config,
    test_metrics=None,
    status: str = "running",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = output_dir / "hybrid_lnn_latest.pt"
    best_path = output_dir / "hybrid_lnn_best.pt"
    archive_path = output_dir / f"hybrid_lnn_{timestamp}.pt"
    report_path = output_dir / f"hybrid_lnn_report_{timestamp}.json"
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
        "status": status,
    }
    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, archive_path)
    if best_state is not None:
        best_checkpoint = dict(checkpoint)
        best_checkpoint["model_state_dict"] = best_state
        torch.save(best_checkpoint, best_path)
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "best_val_top1": best_val_top1,
                "test_metrics": test_metrics,
                "history": history,
            },
            indent=2,
        )
    )
    return latest_path, best_path, archive_path, report_path


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices"))


def _split_drugs(drugs: list[dict], seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1):
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for drug in drugs:
        key = (
            str(drug.get("source", "unknown")),
            str(drug.get("cyp") or drug.get("primary_cyp") or ""),
        )
        grouped.setdefault(key, []).append(drug)

    train_drugs: list[dict] = []
    val_drugs: list[dict] = []
    test_drugs: list[dict] = []
    for bucket in grouped.values():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        n_bucket = len(shuffled)
        if n_bucket <= 2:
            n_val = 0
            n_test = 1 if n_bucket == 2 else 0
        else:
            n_val = int(round(n_bucket * val_ratio))
            n_test = int(round(n_bucket * max(0.0, 1.0 - train_ratio - val_ratio)))
            if n_bucket >= 6:
                n_val = max(1, n_val)
            if n_bucket >= 3:
                n_test = max(1, n_test)
            if n_val + n_test >= n_bucket:
                overflow = n_val + n_test - (n_bucket - 1)
                reduce_val = min(overflow, max(0, n_val - (1 if n_bucket >= 6 else 0)))
                n_val -= reduce_val
                overflow -= reduce_val
                if overflow > 0:
                    n_test = max(1, n_test - overflow)
        n_train = max(1, n_bucket - n_val - n_test)
        train_drugs.extend(shuffled[:n_train])
        val_drugs.extend(shuffled[n_train : n_train + n_val])
        test_drugs.extend(shuffled[n_train + n_val : n_train + n_val + n_test])

    rng.shuffle(train_drugs)
    rng.shuffle(val_drugs)
    rng.shuffle(test_drugs)
    return train_drugs, val_drugs, test_drugs


def _cap_external_train_mix(
    train_drugs: list[dict],
    *,
    seed: int,
    anchor_sources: list[str],
    external_to_anchor_ratio: float,
) -> list[dict]:
    anchor_set = {str(source) for source in anchor_sources}
    anchors = [drug for drug in train_drugs if str(drug.get("source", "unknown")) in anchor_set]
    externals = [drug for drug in train_drugs if str(drug.get("source", "unknown")) not in anchor_set]
    if not anchors or not externals:
        return train_drugs
    max_external = int(round(len(anchors) * max(0.0, float(external_to_anchor_ratio))))
    if max_external <= 0:
        return anchors
    if len(externals) <= max_external:
        return train_drugs
    rng = random.Random(seed)
    sampled_externals = list(externals)
    rng.shuffle(sampled_externals)
    sampled_externals = sampled_externals[:max_external]
    mixed = list(anchors) + sampled_externals
    rng.shuffle(mixed)
    return mixed


def _compute_split_cyp_weights(train_drugs: list[dict]):
    counts = Counter(str(d.get("cyp") or d.get("primary_cyp") or "") for d in train_drugs)
    counts = {cyp: int(counts.get(cyp, 0)) for cyp in MAJOR_CYP_CLASSES}
    return compute_cyp_weights(counts=counts, cyp_order=MAJOR_CYP_CLASSES), counts


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
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--auto-resume-latest", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--anchor-sources", nargs="*", default=["DrugBank", "SuperCYP"])
    parser.add_argument("--external-to-anchor-ratio", type=float, default=None)
    parser.add_argument("--disable-3d-branch", action="store_true")
    parser.add_argument("--disable-nexus-bridge", action="store_true")
    parser.add_argument("--freeze-nexus-memory", action="store_true")
    parser.add_argument("--skip-nexus-memory-rebuild", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("=" * 60, flush=True)
    print("HYBRID LNN: Manual Engine + LNN", flush=True)
    print("=" * 60, flush=True)

    device = _resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.supercyp_dataset:
        supercyp_path = Path(args.supercyp_dataset)
        if not supercyp_path.exists():
            raise FileNotFoundError(f"SuperCYP dataset not found: {supercyp_path}")
        primary_drugs = _load_drugs(dataset_path)
        supercyp_drugs = _load_drugs(supercyp_path)
        merged_drugs = primary_drugs + supercyp_drugs
    else:
        merged_drugs = _load_drugs(dataset_path)

    train_drugs, val_drugs, test_drugs = _split_drugs(
        merged_drugs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    original_train_count = len(train_drugs)
    if args.external_to_anchor_ratio is not None:
        train_drugs = _cap_external_train_mix(
            train_drugs,
            seed=args.seed,
            anchor_sources=list(args.anchor_sources),
            external_to_anchor_ratio=float(args.external_to_anchor_ratio),
        )
        print(
            f"Applied external source cap: train {original_train_count} -> {len(train_drugs)} "
            f"(ratio={args.external_to_anchor_ratio}, anchors={list(args.anchor_sources)})",
            flush=True,
        )
    for split_name, split_drugs in (("train", train_drugs), ("val", val_drugs), ("test", test_drugs)):
        source_counts = Counter(str(d.get("source", "unknown")) for d in split_drugs)
        site_count = sum(1 for d in split_drugs if _has_site_labels(d))
        cyp_counts = Counter(str(d.get("cyp") or d.get("primary_cyp") or "") for d in split_drugs)
        print(
            f"{split_name}: total={len(split_drugs)} | site_supervised={site_count} | "
            f"sources={dict(source_counts)} | cyp={dict(cyp_counts)}",
            flush=True,
        )
    train_loader, val_loader, test_loader = create_dataloaders_from_drugs(
        train_drugs,
        val_drugs,
        test_drugs,
        batch_size=args.batch_size,
        structure_sdf=None if args.disable_3d_branch else args.structure_sdf,
        use_manual_engine_features=True,
        manual_target_bond=args.manual_target_bond,
        manual_feature_cache_dir=args.manual_feature_cache_dir,
        drop_failed=True,
    )

    base_config = ModelConfig.light_advanced(
        use_manual_engine_priors=True,
        use_3d_branch=not bool(args.disable_3d_branch),
        use_nexus_bridge=not bool(args.disable_nexus_bridge),
        nexus_memory_frozen=bool(args.freeze_nexus_memory),
        nexus_rebuild_memory_before_train=not bool(args.skip_nexus_memory_rebuild),
        return_intermediate_stats=True,
        # Architecture improvements
        use_cyp_site_conditioning=True,
        use_cross_atom_attention=True,
        use_bde_prior=True,
    )
    base_model = LiquidMetabolismNetV2(base_config)
    model = HybridLNNModel(base_model)

    warm_start_path = None
    if args.warm_start:
        warm_start_path = Path(args.warm_start)
    elif args.auto_resume_latest:
        candidate = output_dir / "hybrid_lnn_latest.pt"
        if candidate.exists():
            warm_start_path = candidate

    if warm_start_path is not None:
        if not warm_start_path.exists():
            raise FileNotFoundError(f"Warm-start checkpoint not found: {warm_start_path}")
        checkpoint = torch.load(warm_start_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict") or checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded warm-start checkpoint: {warm_start_path}", flush=True)
    else:
        print("No warm-start checkpoint found; starting from current initialization", flush=True)

    if (
        getattr(base_config, "use_nexus_bridge", False)
        and getattr(base_config, "nexus_rebuild_memory_before_train", False)
        and getattr(model, "nexus_bridge", None) is not None
    ):
        memory_stats = model.rebuild_nexus_memory(train_loader, device=device)
        print(
            f"Built NEXUS memory: size={int(memory_stats.get('memory_size', 0.0))} "
            f"from_batches={int(memory_stats.get('batches', 0.0))} "
            f"frozen={'yes' if base_config.nexus_memory_frozen else 'no'}",
            flush=True,
        )

    cyp_class_weights, train_cyp_counts = _compute_split_cyp_weights(train_drugs)
    print(f"Train CYP counts: {train_cyp_counts}", flush=True)
    print(f"Train CYP weights: {[round(float(v), 4) for v in cyp_class_weights.tolist()]}", flush=True)

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
        cyp_class_weights=cyp_class_weights,
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
                    f"elapsed={elapsed_seconds/60.0:.1f}m | "
                    f"eta={eta_seconds/60.0:.1f}m",
                    flush=True,
                )

            latest_path, best_path, _, report_path = _save_training_state(
                model=model,
                output_dir=output_dir,
                args=args,
                history=history,
                best_val_top1=best_val_top1,
                best_state=best_state,
                base_config=base_config,
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
        print("\nInterrupted. Saving current hybrid_lnn progress...", flush=True)
        latest_path, best_path, _, report_path = _save_training_state(
            model=model,
            output_dir=output_dir,
            args=args,
            history=history,
            best_val_top1=best_val_top1,
            best_state=best_state,
            base_config=base_config,
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
        args=args,
        history=history,
        best_val_top1=best_val_top1,
        best_state=best_state,
        base_config=base_config,
        test_metrics=test_metrics,
        status="completed",
    )
    print(f"\nSaved checkpoint: {archive_path}", flush=True)
    print(f"Saved latest checkpoint: {latest_path}", flush=True)
    print(f"Saved best checkpoint: {best_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
