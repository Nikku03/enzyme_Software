from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.recursive_metabolism import (
    PathwayGenerator,
    RecursiveMetabolismConfig,
    RecursiveMetabolismDataset,
    RecursiveMetabolismModel,
    RecursiveMetabolismTrainer,
    RecursivePathwayEvaluator,
    collate_recursive_batch,
    load_base_hybrid_checkpoint,
)
from enzyme_software.recursive_metabolism.utils import (
    filter_site_labeled_drugs,
    initialized_state_dict,
    load_drugs,
    resolve_device,
    split_items,
)


def _load_or_generate_pathways(dataset_path: str, pathways_cache: Path, *, site_labeled_only: bool, max_steps: int, min_heavy_atoms: int, pseudo_step_weight: float):
    if pathways_cache.exists():
        print(f"Loading cached pathways from {pathways_cache}", flush=True)
        return json.loads(pathways_cache.read_text())
    drugs = load_drugs(dataset_path)
    if site_labeled_only:
        drugs = filter_site_labeled_drugs(drugs)
    generator = PathwayGenerator(
        max_steps=max_steps,
        min_heavy_atoms=min_heavy_atoms,
        pseudo_step_weight=pseudo_step_weight,
    )
    pathways = [pathway.to_dict() for pathway in generator.generate_dataset(drugs, output_path=pathways_cache)]
    return pathways


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Train recursive metabolism model")
    parser.add_argument("--dataset", default="data/training_dataset_drugbank.json")
    parser.add_argument("--pathways-cache", default="cache/recursive_metabolism/pathways.json")
    parser.add_argument("--checkpoint", default="checkpoints/hybrid_lnn_latest.pt")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--xtb-cache-dir", default="cache/recursive_metabolism/xtb")
    parser.add_argument("--manual-feature-cache-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-ratio", type=float, default=0.68)
    parser.add_argument("--val-ratio", type=float, default=0.16)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--min-heavy-atoms", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--site-labeled-only", action="store_true")
    parser.add_argument("--ground-truth-only", action="store_true")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--compute-xtb-if-missing", action="store_true")
    parser.add_argument("--freeze-base", dest="freeze_base", action="store_true", default=True)
    parser.add_argument("--no-freeze-base", dest="freeze_base", action="store_false")
    parser.add_argument("--unfreeze-after-epochs", type=int, default=0)
    parser.add_argument("--output-dir", default="checkpoints/recursive_metabolism")
    parser.add_argument("--artifact-dir", default="artifacts/recursive_metabolism")
    parser.add_argument("--device", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    args = parser.parse_args()

    config = RecursiveMetabolismConfig.default(
        base_checkpoint=args.checkpoint,
        checkpoint_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        structure_sdf=args.structure_sdf,
        xtb_cache_dir=args.xtb_cache_dir,
        manual_feature_cache_dir=args.manual_feature_cache_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        min_heavy_atoms=args.min_heavy_atoms,
        freeze_base_model=args.freeze_base,
        unfreeze_after_epochs=args.unfreeze_after_epochs,
        compute_xtb_if_missing=args.compute_xtb_if_missing,
        early_stopping_patience=args.early_stopping_patience,
    )
    config.ensure_dirs()
    device = resolve_device(args.device)

    print("=" * 60, flush=True)
    print("RECURSIVE METABOLISM TRAINING", flush=True)
    print("=" * 60, flush=True)
    print(f"Using device: {device}", flush=True)
    print(f"Base checkpoint: {config.base_checkpoint}", flush=True)
    print(f"Output dir: {config.checkpoint_dir}", flush=True)

    pathways_cache = Path(args.pathways_cache)
    pathways_cache.parent.mkdir(parents=True, exist_ok=True)
    pathways = _load_or_generate_pathways(
        args.dataset,
        pathways_cache,
        site_labeled_only=bool(args.site_labeled_only),
        max_steps=config.max_steps,
        min_heavy_atoms=config.min_heavy_atoms,
        pseudo_step_weight=config.pseudo_step_weight,
    )
    total_steps = sum(int(pathway.get("total_steps", len(pathway.get("steps", [])))) for pathway in pathways)
    print(
        f"Pathways={len(pathways)} | recursive_samples={total_steps} | expansion_factor={total_steps / max(1, len(pathways)):.2f}x",
        flush=True,
    )

    train_pathways, val_pathways, test_pathways = split_items(
        pathways,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )
    print(f"train={len(train_pathways)} | val={len(val_pathways)} | test={len(test_pathways)}", flush=True)

    train_dataset = RecursiveMetabolismDataset(
        train_pathways,
        structure_sdf=config.structure_sdf,
        include_manual_engine_features=config.include_manual_engine_features,
        include_xtb_features=config.include_xtb_features,
        xtb_cache_dir=config.xtb_cache_dir,
        compute_xtb_if_missing=config.compute_xtb_if_missing,
        manual_feature_cache_dir=config.manual_feature_cache_dir,
        allow_partial_sanitize=config.allow_partial_sanitize,
        allow_aggressive_repair=config.allow_aggressive_repair,
        drop_failed=config.drop_failed,
        ground_truth_only=bool(args.ground_truth_only),
        max_step=args.max_step,
    )
    val_dataset = RecursiveMetabolismDataset(
        val_pathways,
        structure_sdf=config.structure_sdf,
        include_manual_engine_features=config.include_manual_engine_features,
        include_xtb_features=config.include_xtb_features,
        xtb_cache_dir=config.xtb_cache_dir,
        compute_xtb_if_missing=False,
        manual_feature_cache_dir=config.manual_feature_cache_dir,
        allow_partial_sanitize=config.allow_partial_sanitize,
        allow_aggressive_repair=config.allow_aggressive_repair,
        drop_failed=config.drop_failed,
        ground_truth_only=bool(args.ground_truth_only),
        max_step=args.max_step,
    )
    test_dataset = RecursiveMetabolismDataset(
        test_pathways,
        structure_sdf=config.structure_sdf,
        include_manual_engine_features=config.include_manual_engine_features,
        include_xtb_features=config.include_xtb_features,
        xtb_cache_dir=config.xtb_cache_dir,
        compute_xtb_if_missing=False,
        manual_feature_cache_dir=config.manual_feature_cache_dir,
        allow_partial_sanitize=config.allow_partial_sanitize,
        allow_aggressive_repair=config.allow_aggressive_repair,
        drop_failed=config.drop_failed,
        ground_truth_only=bool(args.ground_truth_only),
        max_step=args.max_step,
    )
    print(
        f"Filtered dataset stats | ground_truth_only={bool(args.ground_truth_only)} | max_step={args.max_step}",
        flush=True,
    )
    print(
        f"Train samples={len(train_dataset)} | Val samples={len(val_dataset)} | Test samples={len(test_dataset)}",
        flush=True,
    )
    print(json.dumps({"train": train_dataset.get_stats(), "val": val_dataset.get_stats(), "test": test_dataset.get_stats()}, indent=2), flush=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_recursive_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_recursive_batch)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_recursive_batch)

    base_model = load_base_hybrid_checkpoint(config.base_checkpoint, device=device)
    model = RecursiveMetabolismModel(base_model, config=config)
    trainer = RecursiveMetabolismTrainer(model=model, config=config, device=device)

    history = []
    best_val = -1.0
    best_state = None
    patience = 0
    for epoch in range(config.epochs):
        train_metrics = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.evaluate(val_loader)
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        print(
            f"Epoch {epoch + 1:3d} | loss={train_metrics.get('loss', float('nan')):.4f} | "
            f"train_top1={train_metrics.get('top1_acc', 0.0):.3f} | "
            f"val_top1={val_metrics.get('top1_acc', 0.0):.3f} | "
            f"val_top3={val_metrics.get('top3_acc', 0.0):.3f} | "
            f"base_val_top1={val_metrics.get('base_top1_acc', 0.0):.3f}",
            flush=True,
        )
        if val_metrics.get("top1_acc", 0.0) > best_val:
            best_val = float(val_metrics["top1_acc"])
            best_state = initialized_state_dict(model)
            patience = 0
        else:
            patience += 1
        if patience >= config.early_stopping_patience:
            print(f"Early stopping after epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    test_metrics = trainer.evaluate(test_loader)
    evaluator = RecursivePathwayEvaluator(
        model,
        device=device,
        structure_library=train_dataset.structure_library,
        xtb_cache_dir=config.xtb_cache_dir,
        compute_xtb_if_missing=False,
        manual_feature_cache_dir=config.manual_feature_cache_dir,
    )
    rollout_metrics = evaluator.evaluate_rollouts(test_pathways, max_steps=config.max_steps)
    final_metrics = dict(test_metrics)
    final_metrics.update(rollout_metrics)
    print("\nTEST", flush=True)
    print(json.dumps(final_metrics, indent=2), flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = Path(config.checkpoint_dir) / "recursive_latest.pt"
    archive_path = Path(config.checkpoint_dir) / f"recursive_{timestamp}.pt"
    report_path = Path(config.artifact_dir) / f"recursive_report_{timestamp}.json"
    payload = {
        "model_state_dict": initialized_state_dict(model),
        "source_checkpoint": str(config.base_checkpoint),
        "recursive_config": config.__dict__,
        "split": {
            "train_pathways": len(train_pathways),
            "val_pathways": len(val_pathways),
            "test_pathways": len(test_pathways),
            "train_ratio": config.train_ratio,
            "val_ratio": config.val_ratio,
            "seed": config.seed,
        },
        "pathway_stats": {
            "num_pathways": len(pathways),
            "recursive_samples": total_steps,
            "expansion_factor": float(total_steps / max(1, len(pathways))),
        },
        "filters": {
            "ground_truth_only": bool(args.ground_truth_only),
            "max_step": args.max_step,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
        },
        "history": history,
        "test_metrics": final_metrics,
    }
    torch.save(payload, latest_path)
    torch.save(payload, archive_path)
    report_path.write_text(json.dumps({"history": history, "test_metrics": final_metrics}, indent=2), encoding="utf-8")
    print(f"Saved latest checkpoint: {latest_path}", flush=True)
    print(f"Saved checkpoint: {archive_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
