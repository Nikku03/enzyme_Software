from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.config import MicroPatternXTBConfig
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.dataset import (
    create_micropattern_dataloaders_from_drugs,
    filter_site_labeled_drugs,
    print_split_summary,
    split_drugs,
)
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.model import (
    MicroPatternXTBHybridModel,
    load_base_hybrid_checkpoint,
)
from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.trainer import MicroPatternTrainer


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


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
    config,
    args,
    train_drugs,
    val_drugs,
    test_drugs,
    history,
    best_val,
    best_state,
    test_stats=None,
    status: str = "running",
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(config.checkpoint_dir)
    artifact_dir = Path(config.artifact_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "micropattern_xtb_latest.pt"
    best_path = checkpoint_dir / "micropattern_xtb_best.pt"
    archive_path = checkpoint_dir / f"micropattern_xtb_{timestamp}.pt"
    report_path = artifact_dir / f"micropattern_xtb_report_{timestamp}.json"
    payload = {
        "model_state_dict": _initialized_state_dict(model),
        "source_checkpoint": str(config.base_checkpoint),
        "freeze_base_model": bool(config.freeze_base_model),
        "config": config.__dict__,
        "split": {
            "seed": args.seed,
            "site_labeled_only": bool(args.site_labeled_only),
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "counts": {
                "train": len(train_drugs),
                "val": len(val_drugs),
                "test": len(test_drugs),
            },
        },
        "best_val_reranked_top1": float(best_val),
        "history": history,
        "test_metrics": test_stats,
        "status": status,
    }
    torch.save(payload, latest_path)
    if best_state is not None:
        best_payload = dict(payload)
        best_payload["model_state_dict"] = best_state
        best_payload["status"] = f"{status}_best"
        torch.save(best_payload, best_path)
    torch.save(payload, archive_path)
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "best_val_reranked_top1": float(best_val),
                "history": history,
                "test_metrics": test_stats,
            },
            indent=2,
        )
    )
    return latest_path, best_path, archive_path, report_path


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Train the micropattern xTB reranker experiment")
    parser.add_argument("--dataset", default="data/drugbank_standardized.json")
    parser.add_argument("--supercyp-dataset", default=None)
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--checkpoint", default="checkpoints/hybrid_lnn_latest.pt")
    parser.add_argument("--site-labeled-only", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.68)
    parser.add_argument("--val-ratio", type=float, default=0.16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--top-k-candidates", type=int, default=10)
    parser.add_argument("--micropattern-radius", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xtb-cache-dir", default="cache/micropattern_xtb")
    parser.add_argument("--compute-xtb-if-missing", action="store_true")
    parser.add_argument("--mirank-weight", type=float, default=0.75)
    parser.add_argument("--listmle-weight", type=float, default=0.2)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.5)
    parser.add_argument("--freeze-base-model", dest="freeze_base_model", action="store_true", default=True)
    parser.add_argument("--no-freeze-base-model", dest="freeze_base_model", action="store_false")
    parser.add_argument("--unfreeze-after-epochs", type=int, default=0)
    parser.add_argument("--output-dir", default="checkpoints/micropattern_xtb")
    parser.add_argument("--artifact-dir", default="artifacts/micropattern_xtb")
    parser.add_argument("--manual-feature-cache-dir", default=None)
    args = parser.parse_args()

    config = MicroPatternXTBConfig.default(
        base_checkpoint=args.checkpoint,
        checkpoint_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        xtb_cache_dir=args.xtb_cache_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        top_k_candidates=args.top_k_candidates,
        micropattern_radius=args.micropattern_radius,
        mirank_weight=args.mirank_weight,
        listmle_weight=args.listmle_weight,
        hard_negative_fraction=args.hard_negative_fraction,
        freeze_base_model=args.freeze_base_model,
        unfreeze_after_epochs=args.unfreeze_after_epochs,
        compute_xtb_if_missing=args.compute_xtb_if_missing,
    )
    config.ensure_dirs()

    device = _resolve_device(args.device)
    print("=" * 60, flush=True)
    print("MICROPATTERN XTB RERANKER", flush=True)
    print("=" * 60, flush=True)
    print(f"Using device: {device}", flush=True)
    print(f"Base checkpoint: {config.base_checkpoint}", flush=True)
    print(f"Experiment checkpoint dir: {config.checkpoint_dir}", flush=True)

    primary = _load_drugs(Path(args.dataset))
    if args.supercyp_dataset:
        primary.extend(_load_drugs(Path(args.supercyp_dataset)))
    if args.site_labeled_only:
        primary = filter_site_labeled_drugs(primary)
        print(f"Filtered to site-labeled drugs: {len(primary)}", flush=True)
    train_drugs, val_drugs, test_drugs = split_drugs(
        primary,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print_split_summary(train_drugs, val_drugs, test_drugs)

    train_loader, val_loader, test_loader = create_micropattern_dataloaders_from_drugs(
        train_drugs,
        val_drugs,
        test_drugs,
        structure_sdf=args.structure_sdf,
        batch_size=args.batch_size,
        xtb_cache_dir=config.xtb_cache_dir,
        compute_xtb_if_missing=config.compute_xtb_if_missing,
        manual_feature_cache_dir=args.manual_feature_cache_dir,
    )

    base_model = load_base_hybrid_checkpoint(config.base_checkpoint, device=device)
    model = MicroPatternXTBHybridModel(base_model, config=config)
    trainer = MicroPatternTrainer(model=model, config=config, device=device)

    history = []
    best_val = -1.0
    best_state = None
    interrupted = False
    try:
        for epoch in range(config.epochs):
            train_stats = trainer.train_epoch(train_loader, epoch)
            val_stats = trainer.evaluate(val_loader)
            history.append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})
            print(
                f"Epoch {epoch + 1:3d} | loss={train_stats.get('loss', float('nan')):.4f} | "
                f"base_top1={val_stats.get('base_top1', 0.0):.3f} | "
                f"reranked_top1={val_stats.get('reranked_top1', 0.0):.3f} | "
                f"base_top3={val_stats.get('base_top3', 0.0):.3f} | "
                f"reranked_top3={val_stats.get('reranked_top3', 0.0):.3f} | "
                f"candidate_acc={val_stats.get('candidate_accuracy', 0.0):.3f} | "
                f"hard_neg_win={val_stats.get('hard_negative_win_rate', 0.0):.3f} | "
                f"xtb_valid_atoms={val_stats.get('xtb_valid_atoms', 0.0):.3f}",
                flush=True,
            )
            trainer.step_scheduler(val_stats.get("reranked_top1", 0.0))
            if val_stats.get("reranked_top1", 0.0) > best_val:
                best_val = float(val_stats["reranked_top1"])
                best_state = _initialized_state_dict(model)
            latest_path, best_path, _, report_path = _save_training_state(
                model=model,
                config=config,
                args=args,
                train_drugs=train_drugs,
                val_drugs=val_drugs,
                test_drugs=test_drugs,
                history=history,
                best_val=best_val,
                best_state=best_state,
                test_stats=None,
                status="running",
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Saving current micropattern_xtb progress...", flush=True)
        latest_path, best_path, _, report_path = _save_training_state(
            model=model,
            config=config,
            args=args,
            train_drugs=train_drugs,
            val_drugs=val_drugs,
            test_drugs=test_drugs,
            history=history,
            best_val=best_val,
            best_state=best_state,
            test_stats=None,
            status="interrupted",
        )
        print(f"Saved latest checkpoint: {latest_path}", flush=True)
        print(f"Saved best checkpoint: {best_path}", flush=True)
        print(f"Saved report: {report_path}", flush=True)
        return

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    test_stats = trainer.evaluate(test_loader)
    print("\nTEST", flush=True)
    print(json.dumps(test_stats, indent=2), flush=True)

    latest_path, best_path, archive_path, report_path = _save_training_state(
        model=model,
        config=config,
        args=args,
        train_drugs=train_drugs,
        val_drugs=val_drugs,
        test_drugs=test_drugs,
        history=history,
        best_val=best_val,
        best_state=best_state,
        test_stats=test_stats,
        status="completed" if not interrupted else "interrupted",
    )
    print(f"Saved latest checkpoint: {latest_path}", flush=True)
    print(f"Saved best checkpoint: {best_path}", flush=True)
    print(f"Saved checkpoint: {archive_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
