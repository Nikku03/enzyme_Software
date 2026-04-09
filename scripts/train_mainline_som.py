from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb import create_full_xtb_dataloaders_from_drugs
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.model_utils import load_full_xtb_warm_start
from enzyme_software.liquid_nn_v2.model.hybrid_model import HybridLNNModel
from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
from enzyme_software.mainline.config import MainlineRunConfig, get_preset
from enzyme_software.mainline.eval.benchmarks import (
    evaluate_high_confidence_benchmark,
    evaluate_strict_exact_benchmark,
    evaluate_tiered_benchmark,
)
from enzyme_software.mainline.models.winner import build_small_local_winner_head
from enzyme_software.mainline.training import MainlineShortlistFirstTrainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the clean shortlist-first SoM mainline.")
    parser.add_argument("--preset", default="mainline_exact_plus_tiered")
    parser.add_argument("--target-family")
    parser.add_argument("--train-dataset")
    parser.add_argument("--val-dataset")
    parser.add_argument("--test-dataset")
    parser.add_argument("--strict-val-dataset")
    parser.add_argument("--strict-test-dataset")
    parser.add_argument("--tiered-val-dataset")
    parser.add_argument("--tiered-test-dataset")
    parser.add_argument("--structure-sdf")
    parser.add_argument("--xtb-cache-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--shortlist-loss-weight", type=float)
    parser.add_argument("--winner-loss-weight", type=float)
    parser.add_argument("--shortlist-ranking-weight", type=float)
    parser.add_argument("--shortlist-rank-window-weight", type=float)
    parser.add_argument("--shortlist-hard-negative-weight", type=float)
    parser.add_argument("--shortlist-pairwise-margin", type=float)
    parser.add_argument("--shortlist-hard-negative-max-per-true", type=int)
    parser.add_argument("--shortlist-candidate-topk", type=int)
    parser.add_argument("--local-winner-topk", type=int)
    parser.add_argument("--warm-start-checkpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--selection-metric")
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("drugs"), list):
        raise TypeError(f"Expected JSON payload with a 'drugs' list: {path}")
    return payload


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return list(_load_payload(path).get("drugs") or [])


def _load_optional_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    resolved = Path(path)
    if not resolved.exists():
        return []
    return _load_rows(resolved)


def _with_overrides(base: MainlineRunConfig, args: argparse.Namespace) -> MainlineRunConfig:
    return base.with_overrides(
        target_family=args.target_family,
        train_dataset=args.train_dataset,
        val_dataset=args.val_dataset,
        test_dataset=args.test_dataset,
        strict_val_dataset=args.strict_val_dataset,
        strict_test_dataset=args.strict_test_dataset,
        tiered_val_dataset=args.tiered_val_dataset,
        tiered_test_dataset=args.tiered_test_dataset,
        structure_sdf=args.structure_sdf,
        xtb_cache_dir=args.xtb_cache_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        shortlist_loss_weight=args.shortlist_loss_weight,
        winner_loss_weight=args.winner_loss_weight,
        shortlist_ranking_weight=args.shortlist_ranking_weight,
        shortlist_rank_window_weight=args.shortlist_rank_window_weight,
        shortlist_hard_negative_weight=args.shortlist_hard_negative_weight,
        shortlist_pairwise_margin=args.shortlist_pairwise_margin,
        shortlist_hard_negative_max_per_true=args.shortlist_hard_negative_max_per_true,
        shortlist_candidate_topk=args.shortlist_candidate_topk,
        local_winner_topk=args.local_winner_topk,
        warm_start_checkpoint=args.warm_start_checkpoint,
        seed=args.seed,
        selection_metric=args.selection_metric,
    )


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _metric_path(metrics: dict[str, Any], path: str) -> float:
    current: Any = metrics
    for token in str(path or "").split("."):
        if not token:
            continue
        if not isinstance(current, dict):
            return float("-inf")
        current = current.get(token)
    try:
        return float(current)
    except Exception:
        return float("-inf")


def _build_benchmark_report(trainer: MainlineShortlistFirstTrainer, strict_rows: list[dict[str, Any]], tiered_rows: list[dict[str, Any]], *, thresholds: tuple[float, ...]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    if strict_rows:
        strict_metrics = evaluate_strict_exact_benchmark(strict_rows)
        report["strict_exact"] = strict_metrics
        report["high_confidence"] = evaluate_high_confidence_benchmark(strict_rows, thresholds=list(thresholds))
    if tiered_rows:
        report["tier_aware"] = evaluate_tiered_benchmark(tiered_rows)
    return report


def _collect_benchmark_predictions(
    trainer: MainlineShortlistFirstTrainer,
    *,
    strict_loader=None,
    tiered_loader=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strict_rows = trainer.collect_predictions(strict_loader) if strict_loader is not None else []
    tiered_rows = trainer.collect_predictions(tiered_loader) if tiered_loader is not None else []
    return strict_rows, tiered_rows


def _build_loader_from_rows(rows: list[dict[str, Any]], *, config: MainlineRunConfig):
    if not rows:
        return None
    loader, _, _ = create_full_xtb_dataloaders_from_drugs(
        rows,
        rows,
        rows,
        batch_size=int(config.batch_size),
        seed=int(config.seed),
        cyp_classes=[str(config.target_family)],
        structure_sdf=str(config.resolve_path(config.structure_sdf)),
        use_manual_engine_features=True,
        full_xtb_cache_dir=str(config.resolve_path(config.xtb_cache_dir)),
        compute_full_xtb_if_missing=False,
        use_candidate_mask=True,
        candidate_cyp=str(config.target_family),
        balance_train_sources=False,
        drop_failed=True,
    )
    return loader


def main() -> None:
    require_torch()
    args = _parse_args()
    config = _with_overrides(get_preset(args.preset), args)
    _seed_everything(int(config.seed))
    output_dir = config.resolve_path(config.output_dir)
    artifact_dir = output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _load_rows(config.resolve_path(config.train_dataset))
    val_rows = _load_rows(config.resolve_path(config.val_dataset))
    test_rows = _load_rows(config.resolve_path(config.test_dataset))
    strict_val_rows = _load_optional_rows(config.resolve_path(config.strict_val_dataset) if config.strict_val_dataset else None)
    strict_test_rows = _load_optional_rows(config.resolve_path(config.strict_test_dataset) if config.strict_test_dataset else None)
    tiered_val_rows = _load_optional_rows(config.resolve_path(config.tiered_val_dataset) if config.tiered_val_dataset else None)
    tiered_test_rows = _load_optional_rows(config.resolve_path(config.tiered_test_dataset) if config.tiered_test_dataset else None)

    train_loader, val_loader, test_loader = create_full_xtb_dataloaders_from_drugs(
        train_rows,
        val_rows,
        test_rows,
        batch_size=int(config.batch_size),
        seed=int(config.seed),
        cyp_classes=[str(config.target_family)],
        structure_sdf=str(config.resolve_path(config.structure_sdf)),
        use_manual_engine_features=True,
        full_xtb_cache_dir=str(config.resolve_path(config.xtb_cache_dir)),
        compute_full_xtb_if_missing=False,
        use_candidate_mask=True,
        candidate_cyp=str(config.target_family),
        balance_train_sources=False,
        drop_failed=True,
    )
    strict_val_loader = _build_loader_from_rows(strict_val_rows, config=config)
    strict_test_loader = _build_loader_from_rows(strict_test_rows, config=config)
    tiered_val_loader = _build_loader_from_rows(tiered_val_rows, config=config)
    tiered_test_loader = _build_loader_from_rows(tiered_test_rows, config=config)

    base_model = LiquidMetabolismNetV2(config.build_model_config())
    model = HybridLNNModel(base_model)
    if str(config.warm_start_checkpoint or "").strip():
        load_report = load_full_xtb_warm_start(
            model,
            config.resolve_path(config.warm_start_checkpoint),
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            new_manual_atom_dim=int(model.config.manual_atom_feature_dim),
            new_atom_input_dim=int(model.config.atom_input_dim),
        )
        print(f"Warm start loaded={load_report.get('loaded', 0)} missing={load_report.get('missing', 0)} mismatch={load_report.get('mismatch', 0)}", flush=True)
    winner_head = build_small_local_winner_head(int(model.config.som_branch_dim))
    trainer = MainlineShortlistFirstTrainer(
        model=model,
        winner_head=winner_head,
        shortlist_loss_weight=float(config.shortlist_loss_weight),
        winner_loss_weight=float(config.winner_loss_weight),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        max_grad_norm=float(config.max_grad_norm),
        shortlist_candidate_topk=int(config.shortlist_candidate_topk),
        local_winner_topk=int(config.local_winner_topk),
        shortlist_ranking_weight=float(config.shortlist_ranking_weight),
        shortlist_rank_window_weight=float(config.shortlist_rank_window_weight),
        shortlist_hard_negative_weight=float(config.shortlist_hard_negative_weight),
        shortlist_pairwise_margin=float(config.shortlist_pairwise_margin),
        shortlist_hard_negative_max_per_true=int(config.shortlist_hard_negative_max_per_true),
        shortlist_use_rank_weighting=bool(config.shortlist_use_rank_weighting),
    )

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch = 0
    patience_left = int(config.patience)
    best_checkpoint = None
    latest_path = output_dir / "mainline_som_latest.pt"
    best_path = output_dir / "mainline_som_best.pt"

    for epoch in range(1, int(config.epochs) + 1):
        print(f"Epoch {epoch}/{config.epochs}", flush=True)
        train_metrics = trainer.train_loader_epoch(train_loader)
        val_metrics = trainer.evaluate_loader(val_loader)
        strict_val_predictions, tiered_val_predictions = _collect_benchmark_predictions(
            trainer,
            strict_loader=strict_val_loader,
            tiered_loader=tiered_val_loader,
        )
        val_benchmarks = _build_benchmark_report(
            trainer,
            strict_val_predictions,
            tiered_val_predictions,
            thresholds=tuple(config.high_confidence_thresholds),
        )
        selection_value = _metric_path(val_benchmarks, str(config.selection_metric))
        history.append(
            {
                "epoch": int(epoch),
                "train": train_metrics,
                "val": val_metrics,
                "val_benchmarks": val_benchmarks,
                "selection_metric": float(selection_value),
            }
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "winner_head_state_dict": winner_head.state_dict(),
            "config": config.as_dict(),
            "history": history,
            "epoch": int(epoch),
            "selection_metric": str(config.selection_metric),
            "trainable_module_summary": trainer.trainable_module_summary,
            "frozen_module_summary": trainer.frozen_module_summary,
        }
        torch.save(checkpoint, latest_path)
        if selection_value > best_metric:
            best_metric = float(selection_value)
            best_epoch = int(epoch)
            patience_left = int(config.patience)
            best_checkpoint = copy.deepcopy(checkpoint)
            torch.save(best_checkpoint, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_checkpoint is None:
        raise RuntimeError("Mainline training did not produce a best checkpoint")
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=False)
    winner_head.load_state_dict(best_checkpoint["winner_head_state_dict"], strict=False)

    final_train_metrics = trainer.evaluate_loader(train_loader)
    final_val_metrics = trainer.evaluate_loader(val_loader)
    final_test_metrics = trainer.evaluate_loader(test_loader)
    strict_test_predictions, tiered_test_predictions = _collect_benchmark_predictions(
        trainer,
        strict_loader=strict_test_loader,
        tiered_loader=tiered_test_loader,
    )
    test_benchmarks = _build_benchmark_report(
        trainer,
        strict_test_predictions,
        tiered_test_predictions,
        thresholds=tuple(config.high_confidence_thresholds),
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = artifact_dir / f"mainline_som_report_{timestamp}.json"
    report = {
        "preset": str(config.preset_name),
        "config": config.as_dict(),
        "best_epoch": int(best_epoch),
        "best_selection_metric": float(best_metric),
        "selection_metric_name": str(config.selection_metric),
        "history": history,
        "train_metrics_final": final_train_metrics,
        "val_metrics_final": final_val_metrics,
        "test_metrics_final": final_test_metrics,
        "strict_exact_benchmark": test_benchmarks.get("strict_exact", {}),
        "tier_aware_benchmark": test_benchmarks.get("tier_aware", {}),
        "high_confidence_benchmark": test_benchmarks.get("high_confidence", {}),
        "trainable_module_summary": trainer.trainable_module_summary,
        "frozen_module_summary": trainer.frozen_module_summary,
        "paths": {
            "latest_checkpoint": str(latest_path),
            "best_checkpoint": str(best_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["paths"], indent=2))


if __name__ == "__main__":
    main()
