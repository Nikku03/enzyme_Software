from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb import create_full_xtb_dataloaders_from_drugs
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
    parser = argparse.ArgumentParser(description="Evaluate the clean shortlist-first mainline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preset", default="mainline_exact_plus_tiered")
    parser.add_argument("--strict-dataset")
    parser.add_argument("--tiered-dataset")
    parser.add_argument("--structure-sdf")
    parser.add_argument("--xtb-cache-dir")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs") or [])


def _with_overrides(base: MainlineRunConfig, args: argparse.Namespace) -> MainlineRunConfig:
    return base.with_overrides(
        strict_test_dataset=args.strict_dataset,
        tiered_test_dataset=args.tiered_dataset,
        structure_sdf=args.structure_sdf,
        xtb_cache_dir=args.xtb_cache_dir,
    )


def _build_loader(rows: list[dict], *, config: MainlineRunConfig):
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
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = HybridLNNModel(LiquidMetabolismNetV2(config.build_model_config()))
    winner_head = build_small_local_winner_head(int(model.config.som_branch_dim))
    model.load_state_dict(payload.get("model_state_dict") or payload, strict=False)
    winner_head.load_state_dict(payload.get("winner_head_state_dict") or {}, strict=False)
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
    strict_rows = _load_rows(config.resolve_path(config.strict_test_dataset)) if config.strict_test_dataset else []
    tiered_rows = _load_rows(config.resolve_path(config.tiered_test_dataset)) if config.tiered_test_dataset else []
    strict_loader = _build_loader(strict_rows, config=config)
    tiered_loader = _build_loader(tiered_rows, config=config)
    strict_predictions = trainer.collect_predictions(strict_loader) if strict_loader is not None else []
    tiered_predictions = trainer.collect_predictions(tiered_loader) if tiered_loader is not None else []
    report = {
        "strict_exact_benchmark": evaluate_strict_exact_benchmark(strict_predictions) if strict_predictions else {},
        "tier_aware_benchmark": evaluate_tiered_benchmark(tiered_predictions) if tiered_predictions else {},
        "high_confidence_benchmark": evaluate_high_confidence_benchmark(
            strict_predictions,
            thresholds=list(config.high_confidence_thresholds),
        ) if strict_predictions else {},
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"mainline_benchmark_report_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
