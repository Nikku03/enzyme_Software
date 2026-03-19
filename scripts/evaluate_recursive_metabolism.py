from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.recursive_metabolism import (
    RecursiveMetabolismConfig,
    RecursiveMetabolismDataset,
    RecursiveMetabolismModel,
    RecursiveMetabolismTrainer,
    RecursivePathwayEvaluator,
    collate_recursive_batch,
    load_base_hybrid_checkpoint,
)
from enzyme_software.recursive_metabolism.utils import resolve_device, split_items


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Evaluate recursive metabolism experiment")
    parser.add_argument("--pathways", default="cache/recursive_metabolism/pathways.json")
    parser.add_argument("--base-checkpoint", default="checkpoints/hybrid_lnn_latest.pt")
    parser.add_argument("--recursive-checkpoint", default="checkpoints/recursive_metabolism/recursive_latest.pt")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--xtb-cache-dir", default="cache/recursive_metabolism/xtb")
    parser.add_argument("--manual-feature-cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-ratio", type=float, default=0.68)
    parser.add_argument("--val-ratio", type=float, default=0.16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ground-truth-only", action="store_true")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    pathways = json.loads(Path(args.pathways).read_text())
    _, _, test_pathways = split_items(pathways, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    dataset = RecursiveMetabolismDataset(
        test_pathways,
        structure_sdf=args.structure_sdf,
        include_manual_engine_features=True,
        include_xtb_features=True,
        xtb_cache_dir=args.xtb_cache_dir,
        compute_xtb_if_missing=False,
        manual_feature_cache_dir=args.manual_feature_cache_dir,
        ground_truth_only=bool(args.ground_truth_only),
        max_step=args.max_step,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_recursive_batch)

    config = RecursiveMetabolismConfig.default(
        base_checkpoint=args.base_checkpoint,
        structure_sdf=args.structure_sdf,
        xtb_cache_dir=args.xtb_cache_dir,
        manual_feature_cache_dir=args.manual_feature_cache_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    base_model = load_base_hybrid_checkpoint(args.base_checkpoint, device=device)
    model = RecursiveMetabolismModel(base_model, config=config)
    payload = torch.load(args.recursive_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model_state_dict") or payload, strict=False)

    trainer = RecursiveMetabolismTrainer(model=model, config=config, device=device)
    step_metrics = trainer.evaluate(loader)
    evaluator = RecursivePathwayEvaluator(
        model,
        device=device,
        structure_library=dataset.structure_library,
        xtb_cache_dir=args.xtb_cache_dir,
        compute_xtb_if_missing=False,
        manual_feature_cache_dir=args.manual_feature_cache_dir,
    )
    rollout_metrics = evaluator.evaluate_rollouts(test_pathways, max_steps=config.max_steps)
    metrics = dict(step_metrics)
    metrics.update(rollout_metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
