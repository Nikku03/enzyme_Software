from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from enzyme_software.cahml import CAHML, CAHMLConfig, CAHMLDataset, CAHMLTrainer
from enzyme_software.liquid_nn_v2._compat import require_torch, torch


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices") or drug.get("metabolism_sites"))


def _resolve_device(name: str | None):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Train the Chemistry-Aware Hierarchical Meta-Learner")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--site-labeled-only", dest="site_labeled_only", action="store_true", default=True)
    parser.add_argument("--no-site-labeled-only", dest="site_labeled_only", action="store_false")
    parser.add_argument("--output-dir", default="checkpoints/cahml")
    parser.add_argument("--artifact-dir", default="artifacts/cahml")
    parser.add_argument("--cache-dir", default="cache/cahml")
    parser.add_argument("--no-physics-constraints", dest="use_physics_constraints", action="store_false", default=True)
    parser.add_argument("--mirank-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--listmle-weight", type=float, default=0.5)
    parser.add_argument("--focal-weight", type=float, default=0.2)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.5)
    args = parser.parse_args()

    checkpoint_payload = None
    checkpoint_config = {}
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint_payload.get("config") or {}

    drugs = _load_drugs(Path(args.dataset))
    if args.site_labeled_only:
        drugs = [drug for drug in drugs if _has_site_labels(drug)]
    random.Random(args.seed).shuffle(drugs)
    n_train = int(len(drugs) * args.train_ratio)
    n_val = int(len(drugs) * args.val_ratio)
    train_drugs = drugs[:n_train]
    val_drugs = drugs[n_train : n_train + n_val]
    test_drugs = drugs[n_train + n_val :]
    print(f"Train={len(train_drugs)} Val={len(val_drugs)} Test={len(test_drugs)}", flush=True)

    train_dataset = CAHMLDataset(args.predictions, train_drugs)
    val_dataset = CAHMLDataset(args.predictions, val_drugs)
    print(f"Dataset rows: train={len(train_dataset)} val={len(val_dataset)}", flush=True)

    config_kwargs = {
        key: value
        for key, value in checkpoint_config.items()
        if key in CAHMLConfig().__dict__
    }
    config_kwargs.update(
        {
            "checkpoint_dir": args.output_dir,
            "artifact_dir": args.artifact_dir,
            "cache_dir": args.cache_dir,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "use_physics_constraints": args.use_physics_constraints,
            "mirank_weight": args.mirank_weight,
            "bce_weight": args.bce_weight,
            "listmle_weight": args.listmle_weight,
            "focal_weight": args.focal_weight,
            "ranking_margin": args.ranking_margin,
            "hard_negative_fraction": args.hard_negative_fraction,
        }
    )
    if checkpoint_payload is None:
        config_kwargs["hidden_dim"] = args.hidden_dim
    config = CAHMLConfig(
        **config_kwargs,
    )
    model = CAHML(config)
    if checkpoint_payload is not None:
        state_dict = checkpoint_payload.get("model_state_dict") or checkpoint_payload
        model.load_state_dict(state_dict, strict=False)
        if int(args.hidden_dim) != int(config.hidden_dim):
            print(
                f"Requested hidden_dim={args.hidden_dim} ignored for resume; "
                f"using checkpoint hidden_dim={config.hidden_dim}",
                flush=True,
            )
        print(f"Loaded warm-start checkpoint: {args.checkpoint}", flush=True)
    print(f"CAHML parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    trainer = CAHMLTrainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, config=config, device=_resolve_device(args.device))
    payload = trainer.train()
    print(f"Best validation Top-1: {payload['best_val_top1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
