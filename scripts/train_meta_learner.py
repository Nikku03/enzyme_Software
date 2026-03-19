from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.meta_learner import MetaLearner, MetaLearnerConfig, MetaLearnerDataset, MetaLearnerTrainer


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
    parser = argparse.ArgumentParser(description="Train the stacked meta learner")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--site-labeled-only", dest="site_labeled_only", action="store_true", default=True)
    parser.add_argument("--no-site-labeled-only", dest="site_labeled_only", action="store_false")
    parser.add_argument("--output-dir", default="checkpoints/meta_learner")
    parser.add_argument("--artifact-dir", default="artifacts/meta_learner")
    parser.add_argument("--cache-dir", default="cache/meta_learner")
    args = parser.parse_args()

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

    train_dataset = MetaLearnerDataset(args.predictions, train_drugs)
    val_dataset = MetaLearnerDataset(args.predictions, val_drugs)

    model = MetaLearner(
        n_models=3,
        n_cyp=5,
        atom_feature_dim=11,
        global_feature_dim=19,
        hidden_dim=args.hidden_dim,
        use_attention=True,
    )
    print(f"Meta-learner parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    config = MetaLearnerConfig(
        checkpoint_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        cache_dir=args.cache_dir,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    trainer = MetaLearnerTrainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, config=config, device=_resolve_device(args.device))
    payload = trainer.train()
    print(f"Best validation Top-1: {payload['best_val_top1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
