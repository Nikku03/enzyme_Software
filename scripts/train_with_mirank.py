from __future__ import annotations

import argparse
from pathlib import Path

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.meta_learner.config import MetaLearnerConfig
from enzyme_software.meta_learner.meta_trainer import MetaLearnerDataset
from enzyme_software.meta_learner.multi_head_meta_model import MultiHeadMetaLearner
from enzyme_software.meta_learner.multi_head_trainer import MultiHeadTrainer


def _resolve_device(name: str | None):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_prediction_metadata(path: Path) -> tuple[list[str], int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    predictions = payload.get("predictions") or payload
    model_names = list(payload.get("model_names") or [])
    if not predictions:
        raise ValueError(f"No predictions found in {path}")
    first = next(iter(predictions.values()))
    atom_feature_dim = int(first["atom_features"].shape[1])
    global_feature_dim = int(first["global_features"].shape[0])
    if not model_names:
        n_models = int(first["site_scores_raw"].shape[1])
        model_names = [f"model_{idx}" for idx in range(n_models)]
    return model_names, atom_feature_dim, global_feature_dim


def _load_drugs(path: Path) -> list[dict]:
    import json

    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices") or drug.get("metabolism_sites"))


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Train multi-head meta learner with MIRank-driven site loss")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="checkpoints/mirank")
    parser.add_argument("--artifact-dir", default="artifacts/mirank")
    parser.add_argument("--cache-dir", default="cache/mirank")
    parser.add_argument("--mirank-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--listmle-weight", type=float, default=0.5)
    parser.add_argument("--focal-weight", type=float, default=0.2)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("TRAINING WITH MIRANK LOSS", flush=True)
    print("=" * 70, flush=True)

    model_names, atom_feature_dim, global_feature_dim = _load_prediction_metadata(Path(args.predictions))
    drugs = [drug for drug in _load_drugs(Path(args.dataset)) if _has_site_labels(drug)]
    import random

    random.Random(args.seed).shuffle(drugs)
    n_train = int(len(drugs) * args.train_ratio)
    n_val = int(len(drugs) * args.val_ratio)
    train_drugs = drugs[:n_train]
    val_drugs = drugs[n_train : n_train + n_val]
    test_drugs = drugs[n_train + n_val :]
    print(f"Train={len(train_drugs)} Val={len(val_drugs)} Test={len(test_drugs)}", flush=True)

    train_dataset = MetaLearnerDataset(args.predictions, train_drugs)
    val_dataset = MetaLearnerDataset(args.predictions, val_drugs)
    model = MultiHeadMetaLearner(
        n_models=len(model_names),
        n_cyp=5,
        atom_feature_dim=atom_feature_dim,
        global_feature_dim=global_feature_dim,
        hidden_dim=args.hidden_dim,
    )
    if args.checkpoint:
        payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict") or payload
        current_state = model.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
        }
        model.load_state_dict(compatible_state, strict=False)
        print(f"Loaded warm-start checkpoint: {args.checkpoint}", flush=True)

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
        mirank_weight=args.mirank_weight,
        bce_weight=args.bce_weight,
        listmle_weight=args.listmle_weight,
        focal_weight=args.focal_weight,
        ranking_margin=args.ranking_margin,
        hard_negative_fraction=args.hard_negative_fraction,
    )
    trainer = MultiHeadTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=_resolve_device(args.device),
    )
    payload = trainer.train()
    print(f"Best validation Top-1: {payload['best_val_top1']:.3f}", flush=True)


if __name__ == "__main__":
    main()
