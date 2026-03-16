from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.config import TrainingConfig
from enzyme_software.liquid_nn_v2.config_9cyp import IDX_TO_CYP, ModelConfig9CYP
from enzyme_software.liquid_nn_v2.data.cyp_classes import ALL_CYP_CLASSES
from enzyme_software.liquid_nn_v2.data.dataset_loader import create_dataloaders_from_drugs
from enzyme_software.liquid_nn_v2.data.oversampling import oversample_rare_classes
from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
from enzyme_software.liquid_nn_v2.training.loss import compute_cyp_weights
from enzyme_software.liquid_nn_v2.training.trainer import Trainer


def _resolve_device(name: str | None):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_drugs(dataset_path: str):
    payload = json.loads(Path(dataset_path).read_text())
    return list(payload.get("drugs", payload))


def _split_drugs(drugs, train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42):
    rng = random.Random(seed)
    items = list(drugs)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def _count_cyps(drugs):
    return Counter(str(d.get("primary_cyp") or d.get("cyp") or "") for d in drugs if str(d.get("primary_cyp") or d.get("cyp") or "") in ALL_CYP_CLASSES)


def main():
    require_torch()
    parser = argparse.ArgumentParser(description="Balanced Phase 3 DrugBank training with 9 CYP classes")
    parser.add_argument("--dataset", default="data/training_dataset_final.json")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--target-per-class", type=int, default=80)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3 BALANCED: DrugBank (9 CYPs)")
    print("=" * 60)
    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    drugs = _load_drugs(args.dataset)
    print(f"Total drugs: {len(drugs)}")

    train_drugs, val_drugs, test_drugs = _split_drugs(drugs, seed=args.seed)
    train_counts = _count_cyps(train_drugs)

    print("\nOriginal train CYP distribution:")
    for cyp in ALL_CYP_CLASSES:
        print(f"  {cyp}: {train_counts.get(cyp, 0)}")

    train_drugs_balanced = oversample_rare_classes(
        train_drugs,
        target_per_class=args.target_per_class,
        seed=args.seed,
    )
    balanced_counts = _count_cyps(train_drugs_balanced)

    print("\nOversampled train CYP distribution:")
    for cyp in ALL_CYP_CLASSES:
        print(f"  {cyp}: {balanced_counts.get(cyp, 0)}")

    print(f"\nSplit sizes: train={len(train_drugs_balanced)} val={len(val_drugs)} test={len(test_drugs)}")

    config = ModelConfig9CYP()
    train_loader, val_loader, test_loader = create_dataloaders_from_drugs(
        train_drugs_balanced,
        val_drugs,
        test_drugs,
        batch_size=args.batch_size,
        cyp_classes=list(config.cyp_names),
    )

    model = LiquidMetabolismNetV2(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    if args.warm_start and Path(args.warm_start).exists():
        print("\nLoading warm-start weights...")
        checkpoint = torch.load(args.warm_start, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        compatible = {name: param for name, param in state_dict.items() if "cyp_head" not in name}
        model.load_state_dict(compatible, strict=False)
        print("Loaded compatible weights; CYP head reinitialized for 9 classes")

    cyp_weights = compute_cyp_weights(dict(train_counts), max_weight=10.0)
    print("\nCYP class weights:")
    for idx, cyp in enumerate(ALL_CYP_CLASSES):
        print(f"  {cyp}: {float(cyp_weights[idx]):.2f}")

    trainer = Trainer(
        model=model,
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        device=device,
        cyp_class_weights=cyp_weights,
    )

    history = []
    best_metric = -1.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    for epoch in range(args.epochs):
        train_stats = trainer.train_loader_epoch(train_loader)
        val_metrics = trainer.evaluate_loader(val_loader)
        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_metrics})

        val_cyp_acc = float(val_metrics.get("accuracy", 0.0))
        val_cyp_f1 = float(val_metrics.get("f1_macro", 0.0))
        selection_metric = 0.7 * val_cyp_f1 + 0.3 * val_cyp_acc

        if selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d} | loss={train_stats.get('total_loss', float('nan')):.4f} | "
                f"site_top1={val_metrics.get('site_top1_acc', 0.0):.3f} | "
                f"site_top3={val_metrics.get('site_top3_acc', 0.0):.3f} | "
                f"cyp_acc={val_cyp_acc:.3f} | cyp_f1={val_cyp_f1:.3f} | "
                f"best={best_metric:.3f}@{best_epoch}"
            )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    test_metrics = trainer.evaluate_loader(test_loader)

    print("\nOverall Metrics:")
    print(f"  Site Top-1: {test_metrics.get('site_top1_acc', 0):.1%}")
    print(f"  Site Top-2: {test_metrics.get('site_top2_acc', 0):.1%}")
    print(f"  Site Top-3: {test_metrics.get('site_top3_acc', 0):.1%}")
    print(f"  Site F1:    {test_metrics.get('site_f1', 0):.3f}")
    print(f"  CYP Acc:    {test_metrics.get('accuracy', 0):.1%}")
    print(f"  CYP F1:     {test_metrics.get('f1_macro', 0):.3f}")

    print("\nPer-CYP F1 Scores:")
    for i, f1 in enumerate(test_metrics.get("f1_per_class", [0] * len(config.cyp_names))):
        print(f"  {IDX_TO_CYP.get(i, f'CYP{i}')}: {f1:.3f}")

    tau_analysis = trainer.analyze_tau(train_loader)
    gate_analysis = trainer.analyze_gates(train_loader)
    print(f"\nτ-BDE Correlation: {tau_analysis.get('tau_bde_correlation', 0.0):.3f}")
    print(f"Gate Mean: {gate_analysis.get('gate_mean', 0.0):.3f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("checkpoints").mkdir(exist_ok=True)
    checkpoint_path = f"checkpoints/phase3_balanced_9cyp_{timestamp}.pt"
    report_path = f"checkpoints/phase3_balanced_report_{timestamp}.json"

    checkpoint_data = {
        "phase": "3_balanced",
        "dataset": Path(args.dataset).name,
        "num_drugs": len(drugs),
        "num_train_oversampled": len(train_drugs_balanced),
        "num_cyp_classes": len(config.cyp_names),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "test_metrics": test_metrics,
        "tau_analysis": tau_analysis,
        "gate_analysis": gate_analysis,
        "history": history,
        "train_cyp_counts": dict(train_counts),
        "train_cyp_counts_oversampled": dict(balanced_counts),
    }
    torch.save(checkpoint_data, checkpoint_path)
    with open(report_path, "w") as f:
        json.dump(
            {
                "phase": "3_balanced",
                "dataset": Path(args.dataset).name,
                "num_drugs": len(drugs),
                "num_train_oversampled": len(train_drugs_balanced),
                "num_cyp_classes": len(config.cyp_names),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "test_metrics": test_metrics,
                "tau_analysis": tau_analysis,
                "gate_analysis": gate_analysis,
                "train_cyp_counts": dict(train_counts),
                "train_cyp_counts_oversampled": dict(balanced_counts),
            },
            f,
            indent=2,
        )
    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
