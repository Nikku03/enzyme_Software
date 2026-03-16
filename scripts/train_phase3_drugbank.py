from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.config import TrainingConfig
from enzyme_software.liquid_nn_v2.config_9cyp import IDX_TO_CYP, ModelConfig9CYP
from enzyme_software.liquid_nn_v2.data.cyp_classes import ALL_CYP_CLASSES
from enzyme_software.liquid_nn_v2.data.dataset_loader import create_dataloaders
from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
from enzyme_software.liquid_nn_v2.training.trainer import Trainer


def _resolve_device(name: str | None):
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_cyp_weights(dataset_path: str):
    payload = json.loads(Path(dataset_path).read_text())
    drugs = payload.get("drugs", payload)
    counts = Counter(str(d.get("primary_cyp")) for d in drugs if str(d.get("primary_cyp")) in ALL_CYP_CLASSES)
    total = sum(counts.values())
    if total == 0:
        return torch.ones(len(ALL_CYP_CLASSES), dtype=torch.float32)
    weights = [total / (len(ALL_CYP_CLASSES) * max(1, counts.get(cyp, 1))) for cyp in ALL_CYP_CLASSES]
    max_w = max(weights)
    weights = [w / max_w for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


def main():
    require_torch()
    parser = argparse.ArgumentParser(description="Phase 3 DrugBank training with 9 CYP classes")
    parser.add_argument("--dataset", default="data/training_dataset_final.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warm-start", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3: DrugBank Training (9 CYPs)")
    print("=" * 60)
    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    config = ModelConfig9CYP()
    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        cyp_classes=list(config.cyp_names),
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

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

    cyp_weights = compute_cyp_weights(args.dataset)
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

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = []
    best_val = -1.0
    best_state = None
    for epoch in range(args.epochs):
        train_stats = trainer.train_loader_epoch(train_loader)
        val_metrics = trainer.evaluate_loader(val_loader)
        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_metrics})
        val_top1 = float(val_metrics.get("site_top1_acc", 0.0))
        if val_top1 > best_val:
            best_val = val_top1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d} | loss={train_stats.get('total_loss', float('nan')):.4f} | "
                f"site_top1={val_metrics.get('site_top1_acc', 0.0):.3f} | "
                f"site_top3={val_metrics.get('site_top3_acc', 0.0):.3f} | "
                f"cyp_acc={val_metrics.get('accuracy', 0.0):.3f} | "
                f"cyp_f1={val_metrics.get('f1_macro', 0.0):.3f}"
            )

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
    checkpoint_path = f"checkpoints/phase3_drugbank_9cyp_{timestamp}.pt"
    report_path = f"checkpoints/phase3_report_{timestamp}.json"

    checkpoint_data = {
        "phase": 3,
        "dataset": "DrugBank",
        "num_drugs": len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
        "num_cyp_classes": len(config.cyp_names),
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "test_metrics": test_metrics,
        "tau_analysis": tau_analysis,
        "gate_analysis": gate_analysis,
        "history": history,
    }
    torch.save(checkpoint_data, checkpoint_path)
    with open(report_path, "w") as f:
        json.dump(
            {
                "phase": 3,
                "dataset": "DrugBank",
                "num_drugs": checkpoint_data["num_drugs"],
                "num_cyp_classes": len(config.cyp_names),
                "epochs": len(history),
                "test_metrics": test_metrics,
                "tau_analysis": tau_analysis,
                "gate_analysis": gate_analysis,
            },
            f,
            indent=2,
        )
    print(f"\nSaved checkpoint: {checkpoint_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
