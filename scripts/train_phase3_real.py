from __future__ import annotations

import argparse
import json
from pathlib import Path

from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.config import ModelConfig, TrainingConfig
from enzyme_software.liquid_nn_v2.data.dataset_loader import create_dataloaders
from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
from enzyme_software.liquid_nn_v2.training.trainer import Trainer


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


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Train LNN on real CYP-labeled dataset")
    parser.add_argument("--dataset", default="data/training_dataset_real_labeled.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--structure-sdf", default=None)
    parser.add_argument("--warm-start", default=None)
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    print("=" * 60)
    print("PHASE 3: Training on Real CYP Dataset")
    print("=" * 60)
    print(f"dataset={dataset_path}")
    print(f"device={device}")

    train_loader, val_loader, test_loader = create_dataloaders(
        str(dataset_path),
        batch_size=args.batch_size,
        structure_sdf=args.structure_sdf,
    )

    model = LiquidMetabolismNetV2(ModelConfig())
    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print(f"warm_start={args.warm_start}")

    trainer = Trainer(
        model=model,
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        device=device,
    )

    history = []
    for epoch in range(args.epochs):
        train_stats = trainer.train_loader_epoch(train_loader)
        val_metrics = trainer.evaluate_loader(val_loader)
        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_metrics})
        print(
            f"epoch={epoch + 1} loss={train_stats.get('total_loss', float('nan')):.4f} "
            f"top1={val_metrics.get('site_top1_acc', 0.0):.3f} top3={val_metrics.get('site_top3_acc', 0.0):.3f} "
            f"cyp_acc={val_metrics.get('accuracy', 0.0):.3f}"
        )

    test_metrics = trainer.evaluate_loader(test_loader)
    print(json.dumps({"test_metrics": test_metrics}, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dataset_path.stem
    ckpt_path = output_dir / f"{stem}_phase3.pt"
    report_path = output_dir / f"{stem}_phase3_report.json"
    torch.save({"model_state_dict": _initialized_state_dict(trainer.model), "test_metrics": test_metrics}, ckpt_path)
    report_path.write_text(json.dumps({"dataset": str(dataset_path), "epochs": args.epochs, "history": history, "test_metrics": test_metrics}, indent=2))
    print(f"saved_checkpoint={ckpt_path}")
    print(f"saved_report={report_path}")


if __name__ == "__main__":
    main()
