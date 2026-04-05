import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torch.utils.data import DataLoader

from enzyme_software.liquid_nn_v2._compat import torch
from enzyme_software.liquid_nn_v2.data.candidate_set_dataset import CandidateSetDataset, collate_candidate_sets
from enzyme_software.liquid_nn_v2.model.candidate_winner_model import CandidateWinnerModel
from enzyme_software.liquid_nn_v2.training.candidate_winner_trainer import CandidateWinnerTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the offline Phase 4 candidate winner model")
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--margin-value", type=float, default=0.30)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    args = parser.parse_args()

    cache_path = Path(args.candidate_cache)
    train_ds = CandidateSetDataset(cache_path, split="train")
    val_ds = CandidateSetDataset(cache_path, split="val")
    test_ds = CandidateSetDataset(cache_path, split="test")
    feature_dim = int((train_ds.meta or {}).get("feature_dim", 0))
    if feature_dim <= 0:
        raise ValueError(f"Invalid feature_dim in {cache_path}")

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_candidate_sets, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_candidate_sets, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_candidate_sets, num_workers=0)

    model = CandidateWinnerModel(feature_dim=feature_dim, hidden_dim=int(args.hidden_dim), dropout=float(args.dropout))
    trainer = CandidateWinnerTrainer(
        model=model,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        margin_weight=float(args.margin_weight),
        margin_value=float(args.margin_value),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_val = float("-inf")
    patience = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader, split_summary=val_ds.summary)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        winner_acc = float(val_metrics.get("winner_acc_given_proposal", 0.0))
        end_to_end = float(val_metrics.get("end_to_end_top1", 0.0))
        monitor = 0.5 * (winner_acc + end_to_end)
        if monitor > best_val:
            best_val = monitor
            patience = 0
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }
        else:
            patience += 1
        print(
            f"epoch={epoch} train_loss={train_metrics.get('candidate_set_total_loss', 0.0):.4f} "
            f"val_winner_acc={winner_acc:.4f} val_end_to_end={end_to_end:.4f}",
            flush=True,
        )
        if patience >= int(args.early_stopping_patience):
            break

    if best_state is None:
        raise RuntimeError("No best state recorded during Phase 4 training")
    model.load_state_dict(best_state["model_state_dict"])
    test_metrics = trainer.evaluate(test_loader, split_summary=test_ds.summary)
    trainer.save_checkpoint(
        output_dir / "phase4_candidate_winner_best.pt",
        feature_dim=feature_dim,
        hidden_dim=int(args.hidden_dim),
        extra={
            "candidate_cache": str(cache_path),
            "best_epoch": int(best_state["epoch"]),
            "history": history,
            "test_metrics": test_metrics,
        },
    )
    report = {
        "status": "completed",
        "candidate_cache": str(cache_path),
        "feature_dim": int(feature_dim),
        "best_epoch": int(best_state["epoch"]),
        "history": history,
        "split_summaries": {
            "train": train_ds.summary,
            "val": val_ds.summary,
            "test": test_ds.summary,
        },
        "test_metrics": test_metrics,
    }
    report_path = output_dir / "phase4_candidate_winner_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved={report_path}", flush=True)


if __name__ == "__main__":
    main()
