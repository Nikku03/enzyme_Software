import argparse
import json
import random
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
from enzyme_software.liquid_nn_v2.model.pairwise_site_tournament import PairwiseSiteTournamentModel
from enzyme_software.liquid_nn_v2.training.pairwise_site_tournament_trainer import PairwiseSiteTournamentTrainer

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _set_seed(seed: int) -> torch.Generator:
    random.seed(int(seed))
    if np is not None:
        np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an offline pairwise site tournament proposer")
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--blend-weight", type=float, default=0.75)
    parser.add_argument("--pair-loss-weight", type=float, default=1.0)
    parser.add_argument("--site-loss-weight", type=float, default=0.25)
    parser.add_argument("--antisymmetry-weight", type=float, default=0.10)
    parser.add_argument("--compare-top-n", type=int, default=0)
    parser.add_argument("--shortlist-k", type=int, default=6)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_generator = _set_seed(int(args.seed))
    cache_path = Path(args.candidate_cache)
    train_ds = CandidateSetDataset(cache_path, split="train")
    val_ds = CandidateSetDataset(cache_path, split="val")
    test_ds = CandidateSetDataset(cache_path, split="test")
    feature_dim = int((train_ds.meta or {}).get("feature_dim", 0))
    if feature_dim <= 0:
        raise ValueError(f"Invalid feature_dim in {cache_path}")

    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_candidate_sets, num_workers=0, generator=train_generator)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_candidate_sets, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_candidate_sets, num_workers=0)

    model = PairwiseSiteTournamentModel(
        feature_dim=feature_dim,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        blend_weight=float(args.blend_weight),
    )
    trainer = PairwiseSiteTournamentTrainer(
        model=model,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        pair_loss_weight=float(args.pair_loss_weight),
        site_loss_weight=float(args.site_loss_weight),
        antisymmetry_weight=float(args.antisymmetry_weight),
        compare_top_n=int(args.compare_top_n),
        shortlist_k=int(args.shortlist_k),
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
        proposal_recall = float(val_metrics.get("proposal_molecule_recall_at_k", 0.0))
        end_to_end = float(val_metrics.get("end_to_end_top1", 0.0))
        top3 = float(val_metrics.get("tournament_top3_acc_given_cache", 0.0))
        monitor = (0.50 * proposal_recall) + (0.30 * end_to_end) + (0.20 * top3)
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
            f"epoch={epoch} train_loss={train_metrics.get('tournament_total_loss', 0.0):.4f} "
            f"val_recall@{int(args.shortlist_k)}={proposal_recall:.4f} "
            f"val_top1={val_metrics.get('tournament_top1_acc_given_cache', 0.0):.4f} "
            f"val_top3={top3:.4f}",
            flush=True,
        )
        if patience >= int(args.early_stopping_patience):
            break

    if best_state is None:
        raise RuntimeError("No best state recorded during tournament training")
    model.load_state_dict(best_state["model_state_dict"])
    train_metrics = trainer.evaluate(train_loader, split_summary=train_ds.summary)
    val_metrics = trainer.evaluate(val_loader, split_summary=val_ds.summary)
    test_metrics = trainer.evaluate(test_loader, split_summary=test_ds.summary)
    trainer.save_checkpoint(
        output_dir / "pairwise_site_tournament_best.pt",
        feature_dim=feature_dim,
        hidden_dim=int(args.hidden_dim),
        extra={
            "candidate_cache": str(cache_path),
            "best_epoch": int(best_state["epoch"]),
            "seed": int(args.seed),
            "dropout": float(args.dropout),
            "blend_weight": float(args.blend_weight),
            "pair_loss_weight": float(args.pair_loss_weight),
            "site_loss_weight": float(args.site_loss_weight),
            "antisymmetry_weight": float(args.antisymmetry_weight),
            "compare_top_n": int(args.compare_top_n),
            "shortlist_k": int(args.shortlist_k),
            "history": history,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
    )
    report = {
        "status": "completed",
        "candidate_cache": str(cache_path),
        "seed": int(args.seed),
        "feature_dim": int(feature_dim),
        "best_epoch": int(best_state["epoch"]),
        "blend_weight": float(args.blend_weight),
        "compare_top_n": int(args.compare_top_n),
        "shortlist_k": int(args.shortlist_k),
        "history": history,
        "split_summaries": {
            "train": train_ds.summary,
            "val": val_ds.summary,
            "test": test_ds.summary,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    report_path = output_dir / "pairwise_site_tournament_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved={report_path}", flush=True)


if __name__ == "__main__":
    main()
