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
from enzyme_software.liquid_nn_v2.model.pairwise_site_tournament import PairwiseSiteTournamentModel
from enzyme_software.liquid_nn_v2.training.pairwise_site_tournament_trainer import PairwiseSiteTournamentTrainer


def _load_checkpoint(checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_dim = int(payload.get("feature_dim", 0))
    hidden_dim = int(payload.get("hidden_dim", 128))
    model_state = dict(payload.get("model_state_dict") or {})
    extra = dict(payload.get("extra") or {})
    if feature_dim <= 0:
        raise ValueError(f"Invalid feature_dim in checkpoint: {checkpoint_path}")
    return feature_dim, hidden_dim, model_state, extra


def _evaluate_split(trainer: PairwiseSiteTournamentTrainer, dataset: CandidateSetDataset, *, batch_size: int):
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=collate_candidate_sets, num_workers=0)
    return trainer.evaluate(loader, split_summary=dataset.summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved pairwise site tournament model on a candidate cache")
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shortlist-k", type=int, default=0)
    parser.add_argument("--compare-top-n", type=int, default=-1)
    args = parser.parse_args()

    cache_path = Path(args.candidate_cache)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    feature_dim, hidden_dim, model_state, extra = _load_checkpoint(checkpoint_path)
    shortlist_k = int(args.shortlist_k) if int(args.shortlist_k) > 0 else int(extra.get("shortlist_k", 6))
    compare_top_n = int(args.compare_top_n) if int(args.compare_top_n) >= 0 else int(extra.get("compare_top_n", 0))
    model = PairwiseSiteTournamentModel(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=float(extra.get("dropout", 0.10)),
        blend_weight=float(extra.get("blend_weight", 0.75)),
    )
    model.load_state_dict(model_state)
    trainer = PairwiseSiteTournamentTrainer(
        model=model,
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        pair_loss_weight=float(extra.get("pair_loss_weight", 1.0)),
        site_loss_weight=float(extra.get("site_loss_weight", 0.25)),
        antisymmetry_weight=float(extra.get("antisymmetry_weight", 0.10)),
        compare_top_n=compare_top_n,
        shortlist_k=shortlist_k,
    )

    train_ds = CandidateSetDataset(cache_path, split="train")
    val_ds = CandidateSetDataset(cache_path, split="val")
    test_ds = CandidateSetDataset(cache_path, split="test")

    report = {
        "status": "completed",
        "candidate_cache": str(cache_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_extra": extra,
        "shortlist_k": shortlist_k,
        "compare_top_n": compare_top_n,
        "split_summaries": {
            "train": train_ds.summary,
            "val": val_ds.summary,
            "test": test_ds.summary,
        },
        "train_metrics": _evaluate_split(trainer, train_ds, batch_size=int(args.batch_size)),
        "val_metrics": _evaluate_split(trainer, val_ds, batch_size=int(args.batch_size)),
        "test_metrics": _evaluate_split(trainer, test_ds, batch_size=int(args.batch_size)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved={output_path}", flush=True)


if __name__ == "__main__":
    main()
