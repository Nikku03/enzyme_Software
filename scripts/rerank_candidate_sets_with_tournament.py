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


def _rerank_split(trainer: PairwiseSiteTournamentTrainer, dataset: CandidateSetDataset, *, batch_size: int, shortlist_k: int):
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=collate_candidate_sets, num_workers=0)
    return trainer.rerank_split(
        loader,
        shortlist_k=int(shortlist_k),
        total_molecules=int((dataset.summary or {}).get("total_molecules", len(dataset))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerank a candidate cache with a saved pairwise site tournament model")
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shortlist-k", type=int, required=True)
    parser.add_argument("--compare-top-n", type=int, default=-1)
    args = parser.parse_args()

    cache_path = Path(args.candidate_cache)
    checkpoint_path = Path(args.checkpoint)
    output_cache = Path(args.output_cache)
    output_report = Path(args.output_report)

    feature_dim, hidden_dim, model_state, extra = _load_checkpoint(checkpoint_path)
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
        shortlist_k=int(args.shortlist_k),
    )

    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    train_ds = CandidateSetDataset(cache_path, split="train")
    val_ds = CandidateSetDataset(cache_path, split="val")
    test_ds = CandidateSetDataset(cache_path, split="test")

    split_payload = {
        "train": _rerank_split(trainer, train_ds, batch_size=int(args.batch_size), shortlist_k=int(args.shortlist_k)),
        "val": _rerank_split(trainer, val_ds, batch_size=int(args.batch_size), shortlist_k=int(args.shortlist_k)),
        "test": _rerank_split(trainer, test_ds, batch_size=int(args.batch_size), shortlist_k=int(args.shortlist_k)),
    }
    feature_dim = 0
    for split_name in ("train", "val", "test"):
        feature_dim = max(feature_dim, int((split_payload[split_name]["summary"] or {}).get("feature_dim", 0)))
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": {
                **dict(payload.get("meta") or {}),
                "tournament_checkpoint": str(checkpoint_path),
                "shortlist_k": int(args.shortlist_k),
                "compare_top_n": int(compare_top_n),
                "feature_dim": int(feature_dim),
            },
            "splits": split_payload,
        },
        output_cache,
    )
    report = {
        "status": "completed",
        "candidate_cache": str(cache_path),
        "checkpoint": str(checkpoint_path),
        "output_cache": str(output_cache),
        "shortlist_k": int(args.shortlist_k),
        "compare_top_n": int(compare_top_n),
        "split_summaries": {name: split_payload[name]["summary"] for name in ("train", "val", "test")},
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved_cache={output_cache}", flush=True)
    print(f"saved_report={output_report}", flush=True)


if __name__ == "__main__":
    main()
