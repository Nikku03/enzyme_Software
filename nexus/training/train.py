from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import DataLoader, Dataset

from nexus.data.metabolic_dataset import ZaretzkiMetabolicDataset, geometric_collate_fn
from nexus.training.causal_trainer import Metabolic_Causal_Trainer, load_compound_records


class NEXUSJSONDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return dict(self.records[index])


def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("NEXUS causal training currently expects batch_size=1")
    return batch[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NEXUS causal dynamics stack")
    parser.add_argument("--dataset", default=None, help="JSON dataset path")
    parser.add_argument("--sdf-dataset", default=None, help="Optional Zaretzki-style SDF dataset path")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8, help="Dynamics rollout steps")
    parser.add_argument("--dt", type=float, default=0.001, help="Dynamics step size")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointed dynamics")
    parser.add_argument("--no-wsd", action="store_true", help="Disable the warmup-stable-decay scheduler")
    parser.add_argument("--wsd-decay-style", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--metrics-json", default=None, help="Optional path to save epoch metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.dataset) == bool(args.sdf_dataset):
        raise ValueError("Provide exactly one of --dataset or --sdf-dataset")

    if args.sdf_dataset:
        dataset = ZaretzkiMetabolicDataset(args.sdf_dataset)
        if args.batch_size != 1:
            raise ValueError(
                "The current NEXUS causal trainer still expects single-sample batches. "
                "Use --batch-size 1 with --sdf-dataset until the trainer is generalized."
            )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=geometric_collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        assert args.dataset is not None
        records = load_compound_records(args.dataset)
        dataset = NEXUSJSONDataset(records)
        if args.batch_size != 1:
            raise ValueError("JSON training currently expects --batch-size 1")
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_single,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    trainer = Metabolic_Causal_Trainer(
        dynamics_steps=args.steps,
        dynamics_dt=args.dt,
        checkpoint_dynamics=not args.no_checkpoint,
        enable_wsd_scheduler=not args.no_wsd,
        wsd_decay_style=args.wsd_decay_style,
    )
    trainer.set_total_training_steps(args.epochs * max(len(loader), 1))
    trainer.configure_optimizers()

    history: List[Dict[str, float]] = []
    for epoch in range(args.epochs):
        metrics = trainer.fit_epoch(loader, train=True)
        history.append(metrics)
        print(f"epoch={epoch + 1} metrics={metrics}", flush=True)

    if args.metrics_json:
        out_path = Path(args.metrics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
