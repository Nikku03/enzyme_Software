from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import DataLoader, Dataset, Subset

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
    parser.add_argument("--max-samples", type=int, default=0, help="Optional limit on number of samples used from the dataset")
    parser.add_argument("--log-every", type=int, default=10, help="Print running metrics every N batches")
    parser.add_argument("--steps", type=int, default=8, help="Dynamics rollout steps")
    parser.add_argument("--dt", type=float, default=0.001, help="Dynamics step size")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpointed dynamics")
    parser.add_argument("--no-wsd", action="store_true", help="Disable the warmup-stable-decay scheduler")
    parser.add_argument("--wsd-decay-style", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--metrics-json", default=None, help="Optional path to save epoch metrics")
    parser.add_argument("--low-memory-train", action="store_true", help="Use low-memory training mode (skips full dynamics rollout)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (recommended for Colab/CPU)")
    parser.add_argument("--integration-resolution", type=int, default=16, help="Quantum grid resolution per axis (default 16 → 16^3=4096 pts)")
    parser.add_argument("--integration-chunk-size", type=int, default=1024, help="Chunk size for quantum grid integration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.dataset) == bool(args.sdf_dataset):
        raise ValueError("Provide exactly one of --dataset or --sdf-dataset")

    if args.sdf_dataset:
        dataset = ZaretzkiMetabolicDataset(
            args.sdf_dataset,
            max_molecules=max(int(args.max_samples), 0),
        )
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

    if args.max_samples > 0 and not args.sdf_dataset:
        limit = min(int(args.max_samples), len(dataset))
        dataset = Subset(dataset, range(limit))
        collate = geometric_collate_fn if args.sdf_dataset else collate_single
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    trainer = Metabolic_Causal_Trainer(
        dynamics_steps=args.steps,
        dynamics_dt=args.dt,
        checkpoint_dynamics=not args.no_checkpoint,
        enable_wsd_scheduler=not args.no_wsd,
        wsd_decay_style=args.wsd_decay_style,
        low_memory_train_mode=args.low_memory_train,
        enable_static_compile=not args.no_compile,
    )

    # Override quantum enforcer resolution if explicitly requested (important for Colab memory)
    if args.integration_resolution != 16 or args.integration_chunk_size != 1024:
        try:
            qe = trainer.model.module1.field_engine.quantum_enforcer
            qe.integration_resolution = max(int(args.integration_resolution), 4)
            qe.integration_chunk_size = max(int(args.integration_chunk_size), 16)
        except AttributeError:
            pass  # model layout differs — skip silently

    trainer.set_total_training_steps(args.epochs * max(len(loader), 1))
    trainer.configure_optimizers()

    history: List[Dict[str, float]] = []
    for epoch in range(args.epochs):
        reducer: Dict[str, List[float]] = {}
        trainer.train(True)
        total_batches = len(loader)
        for batch_idx, batch in enumerate(loader, start=1):
            metrics_t = trainer.training_step(batch)
            for key, value in metrics_t.items():
                if hasattr(value, "detach"):
                    reducer.setdefault(key, []).append(float(value.detach().cpu().item()))
                else:
                    reducer.setdefault(key, []).append(float(value))
            if args.log_every > 0 and (batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == total_batches):
                running = {key: sum(values) / max(len(values), 1) for key, values in reducer.items()}
                print(
                    f"epoch={epoch + 1} batch={batch_idx}/{total_batches} "
                    f"loss_total={running.get('loss_total', float('nan')):.6g} "
                    f"pred_rate={running.get('pred_rate', float('nan')):.6g} "
                    f"dag_loss={running.get('dag_causal_loss', float('nan')):.6g}",
                    flush=True,
                )
        metrics = {key: sum(values) / max(len(values), 1) for key, values in reducer.items()}
        history.append(metrics)
        print(f"epoch={epoch + 1} metrics={metrics}", flush=True)

    if args.metrics_json:
        out_path = Path(args.metrics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
