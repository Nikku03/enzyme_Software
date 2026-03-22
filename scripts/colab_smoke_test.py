from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nexus.data import ZaretzkiMetabolicDataset, geometric_collate_fn
from nexus.training.causal_trainer import Metabolic_Causal_Trainer


DEFAULT_SDF = "data/ATTNSOM/cyp_dataset/3A4.sdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Colab-friendly NEXUS smoke test")
    parser.add_argument("--sdf", default=DEFAULT_SDF, help="Path to an isoform SDF file")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--steps", type=int, default=2, help="Dynamics rollout steps for the smoke test")
    parser.add_argument("--dt", type=float, default=0.001, help="Dynamics step size")
    parser.add_argument("--forward-only", action="store_true", help="Skip optimizer/backward and run validation_step only")
    parser.add_argument("--allow-compile", action="store_true", help="Enable selective torch.compile; off by default for Colab safety")
    parser.add_argument("--no-compile", action="store_true", help="Disable selective torch.compile")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 hot path")
    parser.add_argument("--integration-resolution", type=int, default=8, help="Quantum normalization grid resolution; lower is safer for smoke tests")
    parser.add_argument("--integration-chunk-size", type=int, default=128, help="Quantum normalization chunk size; lower reduces peak memory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=("cpu", "cuda"))
    return parser.parse_args()


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: _move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(value, device) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_device(value, device) for value in obj)
    return obj


def _select_item(dataset: ZaretzkiMetabolicDataset, preferred_index: int) -> Dict[str, Any]:
    if 0 <= preferred_index < len(dataset):
        return dataset[preferred_index]
    raise IndexError(f"sample-index {preferred_index} out of range for dataset of size {len(dataset)}")


def _scalarize(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    sdf_path = Path(args.sdf)
    if not sdf_path.exists():
        raise SystemExit(f"SDF not found: {sdf_path}")

    dataset = ZaretzkiMetabolicDataset(sdf_path)
    item = _select_item(dataset, args.sample_index)
    batch = geometric_collate_fn([item])
    batch = _move_to_device(batch, device)

    # Colab tends to fragment memory under repeated compile/cudagraph allocations.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    trainer = Metabolic_Causal_Trainer(
        dynamics_steps=args.steps,
        dynamics_dt=args.dt,
        checkpoint_dynamics=False,
        enable_static_compile=args.allow_compile and not args.no_compile,
        enable_bf16_hot_path=not args.no_bf16,
        enable_wsd_scheduler=False,
    ).to(device)

    quantum_enforcer = trainer.model.module1.field_engine.quantum_enforcer
    quantum_enforcer.integration_resolution = max(int(args.integration_resolution), 4)
    quantum_enforcer.integration_chunk_size = max(int(args.integration_chunk_size), 16)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    if args.forward_only:
        trainer.eval()
        with torch.no_grad():
            metrics = trainer.validation_step(batch)
    else:
        trainer.train()
        metrics = trainer.training_step(batch)

    scalars = _scalarize(metrics)
    print("\nNEXUS Colab Smoke Test")
    print(f"device={device}  sdf={sdf_path.name}  sample_index={args.sample_index}  forward_only={args.forward_only}")
    print(
        "static_compile="
        f"{trainer.enable_static_compile}  bf16_hot_path={trainer.enable_bf16_hot_path}  "
        f"integration_resolution={quantum_enforcer.integration_resolution}  "
        f"integration_chunk_size={quantum_enforcer.integration_chunk_size}"
    )
    for key in sorted(scalars):
        print(f"{key}={scalars[key]}")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
        print(f"peak_cuda_memory_mb={peak_mb:.2f}")


if __name__ == "__main__":
    main()
