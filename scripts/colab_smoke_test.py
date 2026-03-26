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


def _normalize_profile_name(value: str) -> str:
    aliases = {
        "a100": "standard",
        "h100": "high_vram",
        "h100_sxm": "ultra_vram",
        "ultra": "ultra_vram",
        "standard": "standard",
        "high_vram": "high_vram",
        "ultra_vram": "ultra_vram",
    }
    return aliases.get(value.strip().lower(), "auto")


def _strict_profile_override_enabled() -> bool:
    raw = os.environ.get("NEXUS_COLAB_STRICT_PROFILE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _detect_gpu_profile(requested: str) -> str:
    normalized = _normalize_profile_name(requested)
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if (
            normalized == "high_vram"
            and total_gb >= 70.0
            and not _strict_profile_override_enabled()
        ):
            print(
                "Auto-promoting profile: high_vram -> ultra_vram "
                "(set NEXUS_COLAB_STRICT_PROFILE=1 to force high_vram)."
            )
            return "ultra_vram"
    if normalized in {"standard", "high_vram", "ultra_vram"}:
        return normalized
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_gb >= 70.0:
            return "ultra_vram"
        if total_gb >= 35.0:
            return "high_vram"
    return "standard"


def _profile_defaults(profile: str) -> Dict[str, float]:
    profiles: Dict[str, Dict[str, float]] = {
        "standard": {
            "max_molecules": 32,
            "integration_resolution": 8,
            "integration_chunk_size": 32,
            "scan_n_points": 8,
            "scan_radius": 1.0,
            "scan_query_chunk_size": 2,
        },
        "high_vram": {
            "max_molecules": 64,
            "integration_resolution": 10,
            "integration_chunk_size": 96,
            "scan_n_points": 12,
            "scan_radius": 1.0,
            "scan_query_chunk_size": 6,
        },
        "ultra_vram": {
            "max_molecules": 64,
            "integration_resolution": 12,
            "integration_chunk_size": 128,
            "scan_n_points": 16,
            "scan_radius": 1.25,
            "scan_query_chunk_size": 8,
        },
    }
    return profiles[profile]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Colab-friendly NEXUS smoke test")
    parser.add_argument("--sdf", default=DEFAULT_SDF, help="Path to an isoform SDF file")
    parser.add_argument("--gpu-profile", default=os.environ.get("NEXUS_COLAB_GPU_PROFILE", "auto"), choices=("auto", "standard", "high_vram", "ultra_vram", "a100", "h100", "h100_sxm", "ultra"), help="Auto-detect or force a Colab memory profile")
    parser.add_argument("--sample-index", type=int, default=-1, help="Dataset sample index; use -1 to auto-pick the smallest molecule")
    parser.add_argument("--steps", type=int, default=2, help="Dynamics rollout steps for the smoke test")
    parser.add_argument("--dt", type=float, default=0.001, help="Dynamics step size")
    parser.add_argument("--max-molecules", type=int, default=0, help="Cap SDF loading for faster notebook smoke runs; 0 uses the selected GPU profile")
    parser.add_argument("--forward-only", action="store_true", help="Skip optimizer/backward and run validation_step only")
    parser.add_argument("--allow-compile", action="store_true", help="Enable selective torch.compile; off by default for Colab safety")
    parser.add_argument("--no-compile", action="store_true", help="Disable selective torch.compile")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 hot path")
    parser.add_argument("--integration-resolution", type=int, default=0, help="Quantum normalization grid resolution; 0 uses the selected GPU profile")
    parser.add_argument("--integration-chunk-size", type=int, default=0, help="Quantum normalization chunk size; 0 uses the selected GPU profile")
    parser.add_argument("--scan-n-points", type=int, default=0, help="Reaction-volume shell points for the smoke test; 0 uses the selected GPU profile")
    parser.add_argument("--scan-radius", type=float, default=0.0, help="Reaction-volume scan radius for the smoke test; 0 uses the selected GPU profile")
    parser.add_argument("--scan-query-chunk-size", type=int, default=0, help="Chunk size for field queries during scan; 0 uses the selected GPU profile")
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
    if preferred_index < 0:
        best_index = min(range(len(dataset)), key=lambda idx: int(dataset.mols[idx].GetNumAtoms()))
        return dataset[best_index]
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
    gpu_profile = _detect_gpu_profile(args.gpu_profile)
    profile = _profile_defaults(gpu_profile)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    sdf_path = Path(args.sdf)
    if not sdf_path.exists():
        raise SystemExit(f"SDF not found: {sdf_path}")

    max_molecules = int(args.max_molecules) if int(args.max_molecules) > 0 else int(profile["max_molecules"])
    dataset = ZaretzkiMetabolicDataset(sdf_path, max_molecules=max(max_molecules, 1))
    item = _select_item(dataset, args.sample_index)
    batch = geometric_collate_fn([item])
    batch = _move_to_device(batch, device)

    # Colab tends to fragment memory under repeated compile/cudagraph allocations.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    full_physics = gpu_profile == "ultra_vram"
    trainer = Metabolic_Causal_Trainer(
        dynamics_steps=args.steps,
        dynamics_dt=args.dt,
        dynamics_summary_mode="full" if full_physics else "lite",
        checkpoint_dynamics=False,
        enable_static_compile=args.allow_compile and not args.no_compile,
        enable_bf16_hot_path=not args.no_bf16,
        enable_wsd_scheduler=False,
        low_memory_train_mode=not full_physics,
        low_memory_scan_gradients=(gpu_profile in {"high_vram", "ultra_vram"}),
        use_galore=False,
    ).to(device)

    quantum_enforcer = trainer.model.module1.field_engine.quantum_enforcer
    integration_resolution = int(args.integration_resolution) if int(args.integration_resolution) > 0 else int(profile["integration_resolution"])
    integration_chunk_size = int(args.integration_chunk_size) if int(args.integration_chunk_size) > 0 else int(profile["integration_chunk_size"])
    scan_n_points = int(args.scan_n_points) if int(args.scan_n_points) > 0 else int(profile["scan_n_points"])
    scan_radius = float(args.scan_radius) if float(args.scan_radius) > 0.0 else float(profile["scan_radius"])
    scan_query_chunk_size = int(args.scan_query_chunk_size) if int(args.scan_query_chunk_size) > 0 else int(profile["scan_query_chunk_size"])
    quantum_enforcer.integration_resolution = max(integration_resolution, 4)
    quantum_enforcer.integration_chunk_size = max(integration_chunk_size, 16)
    query_engine = trainer.model.module1.field_engine.query_engine
    query_engine.n_points = max(scan_n_points, 4)
    query_engine.radius = scan_radius
    query_engine.query_chunk_size = max(scan_query_chunk_size, 1)
    query_engine.shell_fractions = (0.5, 1.0)
    query_engine.refine_steps = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    if args.forward_only:
        # Low-memory smoke keeps train mode so forward_batch takes the reduced path.
        # Full-physics smoke uses eval mode and the real dynamics summary.
        if full_physics:
            trainer.eval()
        else:
            trainer.train()
        metrics = trainer.validation_step(batch)
    else:
        trainer.train()
        metrics = trainer.training_step(batch)

    scalars = _scalarize(metrics)
    print("\nNEXUS Colab Smoke Test")
    print(
        f"device={device}  gpu_profile={gpu_profile}  sdf={sdf_path.name}  "
        f"sample_index={args.sample_index}  forward_only={args.forward_only}"
    )
    print(
        "static_compile="
        f"{trainer.enable_static_compile}  bf16_hot_path={trainer.enable_bf16_hot_path}  "
        f"physics_mode={'full' if full_physics else 'lite'}  "
        f"integration_resolution={quantum_enforcer.integration_resolution}  "
        f"integration_chunk_size={quantum_enforcer.integration_chunk_size}  "
        f"scan_n_points={query_engine.n_points}  "
        f"scan_radius={query_engine.radius}  "
        f"scan_query_chunk_size={query_engine.query_chunk_size}"
    )
    for key in sorted(scalars):
        print(f"{key}={scalars[key]}")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0
        print(f"peak_cuda_memory_mb={peak_mb:.2f}")


if __name__ == "__main__":
    main()
