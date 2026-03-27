"""
Dedicated Colab entrypoint for offline CYP3A4 physics caching.

Run from a Colab cell with:

    exec(open("/content/enzyme_Software/scripts/colab_precompute_cyp3a4_physics_cache.py").read())

This computes per-molecule dynamics targets once, stores them in a `.pt`
cache, and lets training reuse those targets later via
`NEXUS_COLAB_PHYSICS_CACHE_MODE=cached|hybrid`.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*sdp_kernel.*",
)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

REPO_DIR = Path("/content/enzyme_Software")
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))


def _setdefault_env(name: str, value: str) -> None:
    if not os.environ.get(name):
        os.environ[name] = value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_shells(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            return default
    return tuple(values) if values else default


def _env_str(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


def _normalize_profile_name(value: str) -> str:
    aliases = {
        "a100": "standard",
        "h100": "high_vram",
        "h100_sxm": "ultra_vram",
        "ultra": "ultra_vram",
        "standard": "standard",
        "high_vram": "high_vram",
        "ultra_vram": "ultra_vram",
        "l4": "l4",
    }
    return aliases.get(value.strip().lower(), "auto")


def _detect_gpu_profile() -> str:
    env_raw = os.environ.get("NEXUS_COLAB_GPU_PROFILE", "auto").strip().lower()
    normalized = _normalize_profile_name(env_raw)
    valid_profiles = {"standard", "l4", "high_vram", "ultra_vram"}
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = float(props.total_memory) / 1024**3
            print(f"GPU : {props.name}  |  total memory : {total_gb:.1f} GB")
            if normalized in valid_profiles:
                return normalized
            if total_gb >= 70.0:
                return "ultra_vram"
            if total_gb >= 35.0:
                return "high_vram"
            if total_gb >= 20.0:
                return "l4"
        if normalized in valid_profiles:
            return normalized
    except Exception as exc:
        fallback = normalized if normalized in valid_profiles else "standard"
        print(f"GPU profile detection fallback: {type(exc).__name__}: {exc} -> {fallback}")
        return fallback
    return "standard"


def _discover_isoform_sdfs(repo_dir: Path) -> list[Path]:
    search_dirs = [
        repo_dir / "data/ATTNSOM/cyp_dataset",
        repo_dir / "CYP_DBs",
        repo_dir / "data/ATTNSOM",
    ]
    found: dict[str, Path] = {}
    for directory in search_dirs:
        if not directory.exists():
            continue
        for sdf in sorted(directory.glob("*.sdf")):
            if sdf.stem.upper() == "HLM":
                continue
            found.setdefault(sdf.stem.upper(), sdf)
    if not found:
        searched = ", ".join(str(p) for p in search_dirs)
        raise FileNotFoundError(
            "No CYP isoform SDF files were found for cache precompute. "
            f"Searched: {searched}."
        )
    return [found[key] for key in sorted(found)]


def _resolve_target_isoform_path(target_isoform: str, candidates: list[Path]) -> Path:
    token = target_isoform.strip().removesuffix(".sdf")
    by_stem = {p.stem.upper(): p for p in candidates}
    resolved = by_stem.get(token.upper())
    if resolved is None:
        valid = ", ".join(sorted(p.stem for p in candidates))
        raise FileNotFoundError(
            f"NEXUS_COLAB_TARGET_ISOFORM={target_isoform!r} did not match any available SDF. "
            f"Valid isoforms: {valid}"
        )
    return resolved


def _ensure_colab_nexus_assets() -> None:
    target_sdf = REPO_DIR / "data" / "ATTNSOM" / "cyp_dataset" / "3A4.sdf"
    setup_script = REPO_DIR / "scripts" / "setup_colab_nexus.sh"
    if target_sdf.exists():
        return
    if not setup_script.exists():
        raise FileNotFoundError(f"Missing setup bootstrap script: {setup_script}")
    print("ATTNSOM CYP SDF assets not found; running Colab bootstrap...")
    subprocess.run(["bash", str(setup_script), str(REPO_DIR)], check=True)


PRESETS: dict[str, dict[str, str]] = {
    "balanced": {
        "NEXUS_COLAB_GPU_PROFILE": "ultra_vram",
        "NEXUS_COLAB_TARGET_ISOFORM": "3A4",
        "NEXUS_COLAB_MAX_SAMPLES": "64",
        "NEXUS_COLAB_DYNAMICS_STEPS": "1",
        "NEXUS_COLAB_INTEGRATION_RESOLUTION": "10",
        "NEXUS_COLAB_INTEGRATION_CHUNK": "128",
        "NEXUS_COLAB_SCAN_N_POINTS": "12",
        "NEXUS_COLAB_SCAN_RADIUS": "1.20",
        "NEXUS_COLAB_SCAN_CHUNK": "6",
        "NEXUS_COLAB_SCAN_SHELLS": "0.40,0.70,1.00",
        "NEXUS_COLAB_SCAN_REFINE_STEPS": "0",
        "NEXUS_COLAB_NAV_OPT_STEPS": "1",
        "NEXUS_COLAB_NAV_CANDIDATES": "2",
    },
    "full_3a4": {
        "NEXUS_COLAB_GPU_PROFILE": "ultra_vram",
        "NEXUS_COLAB_TARGET_ISOFORM": "3A4",
        "NEXUS_COLAB_MAX_SAMPLES": "0",
        "NEXUS_COLAB_DYNAMICS_STEPS": "1",
        "NEXUS_COLAB_INTEGRATION_RESOLUTION": "10",
        "NEXUS_COLAB_INTEGRATION_CHUNK": "160",
        "NEXUS_COLAB_SCAN_N_POINTS": "16",
        "NEXUS_COLAB_SCAN_RADIUS": "1.35",
        "NEXUS_COLAB_SCAN_CHUNK": "8",
        "NEXUS_COLAB_SCAN_SHELLS": "0.40,0.65,0.85,1.00",
        "NEXUS_COLAB_SCAN_REFINE_STEPS": "1",
        "NEXUS_COLAB_NAV_OPT_STEPS": "2",
        "NEXUS_COLAB_NAV_CANDIDATES": "3",
    },
}


def _save_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    preset = _env_str("NEXUS_COLAB_CACHE_PRESET", "balanced").lower() or "balanced"
    if preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown NEXUS_COLAB_CACHE_PRESET={preset!r}. Valid presets: {valid}")
    for key, value in PRESETS[preset].items():
        _setdefault_env(key, value)

    _ensure_colab_nexus_assets()

    stale_prefixes = (
        "nexus",
        "nexus.training",
        "nexus.training.causal_trainer",
        "nexus.data",
        "nexus.data.metabolic_dataset",
        "nexus.field",
        "nexus.field.query_engine",
        "nexus.physics",
        "nexus.physics.hamiltonian",
    )
    for name in list(sys.modules):
        if name in stale_prefixes or any(name.startswith(prefix + ".") for prefix in stale_prefixes):
            sys.modules.pop(name, None)

    import nexus.training.causal_trainer as _ct_mod
    import nexus.data.metabolic_dataset as _ds_mod
    import nexus.field.query_engine as _qe_mod
    import nexus.physics.hamiltonian as _ham_mod
    importlib.reload(_qe_mod)
    importlib.reload(_ham_mod)
    importlib.reload(_ct_mod)
    importlib.reload(_ds_mod)

    from nexus.data.metabolic_dataset import ZaretzkiMetabolicDataset, geometric_collate_fn
    from nexus.training.causal_trainer import Metabolic_Causal_Trainer
    from nexus.training.losses import NEXUS_God_Loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile = _detect_gpu_profile()
    print(f"Precompute on: {device}")
    print(f"Cache preset : {preset}  profile={profile}")

    all_sdfs = _discover_isoform_sdfs(REPO_DIR)
    target_isoform = _env_str("NEXUS_COLAB_TARGET_ISOFORM", "3A4") or "3A4"
    target_sdf = _resolve_target_isoform_path(target_isoform, all_sdfs)
    max_samples = max(_env_int("NEXUS_COLAB_MAX_SAMPLES", 64), 0)
    dataset = ZaretzkiMetabolicDataset(target_sdf, max_molecules=max_samples)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=geometric_collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Dataset : {len(dataset)} molecules")

    trainer = Metabolic_Causal_Trainer(
        loss_fn=NEXUS_God_Loss(som_loss_mode="focal", focal_gamma=2.0),
        dynamics_steps=max(_env_int("NEXUS_COLAB_DYNAMICS_STEPS", 1), 1),
        dynamics_dt=0.001,
        dynamics_summary_mode="full",
        checkpoint_dynamics=False,
        enable_wsd_scheduler=False,
        enable_static_compile=False,
        use_galore=False,
        dag_loss_weight=0.0,
        dag_loss_cap=1.0,
        analogical_loss_weight=0.0,
        physics_cache_mode="off",
    ).to(device)
    trainer.eval()

    qe = trainer.model.module1.field_engine.quantum_enforcer
    qe.integration_resolution = max(_env_int("NEXUS_COLAB_INTEGRATION_RESOLUTION", 10), 4)
    qe.integration_chunk_size = max(_env_int("NEXUS_COLAB_INTEGRATION_CHUNK", 128), 8)

    se = trainer.model.module1.field_engine.query_engine
    se.n_points = max(_env_int("NEXUS_COLAB_SCAN_N_POINTS", 12), 4)
    se.radius = max(_env_float("NEXUS_COLAB_SCAN_RADIUS", 1.2), 0.1)
    se.query_chunk_size = max(_env_int("NEXUS_COLAB_SCAN_CHUNK", 6), 1)
    se.shell_fractions = _env_shells("NEXUS_COLAB_SCAN_SHELLS", (0.40, 0.70, 1.00))
    se.refine_steps = max(_env_int("NEXUS_COLAB_SCAN_REFINE_STEPS", 0), 0)
    se.create_approach_graph = False

    nav = trainer.model.navigator
    nav.optimization_steps = max(_env_int("NEXUS_COLAB_NAV_OPT_STEPS", 1), 0)
    nav.candidate_batch = max(_env_int("NEXUS_COLAB_NAV_CANDIDATES", 2), 0)

    default_cache = Path("/content/drive/MyDrive/nexus_cyp3a4_physics_cache.pt")
    if not default_cache.parent.exists():
        default_cache = REPO_DIR / "nexus_cyp3a4_physics_cache.pt"
    cache_path = Path(_env_str("NEXUS_COLAB_PHYSICS_CACHE_PATH", str(default_cache)))
    save_every = max(_env_int("NEXUS_COLAB_CACHE_SAVE_EVERY", 8), 1)

    payload = {
        "version": 1,
        "created_at_unix": time.time(),
        "target_isoform": target_isoform,
        "source_sdf": str(target_sdf),
        "config": {
            "dynamics_steps": trainer.dynamics_steps,
            "integration_resolution": qe.integration_resolution,
            "integration_chunk": qe.integration_chunk_size,
            "scan_n_points": se.n_points,
            "scan_radius": se.radius,
            "scan_chunk": se.query_chunk_size,
            "scan_shells": tuple(float(x) for x in se.shell_fractions),
            "scan_refine_steps": se.refine_steps,
            "nav_opt_steps": nav.optimization_steps,
            "nav_candidates": nav.candidate_batch,
        },
        "entries": {},
    }

    if cache_path.exists():
        existing = torch.load(cache_path, map_location="cpu")
        if isinstance(existing, dict) and isinstance(existing.get("entries"), dict):
            payload["entries"].update(existing["entries"])
            print(f"Resuming cache build: {len(payload['entries'])} existing entries from {cache_path}")

    for batch_index, batch in enumerate(loader, start=1):
        batch = trainer._move_to_device(batch, device=device)
        smiles = trainer._resolve_smiles(batch)
        true_atom_index = trainer._resolve_true_atom_index(batch, device=device)
        cache_key = trainer.physics_cache_key(smiles, int(true_atom_index.detach().cpu().item()))
        if cache_key in payload["entries"]:
            continue

        module1_out = trainer._module1_forward_hot_path(smiles)
        true_row_index = trainer._scan_row_index(module1_out.scan.atom_indices, true_atom_index)
        protein_data = trainer._resolve_protein_data(batch)
        accessibility_field = None
        ddi_occupancy = None
        if protein_data is not None:
            true_ranked_index = trainer._scan_row_index(module1_out.ranked_atom_indices, true_atom_index)
            pocket_encoding = trainer._build_pocket_encoding_hot_path(
                module1_out,
                int(true_ranked_index.detach().cpu().item()),
                protein_data,
            )
            accessibility_field = pocket_encoding.accessibility_state
            ddi_occupancy = protein_data.get("ddi_occupancy")

        field = module1_out.field_state.field
        manifold_pos = trainer._sanitize_tensor(
            module1_out.manifold.pos,
            nan=0.0,
            posinf=25.0,
            neginf=-25.0,
            clamp=(-25.0, 25.0),
        )
        target_point_world = trainer._sanitize_tensor(
            module1_out.scan.refined_peak_points[true_row_index],
            nan=0.0,
            posinf=25.0,
            neginf=-25.0,
            clamp=(-25.0, 25.0),
        )
        q_init_internal = trainer._sanitize_tensor(
            field.to_internal_coords(manifold_pos).to(dtype=trainer.model.solver_dtype),
            nan=0.0,
            posinf=25.0,
            neginf=-25.0,
            clamp=(-25.0, 25.0),
        )
        target_point_internal = trainer._sanitize_tensor(
            field.to_internal_coords(target_point_world.view(1, 3)).view(-1).to(dtype=trainer.model.solver_dtype),
            nan=0.0,
            posinf=25.0,
            neginf=-25.0,
            clamp=(-25.0, 25.0),
        )

        pred_rate, h_initial, h_final, ts_eigenvalues = trainer._dynamics_summary_checkpointed(
            q_init_internal,
            target_point_internal,
            smiles=smiles,
            species=module1_out.manifold.species,
            target_atom_index=true_atom_index,
            accessibility_field=accessibility_field,
            ddi_occupancy=ddi_occupancy,
            prebuilt_field=field,
        )
        pred_rate = trainer._sanitize_tensor(
            trainer._to_fp32(pred_rate),
            nan=1.0e-12,
            posinf=trainer._PRED_RATE_MAX,
            neginf=1.0e-12,
            clamp=(1.0e-12, trainer._PRED_RATE_MAX),
        )
        pred_rate_raw = pred_rate.detach()
        n_atoms = max(int(q_init_internal.shape[0]), 1)
        h_initial = trainer._sanitize_tensor(
            trainer._to_fp32(h_initial) / n_atoms,
            nan=2.5,
            posinf=100.0,
            neginf=-100.0,
            clamp=(-100.0, 100.0),
        )
        h_final = trainer._sanitize_tensor(
            trainer._to_fp32(h_final) / n_atoms,
            nan=2.5,
            posinf=100.0,
            neginf=-100.0,
            clamp=(-100.0, 100.0),
        )
        ts_eigenvalues = trainer._sanitize_tensor(
            trainer._to_fp32(ts_eigenvalues),
            nan=0.0,
            posinf=100.0,
            neginf=-100.0,
            clamp=(-100.0, 100.0),
        ).view(-1)
        if ts_eigenvalues.numel() < 2:
            ts_eigenvalues = F.pad(ts_eigenvalues, (0, 2 - ts_eigenvalues.numel()))

        entry = {
            "smiles": smiles,
            "true_atom_index": int(true_atom_index.detach().cpu().item()),
            "pred_rate": pred_rate.detach().cpu(),
            "pred_rate_raw": pred_rate_raw.cpu(),
            "hamiltonian_initial": h_initial.detach().cpu(),
            "hamiltonian_final": h_final.detach().cpu(),
            "ts_eigenvalues": ts_eigenvalues.detach().cpu(),
            "dynamics_fallback": torch.as_tensor(
                1.0 if trainer.last_dynamics_fallback else 0.0,
                dtype=torch.float32,
            ),
            "checkpoint_fallback": torch.as_tensor(
                1.0 if trainer.last_checkpoint_fallback else 0.0,
                dtype=torch.float32,
            ),
        }
        for name, value in trainer.last_kinetics_debug.items():
            entry[name] = value.detach().cpu()
        payload["entries"][cache_key] = entry

        if batch_index == 1 or batch_index % save_every == 0 or batch_index == len(loader):
            _save_cache(cache_path, payload)
            print(
                f"cached={len(payload['entries'])}/{len(dataset)}  "
                f"batch={batch_index}/{len(loader)}  "
                f"fallback={float(entry['dynamics_fallback'].item()):.0f}  "
                f"path={cache_path}",
                flush=True,
            )

    _save_cache(cache_path, payload)
    print(f"Physics cache saved → {cache_path}")


main()
