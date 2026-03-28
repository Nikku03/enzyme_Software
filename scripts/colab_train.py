"""
Colab Cell-5 training script.  Run via:

    exec(open('/content/enzyme_Software/scripts/colab_train.py').read())

All Colab-safe settings are in the TUNABLES block below.
After `git pull` in Cell 1 this file is always up-to-date.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader, Subset

# Suppress FutureWarning from external libraries (e.g. xformers, flash-attn)
# that still call the deprecated torch.backends.cuda.sdp_kernel() API.
# Our code uses torch.nn.attention.sdpa_kernel() throughout.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*sdp_kernel.*",
)

# NaN root causes are fully resolved (hypernetwork scale guards, 2nd-order
# sqrt safety, checkpoint control-flow fix, prebuilt field override).
# set_detect_anomaly is NOT enabled — it adds 2-5× overhead.

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
# TF32: ~20-30% free speedup on A100/RTX Ada for float32 and BF16 matmuls
# with minimal accuracy loss (19-bit mantissa vs 23-bit). Safe for SoM training.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── ensure repo root is on sys.path ────────────────────────────────────────
_REPO_DIR = Path("/content/enzyme_Software")
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

# ── purge stale in-memory nexus modules so git-pulled API changes land ─────
_STALE_PREFIXES = (
    "nexus",
    "nexus.training",
    "nexus.training.losses",
    "nexus.training.causal_trainer",
    "nexus.reasoning",
    "nexus.reasoning.baseline_memory",
    "nexus.data",
    "nexus.data.metabolic_dataset",
    "nexus.field",
    "nexus.field.query_engine",
    "nexus.physics",
    "nexus.physics.hamiltonian",
)
for _name in list(sys.modules):
    _is_stale = _name in _STALE_PREFIXES
    if not _is_stale:
        for _prefix in _STALE_PREFIXES:
            if _name.startswith(_prefix + "."):
                _is_stale = True
                break
    if _is_stale:
        sys.modules.pop(_name, None)

# ── force-reload trainer so git-pulled changes always take effect ──────────
import nexus.training.causal_trainer as _ct_mod          # noqa: E402
import nexus.data.metabolic_dataset as _ds_mod           # noqa: E402
import nexus.field.query_engine as _qe_mod               # noqa: E402
import nexus.reasoning.baseline_memory as _mb_mod        # noqa: E402
import nexus.training.losses as _loss_mod                # noqa: E402
import nexus.physics.hamiltonian as _ham_mod             # noqa: E402
importlib.reload(_qe_mod)
importlib.reload(_mb_mod)
importlib.reload(_loss_mod)
importlib.reload(_ham_mod)   # picks up NaN diagnostics & reactive_reference guard
importlib.reload(_ct_mod)    # must reload after hamiltonian so causal_trainer gets fresh ref
importlib.reload(_ds_mod)

def _discover_isoform_sdfs(repo_dir: Path) -> tuple[list[Path], Path]:
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
    all_sdfs = [found[key] for key in sorted(found)]
    if not all_sdfs:
        searched = ", ".join(str(p) for p in search_dirs)
        raise FileNotFoundError(
            "No CYP isoform SDF files were found for Colab training. "
            f"Searched: {searched}. "
            "This clone does not include the ATTNSOM/CYP_DBs SDF assets."
        )
    preferred_source = next(
        (p.parent for p in all_sdfs if p.parent.name == "cyp_dataset"),
        all_sdfs[0].parent,
    )
    return all_sdfs, preferred_source


# All CYP isoform SDF files — used for both training and the analogical bank.
_ALL_SDFS, _SDF_DIR = _discover_isoform_sdfs(_REPO_DIR)
# Backward-compat: keep SDF pointing at 3A4 for helpers that need a single target.
SDF = next((p for p in _ALL_SDFS if p.stem.upper() == "3A4"), _ALL_SDFS[0])

# Checkpoint paths: prefer Google Drive (persists across sessions) → fall back to repo dir.
_DRIVE_CKPT = Path("/content/drive/MyDrive/nexus_colab_checkpoint.pt")
_LOCAL_CKPT = _REPO_DIR / "colab_nexus_checkpoint.pt"
_DEFAULT_CKPT_PATH = _DRIVE_CKPT if _DRIVE_CKPT.parent.exists() else _LOCAL_CKPT
CKPT_PATH = Path(os.environ.get("NEXUS_COLAB_CHECKPOINT_PATH", str(_DEFAULT_CKPT_PATH))).expanduser()
BATCH_CKPT_PATH = Path(
    os.environ.get(
        "NEXUS_COLAB_BATCH_CHECKPOINT_PATH",
        str(CKPT_PATH.with_name(CKPT_PATH.stem + "_batch.pt")),
    )
).expanduser()
BATCH_METRICS_PATH = Path(
    os.environ.get(
        "NEXUS_COLAB_BATCH_METRICS_PATH",
        str(CKPT_PATH.parent / "nexus_colab_batch_metrics.json"),
    )
).expanduser()
ANALOGICAL_TRACE_PATH_DEFAULT = CKPT_PATH.parent / "nexus_colab_analogical_trace.jsonl"
DEFAULT_PHYSICS_CACHE_PATH = CKPT_PATH.parent / "nexus_cyp3a4_physics_cache.pt"
DEFAULT_ANALOGICAL_BANK_CACHE_PATH = CKPT_PATH.parent / "nexus_continuous_analogical_bank.pt"

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
        "l4_24gb": "l4",
        "rtx3090": "l4",
        "rtx4090": "l4",
    }
    return aliases.get(value.strip().lower(), "auto")


def _strict_profile_override_enabled() -> bool:
    raw = os.environ.get("NEXUS_COLAB_STRICT_PROFILE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


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
    parts = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parts.append(float(item))
        except ValueError:
            return default
    return tuple(parts) if parts else default


def _env_str(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _detect_gpu_profile() -> str:
    env_raw = os.environ.get("NEXUS_COLAB_GPU_PROFILE", "auto").strip().lower()
    normalized = _normalize_profile_name(env_raw)
    valid_profiles = {"standard", "l4", "high_vram", "ultra_vram"}
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = float(props.total_memory) / 1024**3
            print(f"GPU : {props.name}  |  total memory : {total_gb:.1f} GB")
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
            if normalized in valid_profiles:
                return normalized
            if total_gb >= 70.0:   # A100-80GB, H100-80GB, H100 SXM5-94GB, …
                return "ultra_vram"
            if total_gb >= 35.0:   # A100-40GB, L40S-48GB, …
                return "high_vram"
            if total_gb >= 20.0:   # L4-24GB, RTX 3090/4090-24GB, …
                return "l4"
        if normalized in valid_profiles:
            return normalized
    except Exception as e:
        fallback = normalized if normalized in valid_profiles else "standard"
        print(f"GPU profile detection fallback: {type(e).__name__}: {e} -> {fallback}")
        return fallback
    return "standard"


GPU_PROFILE = _detect_gpu_profile()
GPU_PROFILES = {
    "standard": {
        "max_samples": 64,
        "epochs": 1,
        "steps": 1,
        "physics_mode": "lite",
        "low_memory": True,
        "checkpoint": False,
        "integration_resolution": 8,
        "integration_chunk": 32,
        "scan_n_points": 8,
        "scan_radius": 1.0,
        "scan_chunk": 2,
        "scan_shells": (0.5, 1.0),
        "scan_refine_steps": 0,
    },
    "high_vram": {
        "max_samples": 64,
        "epochs": 1,
        "steps": 1,
        "physics_mode": "lite",
        "low_memory": True,
        "checkpoint": False,
        "integration_resolution": 10,
        "integration_chunk": 96,
        "scan_n_points": 12,
        "scan_radius": 1.0,
        "scan_chunk": 6,
        "scan_shells": (0.5, 1.0),
        "scan_refine_steps": 0,
    },
    # ── 20–34 GB (L4-24GB, RTX 3090/4090-24GB) ─────────────────────────────
    # Designed for a 6–7 hour training window.  Key constraints vs ultra_vram:
    #   * dynamics_steps=1  — halves the create_graph ODE graph depth and VRAM
    #   * integration_resolution=10 (1000 pts) vs 12³=1728 for ultra
    #   * 3 scan shells vs 4; 16 scan points vs 24
    #   * nav_opt_steps=2, nav_candidates=3 (full epochs); epoch-adaptive logic
    #     in the training loop starts even lighter (opt=1, cand=2) for epoch 4
    #   * low_memory=False so the adaptive logic controls per-epoch mode:
    #       epochs 1-3 → scan-only (~25-40 min each)
    #       epoch  4   → dynamics on ≤28-atom molecules only (~2 hrs)
    #       epochs 5-6 → full dynamics, all molecules (~3-4 hrs each)
    #     Checkpointing means a second 6-hr session completes the run.
    "l4": {
        "max_samples": 64,
        "epochs": 6,
        "steps": 1,
        "physics_mode": "lite",
        "low_memory": False,
        "checkpoint": False,
        "integration_resolution": 10,
        "integration_chunk": 64,
        "scan_n_points": 16,
        "scan_radius": 1.5,
        "scan_chunk": 8,
        "scan_shells": (0.40, 0.65, 1.00),
        "scan_refine_steps": 1,
        "nav_opt_steps": 2,
        "nav_candidates": 3,
    },
    # ── 85+ GB (H100 SXM5, A100-80, Blackwell 96GB, etc.) ──────────────────
    # Full physics — quantum grid 12^3, 4 scan shells, full Navigator budget.
    # checkpoint=False: the prebuilt-field override means dynamics no longer
    # rebuilds the SIREN on every solver step, so checkpointing buys nothing.
    "ultra_vram": {
        "max_samples": 64,
        "epochs": 8,
        "steps": 2,
        "physics_mode": "full",
        "low_memory": False,
        "checkpoint": False,
        "integration_resolution": 12,
        "integration_chunk": 192,
        "scan_n_points": 24,
        "scan_radius": 1.75,
        "scan_chunk": 12,
        "scan_shells": (0.40, 0.65, 0.85, 1.00),
        "scan_refine_steps": 1,
        "nav_opt_steps": 6,
        "nav_candidates": 8,
    },
}
CFG = GPU_PROFILES[GPU_PROFILE]
CFG = dict(CFG)
CFG["max_samples"] = max(_env_int("NEXUS_COLAB_MAX_SAMPLES", CFG["max_samples"]), 0)
CFG["epochs"] = max(_env_int("NEXUS_COLAB_EPOCHS", CFG["epochs"]), 1)
CFG["steps"] = max(_env_int("NEXUS_COLAB_DYNAMICS_STEPS", CFG["steps"]), 1)
CFG["integration_resolution"] = max(_env_int("NEXUS_COLAB_INTEGRATION_RESOLUTION", CFG["integration_resolution"]), 4)
CFG["integration_chunk"] = max(_env_int("NEXUS_COLAB_INTEGRATION_CHUNK", CFG["integration_chunk"]), 8)
CFG["scan_n_points"] = max(_env_int("NEXUS_COLAB_SCAN_N_POINTS", CFG["scan_n_points"]), 4)
CFG["scan_radius"] = max(_env_float("NEXUS_COLAB_SCAN_RADIUS", CFG["scan_radius"]), 0.1)
CFG["scan_chunk"] = max(_env_int("NEXUS_COLAB_SCAN_CHUNK", CFG["scan_chunk"]), 1)
CFG["scan_shells"] = _env_shells("NEXUS_COLAB_SCAN_SHELLS", CFG["scan_shells"])
CFG["scan_refine_steps"] = max(_env_int("NEXUS_COLAB_SCAN_REFINE_STEPS", CFG["scan_refine_steps"]), 0)
CFG["nav_opt_steps"] = max(_env_int("NEXUS_COLAB_NAV_OPT_STEPS", CFG.get("nav_opt_steps", 6)), 0)
CFG["nav_candidates"] = max(_env_int("NEXUS_COLAB_NAV_CANDIDATES", CFG.get("nav_candidates", 8)), 0)
TARGET_ISOFORM = _env_str("NEXUS_COLAB_TARGET_ISOFORM", "")
SAVE_EVERY_BATCH = _env_bool("NEXUS_COLAB_SAVE_EVERY_BATCH", True)
SHUFFLE_SEED = _env_int("NEXUS_COLAB_SHUFFLE_SEED", 42)
DAG_LOSS_WEIGHT = _env_float("NEXUS_COLAB_DAG_LOSS_WEIGHT", 1.0)
DAG_LOSS_CAP = _env_float("NEXUS_COLAB_DAG_LOSS_CAP", 4.0)
DAG_WARMUP_STEPS = _env_int("NEXUS_COLAB_DAG_WARMUP_STEPS", 0)
ANA_LOSS_WEIGHT = _env_float("NEXUS_COLAB_ANA_LOSS_WEIGHT", 1.0)
ANALOGICAL_BANK_MODE = _env_str("NEXUS_COLAB_ANALOGICAL_BANK_MODE", "fingerprint").strip().lower() or "fingerprint"
if ANALOGICAL_BANK_MODE not in {"fingerprint", "continuous"}:
    ANALOGICAL_BANK_MODE = "fingerprint"
ANALOGICAL_TRACE_ENABLED = _env_bool("NEXUS_COLAB_ANALOGICAL_TRACE", True)
ANALOGICAL_TRACE_PATH = Path(
    _env_str("NEXUS_COLAB_ANALOGICAL_TRACE_PATH", str(ANALOGICAL_TRACE_PATH_DEFAULT))
)
ANALOGICAL_BANK_CACHE_ENABLED = _env_bool("NEXUS_COLAB_ANALOGICAL_BANK_CACHE", True)
ANALOGICAL_BANK_CACHE_PATH = Path(
    _env_str("NEXUS_COLAB_ANALOGICAL_BANK_CACHE_PATH", str(DEFAULT_ANALOGICAL_BANK_CACHE_PATH))
)
PHYSICS_CACHE_MODE = _env_str("NEXUS_COLAB_PHYSICS_CACHE_MODE", "off").lower() or "off"
if PHYSICS_CACHE_MODE not in {"off", "cached", "hybrid"}:
    PHYSICS_CACHE_MODE = "off"
PHYSICS_CACHE_PATH = Path(_env_str("NEXUS_COLAB_PHYSICS_CACHE_PATH", str(DEFAULT_PHYSICS_CACHE_PATH)))
ALLOW_COMPILE = _env_bool("NEXUS_COLAB_ALLOW_COMPILE", False)
DATA_NUM_WORKERS = _env_int("NEXUS_COLAB_NUM_WORKERS", 0)
CURRICULUM_PHYSICS = _env_bool("NEXUS_COLAB_CURRICULUM", False)
IGNORE_CHECKPOINTS = _env_bool("NEXUS_COLAB_IGNORE_CHECKPOINTS", False)

from nexus.data.metabolic_dataset import ZaretzkiMetabolicDataset, geometric_collate_fn
from nexus.training.causal_trainer import Metabolic_Causal_Trainer
from nexus.training.losses import NEXUS_God_Loss
from torch.utils.data import ConcatDataset, Subset


def _canonical_smiles(mol) -> str | None:
    try:
        from rdkit import Chem as _Chem
        base = _Chem.RemoveHs(_Chem.Mol(mol))
        return _Chem.MolToSmiles(base, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _collect_memory_bank_mols(target_sdf: Path, dataset: ZaretzkiMetabolicDataset) -> list:
    from rdkit import Chem as _Chem

    target_name = target_sdf.name
    target_smiles = {
        smi for smi in (_canonical_smiles(m) for m in dataset.mols) if smi
    }
    unique_bank: dict[str, object] = {}

    candidate_paths = []
    attnsom_dir = target_sdf.parent
    if attnsom_dir.exists():
        candidate_paths.extend(sorted(attnsom_dir.glob("*.sdf")))
    cyp_dbs_dir = _REPO_DIR / "CYP_DBs"
    if cyp_dbs_dir.exists():
        candidate_paths.extend(sorted(cyp_dbs_dir.glob("*.sdf")))

    considered = 0
    skipped_same_file = 0
    skipped_train_overlap = 0
    skipped_duplicates = 0

    for path in candidate_paths:
        if path.name == target_name and path.resolve() == target_sdf.resolve():
            skipped_same_file += 1
            continue
        suppl = _Chem.SDMolSupplier(str(path), removeHs=False)
        for mol in suppl:
            if mol is None:
                continue
            considered += 1
            smi = _canonical_smiles(mol)
            if not smi:
                continue
            if smi in target_smiles:
                skipped_train_overlap += 1
                continue
            if smi in unique_bank:
                skipped_duplicates += 1
                continue
            unique_bank[smi] = mol

    print(
        "Analogical bank sources:"
        f" considered={considered}"
        f" unique={len(unique_bank)}"
        f" skipped_same_file={skipped_same_file}"
        f" skipped_train_overlap={skipped_train_overlap}"
        f" skipped_duplicates={skipped_duplicates}"
    )
    return list(unique_bank.values())


def _make_loader(dataset, indices, *, shuffle: bool, device: torch.device) -> DataLoader:
    subset = dataset if len(indices) == len(dataset) else Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=geometric_collate_fn,
        num_workers=DATA_NUM_WORKERS,
        persistent_workers=(DATA_NUM_WORKERS > 0),
        pin_memory=(device.type == "cuda"),
    )


def _epoch_indices(n_items: int, epoch: int, seed: int) -> list[int]:
    if n_items <= 0:
        return []
    g = torch.Generator()
    g.manual_seed(int(seed) + int(epoch))
    return torch.randperm(n_items, generator=g).tolist()


def _serialize_state_dict(module) -> dict:
    serialized = {}
    for key, value in module.state_dict().items():
        if isinstance(value, UninitializedParameter):
            continue
        try:
            serialized[key] = value.detach().cpu()
        except ValueError:
            # Lazy modules can still expose uninitialized parameters in state_dict()
            # before their first forward pass. Skip them so checkpointing remains
            # robust; load_state_dict(strict=False) will tolerate the missing keys.
            continue
    return serialized


def _resolve_target_isoform_path(target_isoform: str, candidates: list[Path]) -> Path:
    token = target_isoform.strip()
    if not token:
        raise ValueError("target_isoform must be non-empty")
    token = token.removesuffix(".sdf")
    by_stem = {p.stem.upper(): p for p in candidates}
    by_name = {p.name.upper(): p for p in candidates}
    resolved = by_stem.get(token.upper()) or by_name.get(f"{token.upper()}.SDF")
    if resolved is None:
        valid = ", ".join(sorted(p.stem for p in candidates))
        raise FileNotFoundError(
            f"NEXUS_COLAB_TARGET_ISOFORM={target_isoform!r} did not match any available SDF. "
            f"Valid isoforms: {valid}"
        )
    return resolved

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}\n")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

# ── dataset — all 9 CYP isoforms ───────────────────────────────────────────
_train_sdfs = _ALL_SDFS
if TARGET_ISOFORM:
    _target_path = _resolve_target_isoform_path(TARGET_ISOFORM, _ALL_SDFS)
    _train_sdfs = [_target_path]

_sub_datasets = []
for _sdf in _train_sdfs:
    try:
        _sub_datasets.append(ZaretzkiMetabolicDataset(_sdf, max_molecules=CFG["max_samples"]))
    except Exception as _sdf_err:
        print(f"  Skipping {_sdf.name}: {_sdf_err}")
dataset = ConcatDataset(_sub_datasets) if len(_sub_datasets) > 1 else _sub_datasets[0]
_all_indices = list(range(len(dataset)))
loader = _make_loader(dataset, _all_indices, shuffle=True, device=device)
print(f"Dataset : {len(dataset)} molecules")
print(f"Profile : {GPU_PROFILE}  (override: NEXUS_COLAB_GPU_PROFILE=ultra_vram|high_vram|l4|standard)")
print(f"Physics mode : {CFG['physics_mode']}")
print(f"Training source : {', '.join(_sdf.name for _sdf in _train_sdfs)}")
print(
    "Runtime knobs : "
    f"max_samples/isoform={CFG['max_samples']}  "
    f"epochs={CFG['epochs']}  "
    f"steps={CFG['steps']}"
)
print(
    f"Loss routing : dag_weight={DAG_LOSS_WEIGHT:g}  "
    f"dag_cap={DAG_LOSS_CAP:g}  "
    f"ana_weight={ANA_LOSS_WEIGHT:g}"
)
print(f"Analogical bank : mode={ANALOGICAL_BANK_MODE}")
if ANALOGICAL_BANK_MODE == "continuous":
    print(
        f"Analogical bank cache: {'on' if ANALOGICAL_BANK_CACHE_ENABLED else 'off'}"
    )
    if ANALOGICAL_BANK_CACHE_ENABLED:
        print(f"  analogical bank cache → {ANALOGICAL_BANK_CACHE_PATH}")
print(f"Analogical trace: {'on' if ANALOGICAL_TRACE_ENABLED else 'off'}")
if ANALOGICAL_TRACE_ENABLED:
    print(f"  analogical trace log → {ANALOGICAL_TRACE_PATH}")
print(f"Physics cache : mode={PHYSICS_CACHE_MODE}  path={PHYSICS_CACHE_PATH}")

# ── trainer ────────────────────────────────────────────────────────────────
trainer = Metabolic_Causal_Trainer(
    loss_fn=NEXUS_God_Loss(som_loss_mode="focal", focal_gamma=2.0),
    dynamics_steps=CFG["steps"],
    dynamics_dt=0.001,
    dynamics_summary_mode=CFG["physics_mode"],
    checkpoint_dynamics=CFG["checkpoint"],
    enable_wsd_scheduler=True,
    low_memory_train_mode=CFG["low_memory"],
    low_memory_scan_gradients=(GPU_PROFILE in {"high_vram", "ultra_vram"}),
    enable_static_compile=ALLOW_COMPILE,  # safe on dedicated GPUs; set NEXUS_COLAB_ALLOW_COMPILE=1
    use_galore=False,             # plain AdamW — avoids GaLore SVD on Colab
    dag_loss_weight=DAG_LOSS_WEIGHT,
    dag_loss_cap=DAG_LOSS_CAP,
    dag_warmup_steps=DAG_WARMUP_STEPS,
    analogical_loss_weight=ANA_LOSS_WEIGHT,
    physics_cache_mode=PHYSICS_CACHE_MODE,
).to(device)
trainer.sync_memory_bank_device(device)
if PHYSICS_CACHE_MODE != "off":
    if PHYSICS_CACHE_PATH.exists():
        _cache_count = trainer.load_physics_cache(PHYSICS_CACHE_PATH, mode=PHYSICS_CACHE_MODE)
        print(f"  loaded { _cache_count } cached physics entries")
    elif PHYSICS_CACHE_MODE == "cached":
        raise FileNotFoundError(
            f"Physics cache mode is 'cached' but no cache file exists at {PHYSICS_CACHE_PATH}"
        )
    else:
        print("  cache file missing; hybrid mode will fall back to live dynamics")

# ── Colab-safe quantum grid ────────────────────────────────────────────────
qe = trainer.model.module1.field_engine.quantum_enforcer
qe.integration_resolution = CFG["integration_resolution"]
qe.integration_chunk_size = CFG["integration_chunk"]
print(
    f"Quantum grid : {CFG['integration_resolution']}^3 = {CFG['integration_resolution']**3} pts,  "
    f"chunk={CFG['integration_chunk']}"
)

# ── Reaction-volume scanner ────────────────────────────────────────────────
se = trainer.model.module1.field_engine.query_engine
se.n_points             = CFG["scan_n_points"]
se.radius               = CFG["scan_radius"]
se.query_chunk_size     = CFG["scan_chunk"]
se.shell_fractions      = CFG["scan_shells"]
se.refine_steps         = CFG["scan_refine_steps"]
se.create_approach_graph = False   # no second-order grad in approach vectors
print(
    f"Query engine : {CFG['scan_n_points']} pts × {len(CFG['scan_shells'])} shells × "
    f"{CFG['scan_refine_steps']} refine step(s)"
)

# ── Navigator budget ───────────────────────────────────────────────────────
# Full budget: opt_steps=6 (Adam, create_graph) + 1 zero-momentum + 8 random
#              = 16 trajectories/atom — each runs CliffordLieIntegrator.
# Lower values can be set via nav_opt_steps / nav_candidates in the profile.
nav = trainer.model.navigator
nav.optimization_steps = CFG.get("nav_opt_steps", 6)
nav.candidate_batch    = CFG.get("nav_candidates", 8)
print(
    f"Navigator    : opt_steps={nav.optimization_steps}  "
    f"candidates={nav.candidate_batch}  "
    f"(total trajectories/atom ≈ {nav.optimization_steps + 1 + nav.candidate_batch + 1})"
)

# ── physics curriculum ─────────────────────────────────────────────────────
# Progressively increases quantum/scan resolution over training epochs.
# Phase 1 (first 35%): coarse fields — 60% of configured resolution.
#   The model is far from any solution; coarse gradients are just as useful.
# Phase 2 (35-70%): 80% resolution + all scan shells — model has direction.
# Phase 3 (70-100%): full configured physics — fine-grain polish.
# Enable by setting NEXUS_COLAB_CURRICULUM=1. Safe to combine with ALLOW_COMPILE.
def _apply_physics_curriculum(epoch: int, total_epochs: int) -> None:
    if not CURRICULUM_PHYSICS:
        return
    phase = epoch / max(total_epochs, 1)
    if phase < 0.35:
        _res  = max(4, int(CFG["integration_resolution"] * 0.60))
        _pts  = max(4, int(CFG["scan_n_points"] * 0.60))
        _shls = CFG["scan_shells"][-1:]  # outer shell only
        _ref  = 0
    elif phase < 0.70:
        _res  = max(6, int(CFG["integration_resolution"] * 0.80))
        _pts  = max(6, int(CFG["scan_n_points"] * 0.80))
        _shls = CFG["scan_shells"]
        _ref  = 0
    else:
        _res  = CFG["integration_resolution"]
        _pts  = CFG["scan_n_points"]
        _shls = CFG["scan_shells"]
        _ref  = CFG["scan_refine_steps"]
    qe.integration_resolution = _res
    se.n_points   = _pts
    se.shell_fractions = _shls
    se.refine_steps    = _ref
    _tag = "lite" if phase < 0.35 else "medium" if phase < 0.70 else "full"
    print(
        f"  [curriculum={_tag}]  integration_res={_res}  "
        f"scan_pts={_pts}  shells={len(_shls)}  refine={_ref}"
    )


# ── memory bank — ALL labeled molecules from ALL 9 CYP isoforms (uncapped) ──
# In fingerprint mode this is cheap. In continuous mode each bank molecule is
# projected through module-1 into the Poincare bank once at startup.
# Exact canonical-SMILES masking still prevents trivial self-retrieval.
print("Populating memory bank from all CYP isoform data (full, uncapped)...")
_bank_mols = []
for _sdf in _ALL_SDFS:
    try:
        _bank_ds = ZaretzkiMetabolicDataset(_sdf, max_molecules=0)  # 0 = all molecules
        _bank_mols.extend(_bank_ds.mols)
    except Exception as _e:
        print(f"  Skipping {_sdf.name} for bank: {_e}")
_continuous_bank_encoder = trainer.encode_mol_for_memory_bank if ANALOGICAL_BANK_MODE == "continuous" else None
trainer.memory_bank.populate_from_mols(
    _bank_mols,
    continuous_encoder=_continuous_bank_encoder,
    continuous_cache_path=(
        ANALOGICAL_BANK_CACHE_PATH
        if ANALOGICAL_BANK_MODE == "continuous" and ANALOGICAL_BANK_CACHE_ENABLED
        else None
    ),
)
print(f"Memory bank ready: {len(trainer.memory_bank.historical_mols)} molecules.\n")
del _bank_mols
_bank_ds = None  # noqa: release last-loop reference

total_training_steps = CFG["epochs"] * max(len(_all_indices), 1)
trainer.set_total_training_steps(total_training_steps)
trainer.configure_optimizers()
if device.type == "cuda":
    torch.cuda.empty_cache()
print("Optimizer ready.\n")
print(f"Batch checkpointing : {'on' if SAVE_EVERY_BATCH else 'off'}")
if SAVE_EVERY_BATCH:
    print(f"  rolling batch checkpoint → {BATCH_CKPT_PATH}")
    print(f"  rolling batch metrics    → {BATCH_METRICS_PATH}")
print()


def _save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    batch_in_epoch: int,
    total_batches: int,
    metrics_history: list,
    last_batch_metrics: dict | None = None,
) -> None:
    payload: dict = {
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "total_batches": total_batches,
        "model_state_dict": _serialize_state_dict(trainer),
        "metrics_history": metrics_history,
        "shuffle_seed": SHUFFLE_SEED,
    }
    if last_batch_metrics is not None:
        payload["last_batch_metrics"] = last_batch_metrics
    if trainer.optimizer is not None:
        payload["optimizer_state_dict"] = trainer.optimizer.state_dict()
    if trainer.scheduler is not None:
        payload["scheduler_state_dict"] = trainer.scheduler.state_dict()
    torch.save(payload, path)


def _write_batch_metrics(
    *,
    epoch: int,
    batch_index: int,
    total_batches: int,
    running_metrics: dict,
    step_metrics: dict,
) -> None:
    payload = {
        "epoch": epoch + 1,
        "batch": batch_index,
        "total_batches": total_batches,
        "running_metrics": running_metrics,
        "step_metrics": step_metrics,
    }
    BATCH_METRICS_PATH.write_text(json.dumps(payload, indent=2))

# ── checkpoint resume ───────────────────────────────────────────────────────
start_epoch = 0
resume_batch_in_epoch = 0
history = []
if IGNORE_CHECKPOINTS:
    print("Ignoring saved checkpoints due to NEXUS_COLAB_IGNORE_CHECKPOINTS=1.\n")
elif CKPT_PATH.exists():
    print(f"Loading checkpoint from {CKPT_PATH} ...")
    _ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    trainer.load_state_dict(_ckpt["model_state_dict"], strict=False)
    if "optimizer_state_dict" in _ckpt and trainer.optimizer is not None:
        try:
            trainer.optimizer.load_state_dict(_ckpt["optimizer_state_dict"])
            print("  Optimizer state restored.")
        except Exception as _oe:
            print(f"  Optimizer state not restored (shape mismatch after arch change): {_oe}")
    if "scheduler_state_dict" in _ckpt and trainer.scheduler is not None:
        try:
            trainer.scheduler.load_state_dict(_ckpt["scheduler_state_dict"])
            print("  Scheduler state restored.")
        except Exception as _se:
            print(f"  Scheduler state not restored: {_se}")
    start_epoch = int(_ckpt.get("epoch", 0))
    history = list(_ckpt.get("metrics_history", []))
    print(f"  Resumed from epoch {start_epoch}  ({len(history)} epochs completed)\n")
else:
    print(f"No checkpoint at {CKPT_PATH} — starting fresh.\n")

if (not IGNORE_CHECKPOINTS) and SAVE_EVERY_BATCH and BATCH_CKPT_PATH.exists():
    print(f"Loading rolling batch checkpoint from {BATCH_CKPT_PATH} ...")
    _bckpt = torch.load(BATCH_CKPT_PATH, map_location=device, weights_only=False)
    _b_epoch = int(_bckpt.get("epoch", 0))
    _b_batch = int(_bckpt.get("batch_in_epoch", 0))
    if _b_epoch < CFG["epochs"] and (_b_epoch > start_epoch or (_b_epoch == start_epoch and _b_batch > 0)):
        trainer.load_state_dict(_bckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in _bckpt and trainer.optimizer is not None:
            try:
                trainer.optimizer.load_state_dict(_bckpt["optimizer_state_dict"])
                print("  Optimizer state restored from batch checkpoint.")
            except Exception as _oe:
                print(f"  Optimizer state not restored from batch checkpoint: {_oe}")
        if "scheduler_state_dict" in _bckpt and trainer.scheduler is not None:
            try:
                trainer.scheduler.load_state_dict(_bckpt["scheduler_state_dict"])
                print("  Scheduler state restored from batch checkpoint.")
            except Exception as _se:
                print(f"  Scheduler state not restored from batch checkpoint: {_se}")
        start_epoch = _b_epoch
        resume_batch_in_epoch = _b_batch
        history = list(_bckpt.get("metrics_history", history))
        print(
            f"  Resuming mid-epoch: epoch {start_epoch + 1}, "
            f"batch {resume_batch_in_epoch}/{int(_bckpt.get('total_batches', 0))}\n"
        )
    else:
        print("  Rolling batch checkpoint is older than epoch checkpoint; ignoring.\n")

if hasattr(trainer, "current_epoch_index"):
    trainer.current_epoch_index = int(start_epoch)
if ANALOGICAL_TRACE_ENABLED:
    trainer.set_analogical_trace(
        ANALOGICAL_TRACE_PATH,
        enabled=True,
        truncate=(start_epoch == 0 and resume_batch_in_epoch == 0),
    )


# Re-encode the memory bank every N epochs so the stored Poincaré embeddings
# stay aligned with the evolving HGNNProjection encoder.  Bank is already
# fresh at epoch 0 (just populated), so we skip the first iteration.
# Use encode_mol_for_memory_bank (SDF-conformer fast path): ~45-120 s for 457
# molecules vs 30-90 min for the full SMILES → MACE-OFF slow path.
_BANK_RE_ENCODE_PERIOD = int(os.environ.get("NEXUS_COLAB_BANK_RE_ENCODE_PERIOD", "5"))

# ── training loop ──────────────────────────────────────────────────────────
for epoch in range(start_epoch, CFG["epochs"]):
    _apply_physics_curriculum(epoch, CFG["epochs"])

    # Re-encode bank at configured interval (skip epoch 0 / start_epoch since bank
    # was just populated before the loop).
    if (
        epoch > start_epoch
        and _BANK_RE_ENCODE_PERIOD > 0
        and (epoch - start_epoch) % _BANK_RE_ENCODE_PERIOD == 0
        and hasattr(trainer, "re_encode_bank")
    ):
        print(f"  [bank] Re-encoding memory bank at epoch {epoch+1} …", flush=True)
        trainer.re_encode_bank()
        print(f"  [bank] Re-encoding complete.", flush=True)

    epoch_indices = _epoch_indices(len(_all_indices), epoch, SHUFFLE_SEED)
    if epoch == start_epoch and resume_batch_in_epoch > 0:
        epoch_indices = epoch_indices[resume_batch_in_epoch:]
        print(
            f"Epoch {epoch+1}/{CFG['epochs']} | {len(_all_indices)} molecules | "
            f"full dynamics | resuming from batch {resume_batch_in_epoch + 1}"
        )
    else:
        print(f"Epoch {epoch+1}/{CFG['epochs']} | {len(_all_indices)} molecules | full dynamics")
    loader = _make_loader(dataset, epoch_indices, shuffle=False, device=device)

    def _on_batch_end(*, batch_index, total_batches, running_metrics, step_metrics, train):
        if not (SAVE_EVERY_BATCH and train):
            return
        absolute_batch = resume_batch_in_epoch + batch_index if epoch == start_epoch else batch_index
        _save_training_checkpoint(
            BATCH_CKPT_PATH,
            epoch=epoch,
            batch_in_epoch=absolute_batch,
            total_batches=len(_all_indices),
            metrics_history=history,
            last_batch_metrics=running_metrics,
        )
        _write_batch_metrics(
            epoch=epoch,
            batch_index=absolute_batch,
            total_batches=len(_all_indices),
            running_metrics=running_metrics,
            step_metrics=step_metrics,
        )

    metrics = trainer.fit_epoch(loader, train=True, log_every=1, on_batch_end=_on_batch_end)
    history.append(metrics)
    print(f"\n── epoch {epoch+1} summary ──────────────────────────────────────")
    for _k in ["loss_total", "som_top1", "som_top2", "pred_rate", "physics_cache_hit",
               "hamiltonian_initial", "dag_causal_loss", "dag_loss_contribution",
               "ana_loss_total", "ana_gate_open", "ana_confidence",
               "ana_peak", "ana_gate_conf_ok", "ana_gate_peak_ok",
               "ana_weight_fp", "ana_weight_ana", "ana_transport_ok",
               "ana_watson_agreement", "ana_encoder_loss"]:
        if _k in metrics:
            print(f"  {_k:<30} {metrics[_k]:.6g}")
    print()

    # ── save checkpoint after every epoch ──────────────────────────────────
    _save_training_checkpoint(
        CKPT_PATH,
        epoch=epoch + 1,
        batch_in_epoch=0,
        total_batches=len(_all_indices),
        metrics_history=history,
        last_batch_metrics=metrics,
    )
    print(f"  Checkpoint saved → {CKPT_PATH}")
    if SAVE_EVERY_BATCH and BATCH_CKPT_PATH.exists():
        BATCH_CKPT_PATH.unlink()
    resume_batch_in_epoch = 0

if device.type == "cuda":
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Peak GPU memory : {peak_mb:.1f} MB")

# ── save metrics log alongside the checkpoint (persistent if Drive is mounted) ──
# history already contains all epochs from previous sessions (loaded from checkpoint),
# so this file always reflects the complete training history across all sessions.
_log_path = CKPT_PATH.parent / "nexus_colab_metrics.json"
_log_path.write_text(json.dumps(history, indent=2))
print(f"Metrics log saved → {_log_path}")
# Also keep a local copy for easy inspection inside the repo dir.
(_REPO_DIR / "colab_train_metrics.json").write_text(json.dumps(history, indent=2))
