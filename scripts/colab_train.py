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
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

# NaN root causes are fully resolved (hypernetwork scale guards, 2nd-order
# sqrt safety, checkpoint control-flow fix, prebuilt field override).
# set_detect_anomaly is NOT enabled — it adds 2-5× overhead.

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

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
    if _name in _STALE_PREFIXES or any(_name.startswith(_prefix + ".") for _prefix in _STALE_PREFIXES):
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

# All 9 CYP isoform SDF files — used for both training and the analogical bank.
_SDF_DIR  = _REPO_DIR / "data/ATTNSOM/cyp_dataset"
_ALL_SDFS = sorted(_SDF_DIR.glob("*.sdf"))
# Backward-compat: keep SDF pointing at 3A4 for the memory-bank helper that needs
# a single "target" to exclude from the bank.
SDF = _SDF_DIR / "3A4.sdf"

# Checkpoint paths: prefer Google Drive (persists across sessions) → fall back to repo dir.
_DRIVE_CKPT = Path("/content/drive/MyDrive/nexus_colab_checkpoint.pt")
_LOCAL_CKPT = _REPO_DIR / "colab_nexus_checkpoint.pt"
CKPT_PATH   = _DRIVE_CKPT if _DRIVE_CKPT.parent.exists() else _LOCAL_CKPT

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


def _detect_gpu_profile() -> str:
    env = os.environ.get("NEXUS_COLAB_GPU_PROFILE", "auto").strip().lower()
    normalized = _normalize_profile_name(env)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
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
    if normalized in {"standard", "high_vram", "ultra_vram"}:
        return normalized
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        if total_gb >= 70.0:   # A100-80GB, H100-80GB, H100 SXM5-94GB, …
            return "ultra_vram"
        if total_gb >= 35.0:   # A100-40GB, L40S, …
            return "high_vram"
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
    # ── 85+ GB (H100 SXM5, A100-80, Blackwell 96GB, etc.) ──────────────────
    # Full physics quality, but trimmed to stay near an 85–90 GB envelope
    # instead of saturating the whole card.  The dominant VRAM drivers were:
    #   * 3 solver steps
    #   * 48 scan points × 5 shells × 2 refinement passes
    #   * 14^3 quantum grid
    # This profile keeps full dynamics active while backing those off to a
    # still-heavy but more survivable setting.
    # checkpoint=False: the prebuilt-field override (746f305 / b238f3f) means
    # dynamics no longer rebuilds the SIREN on every solver step, so activation
    # memory is small enough that checkpointing buys nothing.  Checkpointing
    # _dynamics_summary is also incompatible with the fallback path, which calls
    # energy_and_forces → autograd.grad inside the checkpointed region, causing
    # a dtype mismatch (float64 forward vs float32 recomputation) on every
    # molecule that triggers the fallback.
    "ultra_vram": {
        "max_samples": 64,
        "epochs": 5,
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
    },
}
CFG = GPU_PROFILES[GPU_PROFILE]
CURRICULUM_EPOCHS = int(os.environ.get("NEXUS_COLAB_CURRICULUM_EPOCHS", "3"))
CURRICULUM_SMALL_ATOMS = int(os.environ.get("NEXUS_COLAB_CURRICULUM_SMALL_ATOMS", "20"))
CURRICULUM_MEDIUM_ATOMS = int(os.environ.get("NEXUS_COLAB_CURRICULUM_MEDIUM_ATOMS", "28"))

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


def _dataset_atom_counts(dataset) -> list[int]:
    """Works for both ZaretzkiMetabolicDataset and ConcatDataset."""
    if hasattr(dataset, "mols"):
        return [int(mol.GetNumAtoms()) for mol in dataset.mols]
    counts = []
    for ds in dataset.datasets:
        counts.extend(int(mol.GetNumAtoms()) for mol in ds.mols)
    return counts


def _curriculum_indices(
    atom_counts: list[int],
    *,
    epoch_idx: int,
    total_epochs: int,
) -> tuple[list[int], str]:
    total = len(atom_counts)
    all_indices = list(range(total))
    if total == 0 or total_epochs <= 1 or CURRICULUM_EPOCHS <= 0:
        return all_indices, f"all ({total})"

    sorted_indices = sorted(all_indices, key=lambda idx: (atom_counts[idx], idx))
    min_subset = min(total, max(16, total // 4))

    def _indices_below(max_atoms: int, fallback_fraction: float) -> list[int]:
        selected = [idx for idx in all_indices if atom_counts[idx] <= max_atoms]
        if len(selected) >= min_subset:
            return selected
        fallback = max(min_subset, int(round(total * fallback_fraction)))
        return sorted_indices[: min(total, fallback)]

    small_phase = min(CURRICULUM_EPOCHS, max(total_epochs - 2, 0))
    medium_phase = min(total_epochs - 1, small_phase + 1) if total_epochs > 1 else 0

    if epoch_idx < small_phase:
        selected = _indices_below(CURRICULUM_SMALL_ATOMS, 0.50)
        return selected, f"<= {CURRICULUM_SMALL_ATOMS} atoms ({len(selected)})"
    if epoch_idx < medium_phase:
        selected = _indices_below(CURRICULUM_MEDIUM_ATOMS, 0.75)
        return selected, f"<= {CURRICULUM_MEDIUM_ATOMS} atoms ({len(selected)})"
    return all_indices, f"all ({total})"


def _make_loader(dataset, indices, *, shuffle: bool, device: torch.device) -> DataLoader:
    subset = dataset if len(indices) == len(dataset) else Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=geometric_collate_fn,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}\n")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

# ── dataset — all 9 CYP isoforms ───────────────────────────────────────────
_sub_datasets = []
for _sdf in _ALL_SDFS:
    try:
        _sub_datasets.append(ZaretzkiMetabolicDataset(_sdf, max_molecules=CFG["max_samples"]))
    except Exception as _sdf_err:
        print(f"  Skipping {_sdf.name}: {_sdf_err}")
dataset = ConcatDataset(_sub_datasets) if len(_sub_datasets) > 1 else _sub_datasets[0]
atom_counts = _dataset_atom_counts(dataset)
epoch_indices = [
    _curriculum_indices(atom_counts, epoch_idx=epoch, total_epochs=CFG["epochs"])
    for epoch in range(CFG["epochs"])
]
loader = _make_loader(dataset, list(range(len(dataset))), shuffle=True, device=device)
print(f"Dataset : {len(dataset)} molecules")
print(f"Profile : {GPU_PROFILE}  (override: NEXUS_COLAB_GPU_PROFILE=ultra_vram|high_vram|standard)")
print(f"Physics mode : {CFG['physics_mode']}")
if CFG["epochs"] > 1:
    print("Curriculum : " + " | ".join(
        f"epoch {epoch+1} {label}" for epoch, (_, label) in enumerate(epoch_indices)
    ))

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
    enable_static_compile=False,  # torch.compile is unsafe on Colab
    use_galore=False,             # plain AdamW — avoids GaLore SVD on Colab
).to(device)

# ── Colab-safe quantum grid ────────────────────────────────────────────────
qe = trainer.model.module1.field_engine.quantum_enforcer
qe.integration_resolution = CFG["integration_resolution"]
qe.integration_chunk_size = CFG["integration_chunk"]
print(
    f"Quantum grid : {CFG['integration_resolution']}^3 = {CFG['integration_resolution']**3} pts,  "
    f"chunk={CFG['integration_chunk']}"
)

# ── Colab-safe reaction-volume scanner ────────────────────────────────────
# DEFAULT: 96 pts × 5 shells × 5 refine steps = ~2400 pts/atom w/ gradients
# COLAB  :  8 pts × 2 shells × 1 refine step  =   16 pts/atom  (150× less)
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

# ── Colab-safe navigator budget ────────────────────────────────────────────
# DEFAULT: optimization_steps=6, candidate_batch=8 → (6+1+9)=16 trajectories/atom
# COLAB  : optimization_steps=3, candidate_batch=4 → (3+1+5)= 9 trajectories/atom
# Each trajectory runs the full CliffordLieIntegrator (4 force calls per RK4 step).
nav = trainer.model.navigator
nav.optimization_steps = CFG.get("nav_opt_steps", 3)
nav.candidate_batch    = CFG.get("nav_candidates", 4)
print(
    f"Navigator    : opt_steps={nav.optimization_steps}  "
    f"candidates={nav.candidate_batch}  "
    f"(total trajectories/atom ≈ {nav.optimization_steps + 1 + nav.candidate_batch + 1})"
)

# ── memory bank — ALL labeled molecules from ALL 9 CYP isoforms (uncapped) ──
# The bank uses only ECFP4 fingerprints, so loading all molecules is cheap.
# BaselineMemoryBank.identity_threshold=0.999 automatically prevents any
# training molecule from trivially retrieving itself at query time.
print("Populating memory bank from all CYP isoform data (full, uncapped)...")
_bank_mols = []
for _sdf in _ALL_SDFS:
    try:
        _bank_ds = ZaretzkiMetabolicDataset(_sdf, max_molecules=0)  # 0 = all molecules
        _bank_mols.extend(_bank_ds.mols)
    except Exception as _e:
        print(f"  Skipping {_sdf.name} for bank: {_e}")
trainer.memory_bank.populate_from_mols(_bank_mols)
print(f"Memory bank ready: {len(trainer.memory_bank.historical_mols)} molecules.\n")
del _bank_mols, _bank_ds

total_training_steps = sum(max(len(indices), 1) for indices, _ in epoch_indices)
trainer.set_total_training_steps(total_training_steps)
trainer.configure_optimizers()
if device.type == "cuda":
    torch.cuda.empty_cache()
print("Optimizer ready.\n")

# ── checkpoint resume ───────────────────────────────────────────────────────
start_epoch = 0
history = []
if CKPT_PATH.exists():
    print(f"Loading checkpoint from {CKPT_PATH} ...")
    _ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    trainer.load_state_dict(_ckpt["model_state_dict"], strict=False)
    if "optimizer_state_dict" in _ckpt and trainer.optimizer is not None:
        try:
            trainer.optimizer.load_state_dict(_ckpt["optimizer_state_dict"])
            print("  Optimizer state restored.")
        except Exception as _oe:
            print(f"  Optimizer state not restored (shape mismatch after arch change): {_oe}")
    start_epoch = int(_ckpt.get("epoch", 0))
    history = list(_ckpt.get("metrics_history", []))
    print(f"  Resumed from epoch {start_epoch}  ({len(history)} epochs completed)\n")
else:
    print(f"No checkpoint at {CKPT_PATH} — starting fresh.\n")


# ── training loop ──────────────────────────────────────────────────────────
for epoch in range(start_epoch, CFG["epochs"]):
    indices, label = epoch_indices[epoch]
    loader = _make_loader(dataset, indices, shuffle=True, device=device)
    print(f"Epoch {epoch+1}/{CFG['epochs']} curriculum : {label}")
    metrics = trainer.fit_epoch(loader, train=True, log_every=1)
    history.append(metrics)
    print(f"\n── epoch {epoch+1} summary ──────────────────────────────────────")
    for _k in ["loss_total", "som_top1", "som_top2", "pred_rate",
               "hamiltonian_initial", "dag_causal_loss",
               "ana_loss_total", "ana_gate_open", "ana_confidence",
               "ana_peak", "ana_gate_conf_ok", "ana_gate_peak_ok",
               "ana_weight_fp", "ana_weight_ana", "ana_transport_ok"]:
        if _k in metrics:
            print(f"  {_k:<30} {metrics[_k]:.6g}")
    print()

    # ── save checkpoint after every epoch ──────────────────────────────────
    _ckpt_payload: dict = {
        "epoch": epoch + 1,
        "model_state_dict": {k: v.cpu() for k, v in trainer.state_dict().items()},
        "metrics_history": history,
    }
    if trainer.optimizer is not None:
        _ckpt_payload["optimizer_state_dict"] = trainer.optimizer.state_dict()
    torch.save(_ckpt_payload, CKPT_PATH)
    print(f"  Checkpoint saved → {CKPT_PATH}")

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
