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
from torch.utils.data import DataLoader

# ── NaN forensics: crash the instant any op produces NaN/inf ───────────────
# PyTorch will print the exact operation and stack trace where the first NaN
# is born.  REMOVE this line after the root cause is confirmed — it adds
# ~2–5× overhead and halts the run on the first NaN (no full epoch).
torch.autograd.set_detect_anomaly(True)
# ────────────────────────────────────────────────────────────────────────────

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

SDF = _REPO_DIR / "data/ATTNSOM/cyp_dataset/3A4.sdf"

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


def _detect_gpu_profile() -> str:
    env = os.environ.get("NEXUS_COLAB_GPU_PROFILE", "auto").strip().lower()
    normalized = _normalize_profile_name(env)
    if normalized in {"standard", "high_vram", "ultra_vram"}:
        return normalized
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / 1024**3
        print(f"GPU : {props.name}  |  total memory : {total_gb:.1f} GB")
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
    # ── 85+ GB (H100 SXM5, etc.) ─────────────────────────────────────────────
    # Full physics quality: CliffordLie dynamics rollout re-enabled (not skipped),
    # gradient checkpointing to keep activation memory bounded, all 5 scan
    # shells restored, and a higher-resolution quantum grid.
    # Measured baseline at high_vram: ~36 GB.  Ultra adds ~20-25 GB headroom
    # usage, landing well under 90 GB.
    "ultra_vram": {
        "max_samples": 64,
        "epochs": 5,
        "steps": 3,            # 3 Hamiltonian integration steps (was 1)
        "physics_mode": "full",
        "low_memory": False,   # full CliffordLie rollout (was skipped)
        "checkpoint": True,    # gradient checkpointing trades ~40% compute for ~50% less activation memory
        "integration_resolution": 14,   # 14³=2744 pts (was 1000)
        "integration_chunk": 256,
        "scan_n_points": 48,   # 4× more scan pts (was 12)
        "scan_radius": 2.5,    # full pocket radius (was 1.0)
        "scan_chunk": 24,
        "scan_shells": (0.35, 0.55, 0.75, 0.90, 1.00),   # all 5 shells (was 2)
        "scan_refine_steps": 2,
    },
}
CFG = GPU_PROFILES[GPU_PROFILE]

from nexus.data.metabolic_dataset import ZaretzkiMetabolicDataset, geometric_collate_fn
from nexus.training.causal_trainer import Metabolic_Causal_Trainer


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}\n")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

# ── dataset ────────────────────────────────────────────────────────────────
dataset = ZaretzkiMetabolicDataset(SDF, max_molecules=CFG["max_samples"])
loader  = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=geometric_collate_fn,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)
print(f"Dataset : {len(dataset)} molecules")
print(f"Profile : {GPU_PROFILE}  (override: NEXUS_COLAB_GPU_PROFILE=ultra_vram|high_vram|standard)")
print(f"Physics mode : {CFG['physics_mode']}")

# ── trainer ────────────────────────────────────────────────────────────────
trainer = Metabolic_Causal_Trainer(
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

print("Populating analogical memory bank from other isoform sources (excluding training molecules)...")
from rdkit import Chem as _Chem
_bank_mols = _collect_memory_bank_mols(SDF, dataset)
trainer.memory_bank.populate_from_mols(_bank_mols)
print(f"Memory bank ready: {len(trainer.memory_bank.historical_mols)} molecules.\n")
del _bank_mols

trainer.set_total_training_steps(CFG["epochs"] * max(len(loader), 1))
trainer.configure_optimizers()
if device.type == "cuda":
    torch.cuda.empty_cache()
print("Optimizer ready.\n")


# ── training loop ──────────────────────────────────────────────────────────
history = []
for epoch in range(CFG["epochs"]):
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

if device.type == "cuda":
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Peak GPU memory : {peak_mb:.1f} MB")

out = _REPO_DIR / "colab_train_metrics.json"
out.write_text(json.dumps(history, indent=2))
print(f"Metrics saved -> {out}")
