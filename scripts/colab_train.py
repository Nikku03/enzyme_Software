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
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── ensure repo root is on sys.path ────────────────────────────────────────
_REPO_DIR = Path("/content/enzyme_Software")
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

# ── force-reload trainer so git-pulled changes always take effect ──────────
import nexus.training.causal_trainer as _ct_mod          # noqa: E402
import nexus.data.metabolic_dataset as _ds_mod           # noqa: E402
import nexus.field.query_engine as _qe_mod               # noqa: E402
importlib.reload(_qe_mod)
importlib.reload(_ct_mod)
importlib.reload(_ds_mod)

SDF = _REPO_DIR / "data/ATTNSOM/cyp_dataset/3A4.sdf"

def _detect_gpu_profile() -> str:
    env = os.environ.get("NEXUS_COLAB_GPU_PROFILE", "auto").strip().lower()
    if env in {"a100", "h100"}:
        return env
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).upper()
        if "H100" in name:
            return "h100"
    return "a100"


GPU_PROFILE = _detect_gpu_profile()
GPU_PROFILES = {
    "a100": {
        "max_samples": 16,
        "epochs": 1,
        "steps": 1,
        "integration_resolution": 8,
        "integration_chunk": 32,
        "scan_n_points": 8,
        "scan_radius": 1.0,
        "scan_chunk": 2,
        "scan_shells": (0.5, 1.0),
        "scan_refine_steps": 0,
    },
    "h100": {
        "max_samples": 16,
        "epochs": 1,
        "steps": 1,
        "integration_resolution": 8,
        "integration_chunk": 64,
        "scan_n_points": 8,
        "scan_radius": 1.0,
        "scan_chunk": 4,
        "scan_shells": (0.5, 1.0),
        "scan_refine_steps": 0,
    },
}
CFG = GPU_PROFILES[GPU_PROFILE]

from nexus.data.metabolic_dataset import ZaretzkiMetabolicDataset, geometric_collate_fn
from nexus.training.causal_trainer import Metabolic_Causal_Trainer

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
print(f"Profile : {GPU_PROFILE}")

# ── trainer ────────────────────────────────────────────────────────────────
trainer = Metabolic_Causal_Trainer(
    dynamics_steps=CFG["steps"],
    dynamics_dt=0.001,
    checkpoint_dynamics=False,
    enable_wsd_scheduler=True,
    low_memory_train_mode=True,   # skip full CliffordLie rollout
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

trainer.set_total_training_steps(CFG["epochs"] * max(len(loader), 1))
trainer.configure_optimizers()
if device.type == "cuda":
    torch.cuda.empty_cache()
print("Optimizer ready.\n")


# ── device transfer helper ─────────────────────────────────────────────────
def _to(obj, dev):
    if torch.is_tensor(obj):   return obj.to(dev)
    if isinstance(obj, dict):  return {k: _to(v, dev) for k, v in obj.items()}
    if isinstance(obj, list):  return [_to(v, dev) for v in obj]
    if isinstance(obj, tuple): return tuple(_to(v, dev) for v in obj)
    return obj


# ── training loop ──────────────────────────────────────────────────────────
history = []
for epoch in range(CFG["epochs"]):
    reducer: dict = {}
    skipped = 0
    trainer.train(True)
    total = len(loader)
    for i, batch in enumerate(loader, 1):
        try:
            batch = _to(batch, device)
            m = trainer.training_step(batch)
        except Exception as exc:
            skipped += 1
            print(f"  [batch {i}/{total}] SKIPPED — {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        if m is None:
            skipped += 1
            print(f"  [batch {i}/{total}] SKIPPED — non-finite loss after trainer safeguards", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        for k, v in m.items():
            try:
                val = float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
                reducer.setdefault(k, []).append(val)
            except Exception:
                pass
        running = {k: sum(vs) / len(vs) for k, vs in reducer.items()}
        print(
            f"epoch={epoch+1}  batch={i}/{total}"
            f"  loss={running.get('loss_total', float('nan')):.4g}"
            f"  pred_rate={running.get('pred_rate', float('nan')):.4g}"
            f"  dag_loss={running.get('dag_causal_loss', float('nan')):.4g}",
            flush=True,
        )
    metrics = {k: sum(vs) / len(vs) for k, vs in reducer.items()}
    history.append(metrics)
    print(f"\n── epoch {epoch+1} done  (skipped={skipped}/{total}): {metrics}\n")

if device.type == "cuda":
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Peak GPU memory : {peak_mb:.1f} MB")

out = _REPO_DIR / "colab_train_metrics.json"
out.write_text(json.dumps(history, indent=2))
print(f"Metrics saved -> {out}")
