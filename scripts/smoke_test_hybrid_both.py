#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig, TrainingConfig
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb import create_full_xtb_dataloaders_from_drugs, split_drugs
from enzyme_software.liquid_nn_v2.features.xtb_features import FULL_XTB_FEATURE_DIM
from enzyme_software.liquid_nn_v2.training.trainer import Trainer
from scripts.train_hybrid_full_xtb import _has_site_labels, _load_xenosite_aux_entries, _resolve_device


def _load_bundle(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_drugs(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _build_loaders(train_drugs: list[dict], val_drugs: list[dict], test_drugs: list[dict], *, structure_sdf: str, xtb_cache_dir: str):
    common = dict(
        batch_size=8,
        structure_sdf=structure_sdf,
        manual_target_bond=None,
        manual_feature_cache_dir=None,
        full_xtb_cache_dir=xtb_cache_dir,
        compute_full_xtb_if_missing=False,
        drop_failed=True,
    )
    try:
        return create_full_xtb_dataloaders_from_drugs(train_drugs, val_drugs, test_drugs, use_manual_engine_features=True, **common), True
    except RuntimeError as exc:
        if "zero valid graphs" not in str(exc):
            raise
        return create_full_xtb_dataloaders_from_drugs(train_drugs, val_drugs, test_drugs, use_manual_engine_features=False, **common), False


def main() -> None:
    require_torch()
    parser = argparse.ArgumentParser(description="Lightweight smoke test for the hybrid+both path.")
    parser.add_argument("--bundle", default="data/hybrid_both_bundle/bundle.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--xtb-cache-dir", default="")
    parser.add_argument("--disable-nexus-bridge", action="store_true")
    parser.add_argument("--freeze-nexus-memory", action="store_true")
    parser.add_argument("--skip-nexus-memory-rebuild", action="store_true")
    args = parser.parse_args()

    bundle_path = ROOT / args.bundle
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    bundle = _load_bundle(bundle_path)
    base_dataset = ROOT / bundle["paths"]["base_dataset"]
    xenosite_manifest = ROOT / bundle["paths"]["xenosite_manifest"]
    structure_sdf = str(ROOT / bundle["paths"]["structure_sdf"])
    xtb_cache_dir = args.xtb_cache_dir or bundle["defaults"]["xtb_cache_dir"]
    if not Path(xtb_cache_dir).is_absolute():
        xtb_cache_dir = str(ROOT / xtb_cache_dir)

    drugs = [row for row in _load_drugs(base_dataset) if _has_site_labels(row)]
    random.Random(args.seed).shuffle(drugs)
    drugs = drugs[: max(24, int(args.limit))]
    train_drugs, val_drugs, test_drugs = split_drugs(drugs, seed=args.seed, train_ratio=0.7, val_ratio=0.15)
    xenosite_entries = _load_xenosite_aux_entries(xenosite_manifest, topk=int(bundle["defaults"]["xenosite_topk"]), per_file_limit=8)
    existing = {str(row.get("smiles", "")).strip() for row in train_drugs}
    train_drugs.extend([row for row in xenosite_entries if str(row.get("smiles", "")).strip() not in existing][:8])

    (train_loader, val_loader, test_loader), manual_engine_enabled = _build_loaders(
        train_drugs,
        val_drugs,
        test_drugs,
        structure_sdf=structure_sdf,
        xtb_cache_dir=xtb_cache_dir,
    )

    manual_atom_feature_dim = (32 if manual_engine_enabled else 0) + FULL_XTB_FEATURE_DIM
    full_xtb_atom_input_dim = 140 + FULL_XTB_FEATURE_DIM
    base_config = ModelConfig.light_advanced(
        use_manual_engine_priors=manual_engine_enabled,
        use_3d_branch=True,
        use_nexus_bridge=not bool(args.disable_nexus_bridge),
        nexus_memory_frozen=bool(args.freeze_nexus_memory),
        nexus_rebuild_memory_before_train=not bool(args.skip_nexus_memory_rebuild),
        return_intermediate_stats=True,
        manual_atom_feature_dim=manual_atom_feature_dim,
        atom_input_dim=full_xtb_atom_input_dim,
    )
    model = HybridLNNModel(LiquidMetabolismNetV2(base_config))
    trainer = Trainer(
        model=model,
        config=TrainingConfig(epochs=1, batch_size=8, learning_rate=2e-4, weight_decay=1e-4, early_stopping_patience=0),
        device=_resolve_device(args.device),
    )

    memory_stats = {"used": 0.0, "memory_size": 0.0}
    if (
        getattr(base_config, "use_nexus_bridge", False)
        and getattr(base_config, "nexus_rebuild_memory_before_train", False)
        and getattr(model, "nexus_bridge", None) is not None
    ):
        memory_stats = model.rebuild_nexus_memory(train_loader, device=trainer.device, max_batches=2)

    raw_batch = next(iter(train_loader))
    batch = trainer._prepare_batch(raw_batch)
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        loss, stats = trainer.compute_loss(batch, outputs)

    bridge = outputs.get("nexus_bridge_outputs") or {}
    wave = bridge.get("wave_predictions") if isinstance(bridge, dict) else None
    summary = {
        "lnn_ok": bool(torch.isfinite(outputs["site_logits"]).all() and torch.isfinite(outputs["cyp_logits"]).all()),
        "loss_ok": bool(torch.isfinite(loss)),
        "bridge_enabled": bool(getattr(model, "nexus_bridge", None) is not None),
        "wave_ok": bool(wave is not None and torch.isfinite(wave["predicted_fukui"]).all()),
        "analogical_ok": bool(
            isinstance(bridge, dict)
            and "analogical_site_bias" in bridge
            and torch.isfinite(bridge["analogical_site_bias"]).all()
        ),
        "memory_size": float(memory_stats.get("memory_size", 0.0)),
        "manual_engine_enabled": bool(manual_engine_enabled),
        "nexus_bridge_loss": float(stats.get("nexus_bridge_loss", 0.0)),
        "nexus_wave_loss": float(stats.get("nexus_wave_loss", 0.0)),
        "nexus_analogical_loss": float(stats.get("nexus_analogical_loss", 0.0)),
        "analogical_confidence_mean": float(bridge.get("metrics", {}).get("analogical_confidence_mean", 0.0)) if isinstance(bridge, dict) else 0.0,
    }
    print(json.dumps(summary, indent=2))
    if not (summary["lnn_ok"] and summary["loss_ok"] and (not summary["bridge_enabled"] or (summary["wave_ok"] and summary["analogical_ok"]))):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
