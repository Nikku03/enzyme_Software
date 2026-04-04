from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig, TrainingConfig
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb import FullXTBHybridDataset, load_full_xtb_warm_start
from enzyme_software.liquid_nn_v2.features.steric_features import StructureLibrary
from enzyme_software.liquid_nn_v2.features.xtb_features import (
    load_or_compute_full_xtb_features,
    payload_training_xtb_valid,
    payload_true_xtb_valid,
)
from enzyme_software.liquid_nn_v2.training.trainer import Trainer
from enzyme_software.liquid_nn_v2.data.dataset_loader import collate_fn


def _require_rdkit() -> None:
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "RDKit is required for benchmark evaluation. In Colab, install it before running "
            "the notebook, e.g. `pip install rdkit-pypi`."
        ) from exc


def _load_rows(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return list(payload.get("drugs", payload))
    return list(payload)


def _canonicalize_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return " ".join(str(smiles or "").split())


def _resolve_device(name: str | None):
    require_torch()
    if name:
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _has_site_labels(drug: dict) -> bool:
    return bool(drug.get("som") or drug.get("site_atoms") or drug.get("site_atom_indices") or drug.get("metabolism_sites"))


def _supports_cyp(drug: dict) -> bool:
    cyp = str(drug.get("primary_cyp") or drug.get("cyp") or "").strip()
    return cyp in set(MAJOR_CYP_CLASSES)


def _make_row(drug: dict) -> dict:
    row = dict(drug)
    row.setdefault("source", "benchmark")
    row.setdefault("confidence", "validated")
    if "cyp" not in row and row.get("primary_cyp"):
        row["cyp"] = row["primary_cyp"]
    return row


def _split_by_wave_valid(rows: List[dict], cache_dir: Path) -> tuple[List[dict], List[dict], List[dict], Dict[str, int]]:
    strict_valid_rows: List[dict] = []
    training_usable_rows: List[dict] = []
    missing_rows: List[dict] = []
    statuses: Counter[str] = Counter()
    for row in rows:
        smiles = str(row.get("smiles", "")).strip()
        payload = load_or_compute_full_xtb_features(smiles, cache_dir=cache_dir, compute_if_missing=False)
        status = str(payload.get("status") or "unknown")
        statuses[status] += 1
        if payload_true_xtb_valid(payload):
            strict_valid_rows.append(row)
            training_usable_rows.append(row)
        elif payload_training_xtb_valid(payload):
            training_usable_rows.append(row)
        else:
            missing_rows.append(row)
    return strict_valid_rows, training_usable_rows, missing_rows, dict(sorted(statuses.items()))


def _make_loader(
    rows: List[dict],
    *,
    structure_sdf: str | None,
    xtb_cache_dir: str,
    batch_size: int,
) :
    require_torch()
    structure_library = StructureLibrary.from_sdf(structure_sdf) if structure_sdf else None
    dataset = FullXTBHybridDataset(
        split="test",
        augment=False,
        drugs=[_make_row(row) for row in rows],
        structure_library=structure_library,
        use_manual_engine_features=True,
        full_xtb_cache_dir=xtb_cache_dir,
        compute_full_xtb_if_missing=False,
        drop_failed=True,
    )
    dataset.precompute()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    loader._split_name = "benchmark"
    return dataset, loader


def _load_model(checkpoint_path: Path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_cfg_payload = payload.get("config", {}).get("base_model")
    if base_cfg_payload:
        base_config = ModelConfig(**base_cfg_payload)
    else:
        base_config = ModelConfig.light_advanced(
            use_manual_engine_priors=True,
            use_3d_branch=True,
            return_intermediate_stats=True,
        )
    model = HybridLNNModel(LiquidMetabolismNetV2(base_config))
    load_full_xtb_warm_start(
        model,
        checkpoint_path,
        device=device,
        new_manual_atom_dim=int(getattr(base_config, "manual_atom_feature_dim", 40)),
        new_atom_input_dim=int(getattr(base_config, "atom_input_dim", 148)),
    )
    model.to(device)
    model.eval()
    return model, payload


def _evaluate_rows(
    rows: List[dict],
    *,
    name: str,
    model,
    device,
    structure_sdf: str | None,
    xtb_cache_dir: str,
    batch_size: int,
) -> Dict[str, object]:
    dataset, loader = _make_loader(
        rows,
        structure_sdf=structure_sdf,
        xtb_cache_dir=xtb_cache_dir,
        batch_size=batch_size,
    )
    trainer = Trainer(model=model, config=TrainingConfig(), device=device)
    metrics = trainer.evaluate_loader(loader)
    return {
        "name": name,
        "rows": len(rows),
        "effective_total": int(getattr(dataset, "_valid_count", 0)),
        "invalid_count": int(len(rows) - int(getattr(dataset, "_valid_count", 0))),
        "invalid_reasons": dict(sorted((getattr(dataset, "_invalid_reasons", {}) or {}).items())),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid_full_xtb checkpoint on benchmark datasets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated benchmark JSON paths",
    )
    parser.add_argument("--xtb-cache-dir", default="cache/full_xtb")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    _require_rdkit()
    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    xtb_cache_dir = str(Path(args.xtb_cache_dir))
    structure_sdf = args.structure_sdf
    if structure_sdf and not Path(structure_sdf).exists():
        structure_sdf = None

    model, payload = _load_model(checkpoint_path, device)
    dataset_paths = [Path(part.strip()) for part in args.datasets.split(",") if part.strip()]

    report: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "xtb_cache_dir": xtb_cache_dir,
        "structure_sdf": structure_sdf,
        "checkpoint_best_val_top1": payload.get("best_val_top1"),
        "checkpoint_best_val_monitor": payload.get("best_val_monitor"),
        "wave_valid_definition": "strict_true_xtb_valid_only",
        "wave_training_usable_definition": "strict_true_xtb_valid_or_training_usable_cached_xtb",
        "benchmarks": {},
    }

    print("=" * 60)
    print("HYBRID BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"xtb_cache_dir={xtb_cache_dir}")
    print(f"structure_sdf={structure_sdf}")
    print()

    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {dataset_path}")
        all_rows = [_make_row(row) for row in _load_rows(dataset_path)]
        all_rows = [row for row in all_rows if _has_site_labels(row) and _supports_cyp(row)]
        wave_valid_rows, wave_training_usable_rows, wave_missing_rows, xtb_statuses = _split_by_wave_valid(all_rows, Path(xtb_cache_dir))

        print("-" * 60)
        print(dataset_path)
        print(
            f"rows={len(all_rows)} | wave_strict_valid={len(wave_valid_rows)} | "
            f"wave_training_usable={len(wave_training_usable_rows)} | wave_missing={len(wave_missing_rows)}"
        )
        print(f"xtb_statuses={xtb_statuses}")

        dataset_report: Dict[str, object] = {
            "rows": len(all_rows),
            "wave_valid_rows": len(wave_valid_rows),
            "wave_training_usable_rows": len(wave_training_usable_rows),
            "wave_missing_rows": len(wave_missing_rows),
            "xtb_statuses": xtb_statuses,
        }

        subsets = [
            ("all", all_rows),
            ("wave_valid", wave_valid_rows),
            ("wave_training_usable", wave_training_usable_rows),
            ("wave_missing", wave_missing_rows),
        ]
        for subset_name, subset_rows in subsets:
            if not subset_rows:
                dataset_report[subset_name] = {
                    "name": subset_name,
                    "rows": 0,
                    "effective_total": 0,
                    "invalid_count": 0,
                    "invalid_reasons": {},
                    "metrics": {},
                }
                print(f"  {subset_name}: rows=0")
                continue
            subset_report = _evaluate_rows(
                subset_rows,
                name=subset_name,
                model=model,
                device=device,
                structure_sdf=structure_sdf,
                xtb_cache_dir=xtb_cache_dir,
                batch_size=args.batch_size,
            )
            dataset_report[subset_name] = subset_report
            metrics = subset_report["metrics"]
            print(
                f"  {subset_name}: "
                f"effective={subset_report['effective_total']} "
                f"top1={metrics.get('site_top1_acc', 0.0):.4f} "
                f"top2={metrics.get('site_top2_acc', 0.0):.4f} "
                f"top3={metrics.get('site_top3_acc', 0.0):.4f} "
                f"auc={metrics.get('site_auc', 0.0):.4f}"
            )

        report["benchmarks"][str(dataset_path)] = dataset_report
        print()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved benchmark report: {output_path}")


if __name__ == "__main__":
    main()
