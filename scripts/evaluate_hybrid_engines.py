from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
from enzyme_software.liquid_nn_v2._compat import require_torch, torch
from enzyme_software.liquid_nn_v2.data.cyp_classes import MAJOR_CYP_CLASSES
from enzyme_software.liquid_nn_v2.data.dataset_loader import collate_fn
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb import FullXTBHybridDataset
from enzyme_software.liquid_nn_v2.features.steric_features import StructureLibrary
from enzyme_software.liquid_nn_v2.features.xtb_features import load_or_compute_full_xtb_features
from enzyme_software.liquid_nn_v2.training.metrics import compute_cyp_metrics, compute_site_metrics_v2
from enzyme_software.liquid_nn_v2.training.utils import move_to_device


DEFAULT_MODES = ("final", "base_lnn", "lnn_vote", "wave_vote", "analogical_vote")


def _load_rows(path: Path) -> List[dict]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return list(payload.get("drugs", payload))
    return list(payload)


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


def _split_by_wave_valid(rows: List[dict], cache_dir: Path) -> tuple[List[dict], List[dict], Dict[str, int]]:
    valid_rows: List[dict] = []
    missing_rows: List[dict] = []
    statuses: Counter[str] = Counter()
    for row in rows:
        smiles = str(row.get("smiles", "")).strip()
        payload = load_or_compute_full_xtb_features(smiles, cache_dir=cache_dir, compute_if_missing=False)
        status = str(payload.get("status") or "unknown")
        statuses[status] += 1
        if bool(payload.get("xtb_valid")):
            valid_rows.append(row)
        else:
            missing_rows.append(row)
    return valid_rows, missing_rows, dict(sorted(statuses.items()))


def _make_loader(
    rows: List[dict],
    *,
    structure_sdf: str | None,
    xtb_cache_dir: str,
    batch_size: int,
):
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
    state_dict = payload.get("model_state_dict") or payload
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, payload


def _safe_sigmoid(value: torch.Tensor | None, fallback: torch.Tensor) -> torch.Tensor:
    if value is None:
        return fallback
    return torch.sigmoid(torch.nan_to_num(value, nan=0.0, posinf=20.0, neginf=-20.0))


def _score_tensors_for_modes(outputs: Dict[str, object], modes: Iterable[str]) -> Dict[str, torch.Tensor]:
    site_logits = outputs["site_logits"]
    fallback = torch.sigmoid(site_logits)
    vote_heads = outputs.get("site_vote_heads") or {}
    base_logits = outputs.get("site_logits_base", site_logits)
    available = {
        "final": fallback,
        "base_lnn": _safe_sigmoid(base_logits, fallback),
        "lnn_vote": _safe_sigmoid(vote_heads.get("lnn_vote"), _safe_sigmoid(base_logits, fallback)),
        "wave_vote": _safe_sigmoid(vote_heads.get("wave_vote"), torch.zeros_like(fallback)),
        "analogical_vote": _safe_sigmoid(vote_heads.get("analogical_vote"), torch.zeros_like(fallback)),
        "council_only": _safe_sigmoid(vote_heads.get("council_logit"), torch.zeros_like(fallback)),
    }
    return {mode: available[mode] for mode in modes if mode in available}


def _evaluate_rows(
    rows: List[dict],
    *,
    name: str,
    modes: List[str],
    model,
    device,
    structure_sdf: str | None,
    xtb_cache_dir: str,
    batch_size: int,
) -> Dict[str, object]:
    require_torch()
    dataset, loader = _make_loader(
        rows,
        structure_sdf=structure_sdf,
        xtb_cache_dir=xtb_cache_dir,
        batch_size=batch_size,
    )
    site_scores_by_mode: Dict[str, List[torch.Tensor]] = {mode: [] for mode in modes}
    site_labels: List[torch.Tensor] = []
    site_masks: List[torch.Tensor] = []
    site_batch: List[torch.Tensor] = []
    cyp_logits: List[torch.Tensor] = []
    cyp_labels: List[torch.Tensor] = []
    cyp_masks: List[torch.Tensor] = []
    batch_offset = 0

    with torch.no_grad():
        for raw_batch in loader:
            if raw_batch is None:
                continue
            batch = move_to_device(raw_batch, device)
            outputs = model(batch)
            mode_scores = _score_tensors_for_modes(outputs, modes)
            for mode, scores in mode_scores.items():
                site_scores_by_mode[mode].append(scores.detach().cpu())
            site_labels.append(batch["site_labels"].detach().cpu())
            site_masks.append(
                batch.get("site_supervision_mask", torch.ones_like(batch["site_labels"])).detach().cpu()
            )
            site_batch.append(batch["batch"].detach().cpu() + batch_offset)
            cyp_logits.append(outputs["cyp_logits"].detach().cpu())
            cyp_labels.append(batch["cyp_labels"].detach().cpu())
            cyp_masks.append(
                batch.get("cyp_supervision_mask", torch.ones_like(batch["cyp_labels"])).detach().cpu()
            )
            batch_offset += int(batch["cyp_labels"].shape[0])

    if not site_labels:
        return {
            "name": name,
            "rows": len(rows),
            "effective_total": 0,
            "invalid_count": len(rows),
            "invalid_reasons": dict(sorted((getattr(dataset, "_invalid_reasons", {}) or {}).items())),
            "site_metrics_by_mode": {},
            "cyp_metrics": {},
        }

    merged_site_labels = torch.cat(site_labels, dim=0)
    merged_site_masks = torch.cat(site_masks, dim=0)
    merged_site_batch = torch.cat(site_batch, dim=0)
    merged_cyp_logits = torch.cat(cyp_logits, dim=0)
    merged_cyp_labels = torch.cat(cyp_labels, dim=0)
    merged_cyp_masks = torch.cat(cyp_masks, dim=0)

    site_metrics_by_mode: Dict[str, Dict[str, float]] = {}
    for mode in modes:
        merged_scores = torch.cat(site_scores_by_mode[mode], dim=0) if site_scores_by_mode[mode] else torch.zeros_like(merged_site_labels)
        metrics = compute_site_metrics_v2(
            merged_scores,
            merged_site_labels,
            merged_site_batch,
            supervision_mask=merged_site_masks,
        )
        metrics["mean_score"] = float(merged_scores.mean().item()) if merged_scores.numel() else 0.0
        metrics["active_fraction"] = float((merged_scores > 0.5).float().mean().item()) if merged_scores.numel() else 0.0
        site_metrics_by_mode[mode] = metrics

    return {
        "name": name,
        "rows": len(rows),
        "effective_total": int(getattr(dataset, "_valid_count", 0)),
        "invalid_count": int(len(rows) - int(getattr(dataset, "_valid_count", 0))),
        "invalid_reasons": dict(sorted((getattr(dataset, "_invalid_reasons", {}) or {}).items())),
        "site_metrics_by_mode": site_metrics_by_mode,
        "cyp_metrics": compute_cyp_metrics(
            merged_cyp_logits,
            merged_cyp_labels,
            supervision_mask=merged_cyp_masks,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate individual hybrid engines on benchmark datasets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", required=True, help="Comma-separated benchmark JSON paths")
    parser.add_argument("--xtb-cache-dir", default="cache/full_xtb")
    parser.add_argument("--structure-sdf", default="3D structures.sdf")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    xtb_cache_dir = str(Path(args.xtb_cache_dir))
    structure_sdf = args.structure_sdf
    if structure_sdf and not Path(structure_sdf).exists():
        structure_sdf = None
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]

    model, payload = _load_model(checkpoint_path, device)
    dataset_paths = [Path(part.strip()) for part in args.datasets.split(",") if part.strip()]

    report: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "xtb_cache_dir": xtb_cache_dir,
        "structure_sdf": structure_sdf,
        "modes": modes,
        "checkpoint_best_val_top1": payload.get("best_val_top1"),
        "checkpoint_best_val_monitor": payload.get("best_val_monitor"),
        "benchmarks": {},
    }

    print("=" * 60)
    print("HYBRID ENGINE BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"modes={modes}")
    print()

    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {dataset_path}")
        all_rows = [_make_row(row) for row in _load_rows(dataset_path)]
        all_rows = [row for row in all_rows if _has_site_labels(row) and _supports_cyp(row)]
        wave_valid_rows, wave_missing_rows, xtb_statuses = _split_by_wave_valid(all_rows, Path(xtb_cache_dir))

        print("-" * 60)
        print(dataset_path)
        print(f"rows={len(all_rows)} | wave_valid={len(wave_valid_rows)} | wave_missing={len(wave_missing_rows)}")
        print(f"xtb_statuses={xtb_statuses}")

        dataset_report: Dict[str, object] = {
            "rows": len(all_rows),
            "wave_valid_rows": len(wave_valid_rows),
            "wave_missing_rows": len(wave_missing_rows),
            "xtb_statuses": xtb_statuses,
        }
        subsets = [("all", all_rows), ("wave_valid", wave_valid_rows), ("wave_missing", wave_missing_rows)]
        for subset_name, subset_rows in subsets:
            subset_report = _evaluate_rows(
                subset_rows,
                name=subset_name,
                modes=modes,
                model=model,
                device=device,
                structure_sdf=structure_sdf,
                xtb_cache_dir=xtb_cache_dir,
                batch_size=args.batch_size,
            )
            dataset_report[subset_name] = subset_report
            print(f"  {subset_name}: effective={subset_report['effective_total']}")
            for mode in modes:
                metrics = subset_report["site_metrics_by_mode"].get(mode, {})
                print(
                    f"    {mode}: "
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
        print(f"Saved engine benchmark report: {output_path}")


if __name__ == "__main__":
    main()
