from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.dataset import split_drugs


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("drugs"), list):
        raise TypeError(f"Expected dataset payload with a 'drugs' list: {path}")
    return payload


def _family_from_row(row: dict[str, Any]) -> str:
    for key in ("target_family", "enzyme_family", "primary_family", "primary_cyp", "cyp"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "unknown"


def _source_family_for_name(source_name: Any) -> str:
    token = str(source_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if token in {"attnsom", "cyp_dbs_external"}:
        return "attnsom_family"
    if token == "metxbiodb":
        return "metxbiodb_family"
    if token == "peng_external":
        return "peng_family"
    if token == "rudik_external":
        return "rudik_family"
    return f"{token}_family" if token else "unknown_family"


def _with_training_regime(rows: list[dict[str, Any]], *, training_regime: str, target_family: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["training_regime"] = str(training_regime)
        updated.setdefault("target_family", target_family)
        updated.setdefault("enzyme_family", target_family)
        updated.setdefault("source_family", _source_family_for_name(updated.get("molecule_source") or updated.get("source")))
        output.append(updated)
    return output


def _make_payload(
    template: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    subset_name: str,
    split_name: str,
    build_metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(template)
    payload["drugs"] = rows
    payload["n_drugs"] = int(len(rows))
    payload["n_site_labeled"] = int(sum(1 for row in rows if list(row.get("all_labeled_site_atoms") or row.get("site_atoms") or [])))
    payload["summary"] = {
        **dict(template.get("summary") or {}),
        "subset_name": subset_name,
        "split_name": split_name,
        "row_count": int(len(rows)),
    }
    payload["build_stats"] = {
        **dict(template.get("build_stats") or {}),
        **build_metadata,
    }
    return payload


def _rows_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": int(len(rows)),
        "source_breakdown": dict(sorted(Counter(str(row.get("molecule_source") or row.get("source") or "unknown") for row in rows).items())),
        "source_family_breakdown": dict(sorted(Counter(str(row.get("source_family") or "unknown_family") for row in rows).items())),
        "label_regime_breakdown": dict(sorted(Counter(str(row.get("label_regime") or "unknown") for row in rows).items())),
        "training_regime_breakdown": dict(sorted(Counter(str(row.get("training_regime") or "unknown") for row in rows).items())),
        "multisite_count": int(sum(1 for row in rows if bool(row.get("is_multisite")))),
    }


def build_mainline_splits(
    *,
    strict_exact_input: Path,
    tiered_input: Path,
    output_dir: Path,
    dataset_prefix: str,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    split_mode: str = "scaffold_source_size",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    strict_payload = _load_payload(strict_exact_input)
    tiered_payload = _load_payload(tiered_input)
    target_family = _family_from_row((strict_payload.get("drugs") or tiered_payload.get("drugs") or [{}])[0])
    strict_rows = _with_training_regime(list(strict_payload.get("drugs") or []), training_regime="exact_clean", target_family=target_family)
    tiered_rows = _with_training_regime(list(tiered_payload.get("drugs") or []), training_regime="tiered_multisite", target_family=target_family)

    def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return split_drugs(
            rows,
            seed=int(seed),
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            mode=str(split_mode),
        )

    strict_train, strict_val, strict_test = split_rows(strict_rows)
    combined_rows = list(strict_rows) + list(tiered_rows)
    combo_train, combo_val, combo_test = split_rows(combined_rows)

    metadata = {
        "mainline_split_seed": int(seed),
        "mainline_split_mode": str(split_mode),
        "mainline_train_ratio": float(train_ratio),
        "mainline_val_ratio": float(val_ratio),
        "target_family": target_family,
    }
    outputs = {
        f"{dataset_prefix}_strict_exact_clean_train.json": _make_payload(strict_payload, strict_train, subset_name=f"{dataset_prefix}_strict_exact_clean", split_name="train", build_metadata=metadata),
        f"{dataset_prefix}_strict_exact_clean_val.json": _make_payload(strict_payload, strict_val, subset_name=f"{dataset_prefix}_strict_exact_clean", split_name="val", build_metadata=metadata),
        f"{dataset_prefix}_strict_exact_clean_test.json": _make_payload(strict_payload, strict_test, subset_name=f"{dataset_prefix}_strict_exact_clean", split_name="test", build_metadata=metadata),
        f"{dataset_prefix}_exact_plus_tiered_train.json": _make_payload(strict_payload, combo_train, subset_name=f"{dataset_prefix}_exact_plus_tiered", split_name="train", build_metadata=metadata),
        f"{dataset_prefix}_exact_plus_tiered_val.json": _make_payload(strict_payload, combo_val, subset_name=f"{dataset_prefix}_exact_plus_tiered", split_name="val", build_metadata=metadata),
        f"{dataset_prefix}_exact_plus_tiered_test.json": _make_payload(strict_payload, combo_test, subset_name=f"{dataset_prefix}_exact_plus_tiered", split_name="test", build_metadata=metadata),
    }
    for name, payload in outputs.items():
        (output_dir / name).write_text(json.dumps(payload, indent=2))

    for split_name, rows in (("val", combo_val), ("test", combo_test)):
        exact_rows = [row for row in rows if str(row.get("training_regime") or "") == "exact_clean"]
        tiered_split_rows = [row for row in rows if str(row.get("training_regime") or "") == "tiered_multisite"]
        (output_dir / f"{dataset_prefix}_exact_plus_tiered_{split_name}_exact_clean_eval.json").write_text(
            json.dumps(
                _make_payload(strict_payload, exact_rows, subset_name=f"{dataset_prefix}_exact_plus_tiered_exact_eval", split_name=split_name, build_metadata={"parent_subset": f"{dataset_prefix}_exact_plus_tiered"}),
                indent=2,
            )
        )
        (output_dir / f"{dataset_prefix}_exact_plus_tiered_{split_name}_tiered_eval.json").write_text(
            json.dumps(
                _make_payload(tiered_payload, tiered_split_rows, subset_name=f"{dataset_prefix}_exact_plus_tiered_tiered_eval", split_name=split_name, build_metadata={"parent_subset": f"{dataset_prefix}_exact_plus_tiered"}),
                indent=2,
            )
        )

    summary = {
        "target_family": target_family,
        "dataset_prefix": dataset_prefix,
        "input_strict_exact_dataset_path": str(strict_exact_input),
        "input_tiered_dataset_path": str(tiered_input),
        "seed": int(seed),
        "split_mode": str(split_mode),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "strict_exact_clean": {
            "total_rows": int(len(strict_rows)),
            "train": _rows_summary(strict_train),
            "val": _rows_summary(strict_val),
            "test": _rows_summary(strict_test),
        },
        "exact_plus_tiered": {
            "total_rows": int(len(combined_rows)),
            "exact_clean_row_count": int(len(strict_rows)),
            "tiered_multisite_row_count": int(len(tiered_rows)),
            "excluded_non_mainline_rows": int(0),
            "train": _rows_summary(combo_train),
            "val": _rows_summary(combo_val),
            "test": _rows_summary(combo_test),
        },
    }
    (output_dir / f"{dataset_prefix}_split_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
