from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(SRC), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.dataset import split_drugs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 1/Phase 2 CYP3A4 split artifacts from downstream subset datasets."
    )
    parser.add_argument(
        "--strict-exact-input",
        default="data/prepared_training/cyp3a4_downstream_subsets_stricter/cyp3a4_strict_exact_clean.json",
    )
    parser.add_argument(
        "--tiered-input",
        default="data/prepared_training/cyp3a4_downstream_subsets_stricter/cyp3a4_tiered_multisite_eval.json",
    )
    parser.add_argument(
        "--output-dir",
        default="data/prepared_training/cyp3a4_phase12_splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument(
        "--split-mode",
        choices=("random", "scaffold_source", "scaffold_source_size"),
        default="scaffold_source_size",
    )
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("drugs"), list):
        raise TypeError(f"Expected dataset payload with a 'drugs' list: {path}")
    return payload


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


def _with_training_regime(rows: list[dict[str, Any]], *, training_regime: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["training_regime"] = str(training_regime)
        updated.setdefault("source_family", _source_family_for_name(updated.get("molecule_source") or updated.get("source")))
        merged_sources = [str(v) for v in list(updated.get("merged_from_sources") or []) if str(v).strip()]
        updated.setdefault("merged_from_source_families", sorted({_source_family_for_name(v) for v in merged_sources}))
        out.append(updated)
    return out


def _make_payload(template: dict[str, Any], rows: list[dict[str, Any]], *, subset_name: str, split_name: str, build_metadata: dict[str, Any]) -> dict[str, Any]:
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


def _split_and_write(
    *,
    rows: list[dict[str, Any]],
    template: dict[str, Any],
    subset_stem: str,
    output_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    split_mode: str,
) -> dict[str, Any]:
    train_rows, val_rows, test_rows = split_drugs(
        rows,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        mode=split_mode,
    )
    build_metadata = {
        "phase12_split_seed": int(seed),
        "phase12_split_mode": str(split_mode),
        "phase12_train_ratio": float(train_ratio),
        "phase12_val_ratio": float(val_ratio),
    }
    split_payloads = {
        "train": _make_payload(template, train_rows, subset_name=subset_stem, split_name="train", build_metadata=build_metadata),
        "val": _make_payload(template, val_rows, subset_name=subset_stem, split_name="val", build_metadata=build_metadata),
        "test": _make_payload(template, test_rows, subset_name=subset_stem, split_name="test", build_metadata=build_metadata),
    }
    for split_name, payload in split_payloads.items():
        (output_dir / f"{subset_stem}_{split_name}.json").write_text(json.dumps(payload, indent=2))
    return {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
        "summary": {
            "subset_name": subset_stem,
            "seed": int(seed),
            "split_mode": str(split_mode),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
            "total_rows": int(len(rows)),
            "train": _rows_summary(train_rows),
            "val": _rows_summary(val_rows),
            "test": _rows_summary(test_rows),
        },
    }


def main() -> None:
    args = _parse_args()
    strict_input = Path(args.strict_exact_input)
    tiered_input = Path(args.tiered_input)
    if not strict_input.is_absolute():
        strict_input = (ROOT / strict_input).resolve()
    if not tiered_input.is_absolute():
        tiered_input = (ROOT / tiered_input).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_payload = _load_payload(strict_input)
    tiered_payload = _load_payload(tiered_input)
    strict_rows = _with_training_regime(list(strict_payload.get("drugs") or []), training_regime="exact_clean")
    tiered_rows = _with_training_regime(list(tiered_payload.get("drugs") or []), training_regime="tiered_multisite")

    phase1 = _split_and_write(
        rows=strict_rows,
        template=strict_payload,
        subset_stem="cyp3a4_strict_exact_clean",
        output_dir=output_dir,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        split_mode=str(args.split_mode),
    )

    combined_rows = list(strict_rows) + list(tiered_rows)
    phase2 = _split_and_write(
        rows=combined_rows,
        template=strict_payload,
        subset_stem="cyp3a4_exact_plus_tiered",
        output_dir=output_dir,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        split_mode=str(args.split_mode),
    )

    phase2_exact_excluded = 0
    phase2_tiered_excluded = 0
    phase2_broad_conflict_excluded = int(
        max(
            0,
            int(strict_payload.get("n_drugs", len(strict_rows)))
            + int(tiered_payload.get("n_drugs", len(tiered_rows)))
            - int(len(combined_rows)),
        )
    )

    for split_name in ("val", "test"):
        split_rows = list(phase2[split_name])
        exact_rows = [row for row in split_rows if str(row.get("training_regime") or "") == "exact_clean"]
        tiered_split_rows = [row for row in split_rows if str(row.get("training_regime") or "") == "tiered_multisite"]
        exact_payload = _make_payload(
            strict_payload,
            exact_rows,
            subset_name="cyp3a4_exact_plus_tiered_exact_eval",
            split_name=split_name,
            build_metadata={"phase12_parent_subset": "cyp3a4_exact_plus_tiered"},
        )
        tiered_eval_payload = _make_payload(
            tiered_payload,
            tiered_split_rows,
            subset_name="cyp3a4_exact_plus_tiered_tiered_eval",
            split_name=split_name,
            build_metadata={"phase12_parent_subset": "cyp3a4_exact_plus_tiered"},
        )
        (output_dir / f"cyp3a4_exact_plus_tiered_{split_name}_exact_clean_eval.json").write_text(json.dumps(exact_payload, indent=2))
        (output_dir / f"cyp3a4_exact_plus_tiered_{split_name}_tiered_eval.json").write_text(json.dumps(tiered_eval_payload, indent=2))
        phase2_exact_excluded += max(0, len(split_rows) - len(exact_rows))
        phase2_tiered_excluded += max(0, len(split_rows) - len(tiered_split_rows))

    strict_summary = {
        **phase1["summary"],
        "input_dataset_path": str(strict_input),
        "scaffold_split_policy": str(args.split_mode),
    }
    exact_plus_tiered_summary = {
        **phase2["summary"],
        "input_strict_exact_dataset_path": str(strict_input),
        "input_tiered_dataset_path": str(tiered_input),
        "scaffold_split_policy": str(args.split_mode),
        "exact_clean_row_count": int(len(strict_rows)),
        "tiered_multisite_row_count": int(len(tiered_rows)),
        "broad_conflict_rows_excluded_count": int(phase2_broad_conflict_excluded),
        "phase2_eval_artifacts": {
            "val_exact_clean": str(output_dir / "cyp3a4_exact_plus_tiered_val_exact_clean_eval.json"),
            "test_exact_clean": str(output_dir / "cyp3a4_exact_plus_tiered_test_exact_clean_eval.json"),
            "val_tiered_eval": str(output_dir / "cyp3a4_exact_plus_tiered_val_tiered_eval.json"),
            "test_tiered_eval": str(output_dir / "cyp3a4_exact_plus_tiered_test_tiered_eval.json"),
        },
    }

    (output_dir / "cyp3a4_strict_exact_clean_split_summary.json").write_text(json.dumps(strict_summary, indent=2))
    (output_dir / "cyp3a4_exact_plus_tiered_split_summary.json").write_text(json.dumps(exact_plus_tiered_summary, indent=2))

    print(
        "Built Phase 1/2 CYP3A4 split artifacts | "
        f"strict_exact_clean={len(strict_rows)} | "
        f"tiered_multisite_eval={len(tiered_rows)} | "
        f"combined={len(combined_rows)}",
        flush=True,
    )
    print(f"Phase 1 summary: {output_dir / 'cyp3a4_strict_exact_clean_split_summary.json'}", flush=True)
    print(f"Phase 2 summary: {output_dir / 'cyp3a4_exact_plus_tiered_split_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
