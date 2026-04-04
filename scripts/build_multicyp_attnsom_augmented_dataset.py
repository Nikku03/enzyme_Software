from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_cyp3a4_augmented_dataset import (
    ROOT,
    _normalize_row,
    load_astrazeneca_csv,
    load_attnsom_sdf,
    merge_rows,
)


ATTNSOM_CYP_MAP = {
    "1A2": "CYP1A2",
    "2A6": "CYP2A6",
    "2B6": "CYP2B6",
    "2C8": "CYP2C8",
    "2C9": "CYP2C9",
    "2C19": "CYP2C19",
    "2D6": "CYP2D6",
    "2E1": "CYP2E1",
    "3A4": "CYP3A4",
}


def load_main8_all(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = list(payload.get("drugs", payload))
    out = []
    for row in rows:
        normalized = _normalize_row(row)
        if normalized is not None:
            out.append(normalized)
    return out


def load_attnsom_all(root: Path, *, source_name: str = "ATTNSOM") -> list[dict]:
    rows: list[dict] = []
    for stem, cyp_name in sorted(ATTNSOM_CYP_MAP.items()):
        sdf_path = root / f"{stem}.sdf"
        rows.extend(load_attnsom_sdf(sdf_path, source_name=source_name, cyp_name=cyp_name))
    return rows


def build_summary(rows: list[dict]) -> dict:
    sources = Counter(str(row.get("source", "unknown")) for row in rows)
    confidences = Counter(str(row.get("confidence", "unknown")) for row in rows)
    cyps = Counter(str(row.get("primary_cyp", "unknown")) for row in rows)
    return {
        "n_rows": len(rows),
        "n_site_labeled": len(rows),
        "sources": dict(sorted(sources.items())),
        "confidences": dict(sorted(confidences.items())),
        "cyps": dict(sorted(cyps.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dataset",
        default="data/prepared_training/main8_site_conservative_singlecyp_clean_symm.json",
    )
    parser.add_argument(
        "--astra-csv",
        default="AstraZeneca_SoM_annotations_120_Compounds.csv",
    )
    parser.add_argument(
        "--attnsom-root",
        default="data/ATTNSOM/cyp_dataset",
    )
    parser.add_argument(
        "--out",
        default="data/prepared_training/main8_multicyp_attnsom_sourceaware.json",
    )
    parser.add_argument(
        "--merge-policy",
        choices=("union", "base_priority", "keep_sources"),
        default="keep_sources",
    )
    args = parser.parse_args()

    base_rows = load_main8_all(ROOT / args.base_dataset)
    astra_rows = load_astrazeneca_csv(ROOT / args.astra_csv)
    attnsom_rows = load_attnsom_all(ROOT / args.attnsom_root, source_name="ATTNSOM")
    imported_rows = astra_rows + attnsom_rows
    merged_rows, merge_stats = merge_rows(base_rows, imported_rows, merge_policy=args.merge_policy)
    summary = build_summary(merged_rows)

    payload = {
        "n_drugs": len(merged_rows),
        "n_site_labeled": len(merged_rows),
        "summary": summary,
        "build_stats": {
            "base_dataset": args.base_dataset,
            "astra_rows": len(astra_rows),
            "attnsom_rows": len(attnsom_rows),
            "merge_policy": args.merge_policy,
            **merge_stats,
        },
        "drugs": merged_rows,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved augmented multi-CYP dataset to {out_path}")
    print(json.dumps(payload["build_stats"], indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
