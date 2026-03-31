from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from rdkit import Chem


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _canonical_smiles(smiles: str) -> str:
    text = " ".join(str(smiles or "").split())
    if not text:
        return ""
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return text
    return Chem.MolToSmiles(mol, canonical=True)


def _site_atoms(row: dict[str, Any]) -> list[int]:
    if row.get("som"):
        values = []
        for item in row.get("som", []):
            atom_idx = item.get("atom_idx", item) if isinstance(item, dict) else item
            if isinstance(atom_idx, int):
                values.append(int(atom_idx))
        return sorted(set(values))
    if row.get("site_atoms"):
        return sorted(set(int(v) for v in row.get("site_atoms", []) if isinstance(v, int)))
    if row.get("site_atom_indices"):
        return sorted(set(int(v) for v in row.get("site_atom_indices", []) if isinstance(v, int)))
    if row.get("metabolism_sites"):
        return sorted(set(int(v) for v in row.get("metabolism_sites", []) if isinstance(v, int)))
    return []


def _size_bucket(num_atoms: int) -> str:
    if num_atoms <= 0:
        return "unknown"
    if num_atoms <= 15:
        return "<=15"
    if num_atoms <= 25:
        return "16-25"
    if num_atoms <= 40:
        return "26-40"
    if num_atoms <= 60:
        return "41-60"
    return "61+"


def _row_summary(row: dict[str, Any], mol: Chem.Mol | None) -> dict[str, Any]:
    sites = _site_atoms(row)
    num_atoms = int(mol.GetNumAtoms()) if mol is not None else 0
    metabolism_sites = row.get("metabolism_sites")
    metabolism_set = (
        sorted(set(int(v) for v in metabolism_sites if isinstance(v, int)))
        if isinstance(metabolism_sites, list)
        else []
    )
    duplicate_sites = len(sites) != len(list(row.get("site_atoms") or sites))
    out_of_range = [idx for idx in sites if idx < 0 or idx >= num_atoms]
    source = str(row.get("source", "unknown"))
    confidence = str(row.get("confidence", "unknown"))
    return {
        "id": str(row.get("id", "")),
        "name": str(row.get("name", "")),
        "smiles": str(row.get("smiles", "")),
        "canonical_smiles": _canonical_smiles(row.get("smiles", "")),
        "source": source,
        "site_source": str(row.get("site_source", "")),
        "confidence": confidence,
        "primary_cyp": str(row.get("primary_cyp", "")),
        "site_atoms": sites,
        "metabolism_sites": metabolism_set,
        "num_atoms": num_atoms,
        "size_bucket": _size_bucket(num_atoms),
        "site_count": len(sites),
        "site_count_bucket": "single" if len(sites) == 1 else "multi" if len(sites) > 1 else "none",
        "duplicate_sites": duplicate_sites,
        "out_of_range_sites": out_of_range,
        "site_vs_metabolism_match": sites == metabolism_set if metabolism_set else True,
    }


def _symmetry_audit(summary: dict[str, Any], mol: Chem.Mol | None) -> dict[str, Any]:
    if mol is None:
        return {
            "symmetry_classes": {},
            "ambiguous_labeled_sites": [],
            "missing_equivalent_sites": [],
            "equivalent_site_groups": [],
        }
    try:
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    except Exception:
        ranks = list(range(int(mol.GetNumAtoms())))
    class_to_atoms: dict[int, list[int]] = defaultdict(list)
    for idx, rank in enumerate(ranks):
        class_to_atoms[int(rank)].append(idx)
    labeled = set(summary["site_atoms"])
    ambiguous_labeled_sites: list[int] = []
    missing_equivalent_sites: list[dict[str, Any]] = []
    equivalent_site_groups: list[list[int]] = []
    for atoms in class_to_atoms.values():
        if len(atoms) <= 1:
            continue
        labeled_group = sorted(idx for idx in atoms if idx in labeled)
        unlabeled_group = sorted(idx for idx in atoms if idx not in labeled)
        if labeled_group:
            equivalent_site_groups.append(sorted(atoms))
        if labeled_group and unlabeled_group:
            ambiguous_labeled_sites.extend(labeled_group)
            missing_equivalent_sites.append(
                {
                    "labeled": labeled_group,
                    "unlabeled_equivalents": unlabeled_group,
                    "all_equivalents": sorted(atoms),
                }
            )
    return {
        "symmetry_classes": {str(rank): atoms for rank, atoms in class_to_atoms.items() if len(atoms) > 1},
        "ambiguous_labeled_sites": sorted(set(ambiguous_labeled_sites)),
        "missing_equivalent_sites": missing_equivalent_sites,
        "equivalent_site_groups": equivalent_site_groups,
    }


def _conflict_key(summary: dict[str, Any]) -> tuple[str, tuple[int, ...], str]:
    return (
        summary["canonical_smiles"],
        tuple(summary["site_atoms"]),
        summary["primary_cyp"],
    )


def audit_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_row: list[dict[str, Any]] = []
    duplicate_smiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_smiles: list[str] = []

    for row in rows:
        smiles = str(row.get("smiles", ""))
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        summary = _row_summary(row, mol)
        summary["symmetry"] = _symmetry_audit(summary, mol)
        if mol is None:
            invalid_smiles.append(summary["id"])
        per_row.append(summary)
        if summary["canonical_smiles"]:
            duplicate_smiles[summary["canonical_smiles"]].append(summary)

    source_counts = Counter(item["source"] for item in per_row)
    confidence_counts = Counter(item["confidence"] for item in per_row)
    size_counts = Counter(item["size_bucket"] for item in per_row)
    site_count_buckets = Counter(item["site_count_bucket"] for item in per_row)

    source_breakdown: dict[str, dict[str, Any]] = {}
    for source in sorted(source_counts):
        items = [item for item in per_row if item["source"] == source]
        source_breakdown[source] = {
            "total": len(items),
            "confidence": dict(Counter(item["confidence"] for item in items)),
            "size_buckets": dict(Counter(item["size_bucket"] for item in items)),
            "site_count_buckets": dict(Counter(item["site_count_bucket"] for item in items)),
            "ambiguous_equivalent_labels": sum(1 for item in items if item["symmetry"]["missing_equivalent_sites"]),
            "site_vs_metabolism_mismatch": sum(1 for item in items if not item["site_vs_metabolism_match"]),
            "out_of_range": sum(1 for item in items if item["out_of_range_sites"]),
        }

    duplicate_conflicts: list[dict[str, Any]] = []
    for smiles, items in duplicate_smiles.items():
        if len(items) <= 1:
            continue
        distinct_labels = sorted({tuple(item["site_atoms"]) for item in items})
        distinct_sources = sorted({item["source"] for item in items})
        distinct_cyps = sorted({item["primary_cyp"] for item in items})
        if len(distinct_labels) > 1 or len(distinct_sources) > 1 or len(distinct_cyps) > 1:
            duplicate_conflicts.append(
                {
                    "canonical_smiles": smiles,
                    "count": len(items),
                    "distinct_site_labels": [list(v) for v in distinct_labels],
                    "distinct_sources": distinct_sources,
                    "distinct_cyps": distinct_cyps,
                    "examples": [
                        {
                            "id": item["id"],
                            "name": item["name"],
                            "source": item["source"],
                            "confidence": item["confidence"],
                            "primary_cyp": item["primary_cyp"],
                            "site_atoms": item["site_atoms"],
                        }
                        for item in items[:5]
                    ],
                }
            )

    ambiguous_equivalent_rows = [item for item in per_row if item["symmetry"]["missing_equivalent_sites"]]
    site_mismatch_rows = [item for item in per_row if not item["site_vs_metabolism_match"]]
    out_of_range_rows = [item for item in per_row if item["out_of_range_sites"]]
    low_conf_rows = [item for item in per_row if item["confidence"] in {"low", "unknown"}]

    return {
        "total_rows": len(per_row),
        "invalid_smiles": invalid_smiles,
        "source_counts": dict(source_counts),
        "confidence_counts": dict(confidence_counts),
        "size_counts": dict(size_counts),
        "site_count_buckets": dict(site_count_buckets),
        "mean_site_count": mean(item["site_count"] for item in per_row if item["site_count"] > 0) if any(item["site_count"] > 0 for item in per_row) else 0.0,
        "source_breakdown": source_breakdown,
        "ambiguous_equivalent_label_rows": len(ambiguous_equivalent_rows),
        "site_vs_metabolism_mismatch_rows": len(site_mismatch_rows),
        "out_of_range_rows": len(out_of_range_rows),
        "low_confidence_rows": len(low_conf_rows),
        "duplicate_smiles_conflicts": duplicate_conflicts,
        "top_ambiguous_equivalent_examples": [
            {
                "id": item["id"],
                "name": item["name"],
                "source": item["source"],
                "confidence": item["confidence"],
                "site_atoms": item["site_atoms"],
                "missing_equivalent_sites": item["symmetry"]["missing_equivalent_sites"],
            }
            for item in ambiguous_equivalent_rows[:20]
        ],
        "top_site_mismatch_examples": [
            {
                "id": item["id"],
                "name": item["name"],
                "source": item["source"],
                "site_atoms": item["site_atoms"],
                "metabolism_sites": item["metabolism_sites"],
            }
            for item in site_mismatch_rows[:20]
        ],
        "top_out_of_range_examples": [
            {
                "id": item["id"],
                "name": item["name"],
                "source": item["source"],
                "num_atoms": item["num_atoms"],
                "site_atoms": item["site_atoms"],
                "out_of_range_sites": item["out_of_range_sites"],
            }
            for item in out_of_range_rows[:20]
        ],
        "top_duplicate_conflicts": duplicate_conflicts[:20],
    }


def _to_markdown(report: dict[str, Any], dataset_path: Path) -> str:
    lines = [
        f"# Label Audit: `{dataset_path}`",
        "",
        "## Summary",
        "",
        f"- Total rows: `{report['total_rows']}`",
        f"- Invalid SMILES: `{len(report['invalid_smiles'])}`",
        f"- Ambiguous equivalent-label rows: `{report['ambiguous_equivalent_label_rows']}`",
        f"- Site/metabolism mismatch rows: `{report['site_vs_metabolism_mismatch_rows']}`",
        f"- Out-of-range rows: `{report['out_of_range_rows']}`",
        f"- Low-confidence rows: `{report['low_confidence_rows']}`",
        f"- Duplicate SMILES conflicts: `{len(report['duplicate_smiles_conflicts'])}`",
        "",
        "## Sources",
        "",
    ]
    for source, info in report["source_breakdown"].items():
        lines.extend(
            [
                f"### {source}",
                "",
                f"- Total: `{info['total']}`",
                f"- Confidence: `{info['confidence']}`",
                f"- Size buckets: `{info['size_buckets']}`",
                f"- Site-count buckets: `{info['site_count_buckets']}`",
                f"- Ambiguous equivalent labels: `{info['ambiguous_equivalent_labels']}`",
                f"- Site/metabolism mismatches: `{info['site_vs_metabolism_mismatch']}`",
                f"- Out-of-range rows: `{info['out_of_range']}`",
                "",
            ]
        )
    if report["top_ambiguous_equivalent_examples"]:
        lines.extend(["## Ambiguous Equivalent Labels", ""])
        for item in report["top_ambiguous_equivalent_examples"][:10]:
            lines.append(
                f"- `{item['name']}` [{item['source']}] labeled={item['site_atoms']} "
                f"missing_equivalents={item['missing_equivalent_sites']}"
            )
        lines.append("")
    if report["top_duplicate_conflicts"]:
        lines.extend(["## Duplicate SMILES Conflicts", ""])
        for item in report["top_duplicate_conflicts"][:10]:
            lines.append(
                f"- `{item['canonical_smiles']}` labels={item['distinct_site_labels']} "
                f"sources={item['distinct_sources']} cyps={item['distinct_cyps']}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit site labels for hybrid SOM datasets")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rows = _load_rows(dataset_path)
    report = audit_dataset(rows)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote JSON audit: {out_json}")
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.write_text(_to_markdown(report, dataset_path), encoding="utf-8")
        print(f"Wrote Markdown audit: {out_md}")
    if not args.output_json and not args.output_md:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
