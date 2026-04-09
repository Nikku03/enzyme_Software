from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (str(SRC), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from enzyme_software.liquid_nn_v2.data.dataset_loader import _extract_site_atoms
from enzyme_software.liquid_nn_v2.experiments.hybrid_full_xtb.dataset import split_drugs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit novelty of the filtered CYP3A4 benchmark after excluding ATTNSOM and cyp_dbs_external."
    )
    parser.add_argument("--dataset", default="data/prepared_training/main8_cyp3a4_no_attnsom.json")
    parser.add_argument("--target-cyp", default="CYP3A4")
    parser.add_argument("--include-sources", default="AZ120,DrugBank,MetXBioDB")
    parser.add_argument("--exclude-sources", default="cyp_dbs_external")
    parser.add_argument("--split-mode", default="scaffold_source_size", choices=("random", "scaffold_source", "scaffold_source_size"))
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.80)
    parser.add_argument("--strong-near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--output-dir", default="artifacts/benchmark_novelty_audit")
    return parser.parse_args()


def _normalize_source(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_bucket(raw_source: str) -> str:
    normalized = _normalize_source(raw_source)
    mapping = {
        "az120": "AZ120",
        "drugbank": "DrugBank",
        "metxbiodb": "MetXBioDB",
        "cyp_dbs_external": "cyp_dbs_external",
        "cyp_dbs_experimental": "cyp_dbs_external",
        "attnsom": "ATTNSOM",
        "metapred": "MetaPred",
        "literature": "literature",
        "validated": "validated",
    }
    return mapping.get(normalized, str(raw_source or "").strip() or "unknown")


def _parse_source_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def _inchikey(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return ""
    try:
        return str(Chem.MolToInchiKey(mol) or "")
    except Exception:
        return ""


def _murcko_scaffold(smiles: str, *, generic: bool = False) -> str:
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return ""
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold_mol is None:
            return ""
        if generic:
            scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        return str(Chem.MolToSmiles(scaffold_mol, canonical=True) or "")
    except Exception:
        return ""


def _fingerprint(smiles: str, *, radius: int, n_bits: int):
    mol = Chem.MolFromSmiles(str(smiles or "").strip())
    if mol is None:
        return None
    try:
        generator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return generator.GetFingerprint(mol)
    except Exception:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = []
    for row in rows:
        normalized = {}
        for key, value in row.items():
            if isinstance(value, (list, dict)):
                normalized[key] = json.dumps(value, sort_keys=True)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    fieldnames = list(fieldnames or sorted({key for row in normalized_rows for key in row}))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(row)


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def _bucket_similarity(value: float) -> str:
    if value < 0.40:
        return "<0.40"
    if value < 0.60:
        return "0.40-0.60"
    if value < 0.80:
        return "0.60-0.80"
    return ">0.80"


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "max": None, "min": None}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
        "min": float(min(values)),
    }


def _pair_key(left: str, right: str) -> str:
    return f"{left} -> {right}"


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    target_cyp: str,
    include_sources: set[str],
    exclude_sources: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    raw_source_counts = Counter()
    kept_source_counts = Counter()
    excluded_source_counts = Counter()
    excluded_reason_counts = Counter()
    invalid_row_count = 0
    for index, row in enumerate(rows):
        source = _canonical_bucket(str(row.get("source") or row.get("site_source") or "unknown"))
        raw_source_counts[source] += 1
        cyp = str(row.get("primary_cyp") or row.get("cyp") or "").strip()
        if target_cyp and cyp and cyp != target_cyp:
            excluded_source_counts[source] += 1
            excluded_reason_counts["target_cyp_mismatch"] += 1
            continue
        if source in exclude_sources:
            excluded_source_counts[source] += 1
            excluded_reason_counts["explicit_exclude_source"] += 1
            continue
        if include_sources and source not in include_sources:
            excluded_source_counts[source] += 1
            excluded_reason_counts["not_in_include_sources"] += 1
            continue
        if not _extract_site_atoms(row):
            excluded_source_counts[source] += 1
            excluded_reason_counts["missing_site_atoms"] += 1
            invalid_row_count += 1
            continue
        smiles = _canonical_smiles(str(row.get("smiles") or ""))
        if not smiles:
            excluded_source_counts[source] += 1
            excluded_reason_counts["invalid_smiles"] += 1
            invalid_row_count += 1
            continue
        normalized = dict(row)
        normalized["source"] = source
        normalized["site_source"] = source
        normalized["smiles"] = smiles
        normalized["_audit_original_index"] = int(index)
        kept.append(normalized)
        kept_source_counts[source] += 1
    return kept, {
        "raw_source_counts": dict(sorted(raw_source_counts.items())),
        "kept_source_counts": dict(sorted(kept_source_counts.items())),
        "excluded_source_counts": dict(sorted(excluded_source_counts.items())),
        "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
        "invalid_row_count": int(invalid_row_count),
    }


def _prepare_rows(rows: list[dict[str, Any]], *, radius: int, n_bits: int) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        smiles = str(row.get("smiles") or "")
        prepared_row = dict(row)
        prepared_row["canonical_smiles"] = smiles
        prepared_row["inchikey"] = _inchikey(smiles)
        prepared_row["murcko_scaffold"] = _murcko_scaffold(smiles, generic=False)
        prepared_row["murcko_scaffold_generic"] = _murcko_scaffold(smiles, generic=True)
        prepared_row["_fingerprint"] = _fingerprint(smiles, radius=radius, n_bits=n_bits)
        prepared.append(prepared_row)
    return prepared


def _annotate_split(rows: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    annotated = []
    for local_index, row in enumerate(rows):
        tagged = dict(row)
        tagged["split"] = str(split_name)
        tagged["split_local_index"] = int(local_index)
        annotated.append(tagged)
    return annotated


def _exact_overlap_rows(
    *,
    train_rows: list[dict[str, Any]],
    compare_rows: list[dict[str, Any]],
    compare_split: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    overlaps: list[dict[str, Any]] = []
    train_by_smiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    train_by_inchikey: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for train_row in train_rows:
        if train_row.get("canonical_smiles"):
            train_by_smiles[str(train_row["canonical_smiles"])].append(train_row)
        if train_row.get("inchikey"):
            train_by_inchikey[str(train_row["inchikey"])].append(train_row)
    canonical_hits = 0
    inchikey_hits = 0
    union_keys: set[tuple[int, int]] = set()
    for compare_row in compare_rows:
        candidates: list[tuple[str, dict[str, Any]]] = []
        smiles_key = str(compare_row.get("canonical_smiles") or "")
        inchikey_key = str(compare_row.get("inchikey") or "")
        for train_row in train_by_smiles.get(smiles_key, []):
            candidates.append(("canonical_smiles", train_row))
        for train_row in train_by_inchikey.get(inchikey_key, []):
            candidates.append(("inchikey", train_row))
        seen_pairs: set[tuple[str, int]] = set()
        pair_matched_on: dict[int, set[str]] = defaultdict(set)
        for match_field, train_row in candidates:
            train_idx = int(train_row["_audit_original_index"])
            pair_signature = (match_field, train_idx)
            if pair_signature in seen_pairs:
                continue
            seen_pairs.add(pair_signature)
            pair_matched_on[train_idx].add(match_field)
        for train_idx, matched_on in sorted(pair_matched_on.items()):
            train_row = next(row for row in train_rows if int(row["_audit_original_index"]) == train_idx)
            if "canonical_smiles" in matched_on:
                canonical_hits += 1
            if "inchikey" in matched_on:
                inchikey_hits += 1
            union_keys.add((train_idx, int(compare_row["_audit_original_index"])))
            overlaps.append(
                {
                    "compare_split": str(compare_split),
                    "match_type": ",".join(sorted(matched_on)),
                    "train_dataset_index": int(train_row["_audit_original_index"]),
                    "compare_dataset_index": int(compare_row["_audit_original_index"]),
                    "train_id": str(train_row.get("id") or ""),
                    "compare_id": str(compare_row.get("id") or ""),
                    "train_name": str(train_row.get("name") or ""),
                    "compare_name": str(compare_row.get("name") or ""),
                    "train_source": str(train_row.get("source") or ""),
                    "compare_source": str(compare_row.get("source") or ""),
                    "train_smiles": str(train_row.get("canonical_smiles") or ""),
                    "compare_smiles": str(compare_row.get("canonical_smiles") or ""),
                    "train_inchikey": str(train_row.get("inchikey") or ""),
                    "compare_inchikey": str(compare_row.get("inchikey") or ""),
                    "cross_source": bool(str(train_row.get("source")) != str(compare_row.get("source"))),
                }
            )
    return overlaps, {
        f"canonical_smiles_train_{compare_split}_overlap_count": int(canonical_hits),
        f"inchikey_train_{compare_split}_overlap_count": int(inchikey_hits),
        f"exact_train_{compare_split}_overlap_count": int(len(union_keys)),
    }


def _scaffold_overlap_rows(
    *,
    train_rows: list[dict[str, Any]],
    compare_rows: list[dict[str, Any]],
    compare_split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_scaffolds = {str(row.get("murcko_scaffold") or "") for row in train_rows if str(row.get("murcko_scaffold") or "")}
    compare_scaffolds = {str(row.get("murcko_scaffold") or "") for row in compare_rows if str(row.get("murcko_scaffold") or "")}
    shared = train_scaffolds & compare_scaffolds
    rows: list[dict[str, Any]] = []
    seen_count = 0
    unseen_count = 0
    for row in compare_rows:
        scaffold = str(row.get("murcko_scaffold") or "")
        seen = bool(scaffold and scaffold in train_scaffolds)
        seen_count += int(seen)
        unseen_count += int(not seen)
        rows.append(
            {
                "compare_split": str(compare_split),
                "dataset_index": int(row["_audit_original_index"]),
                "id": str(row.get("id") or ""),
                "name": str(row.get("name") or ""),
                "source": str(row.get("source") or ""),
                "canonical_smiles": str(row.get("canonical_smiles") or ""),
                "murcko_scaffold": scaffold,
                "murcko_scaffold_generic": str(row.get("murcko_scaffold_generic") or ""),
                "train_scaffold_seen": bool(seen),
            }
        )
    return rows, {
        f"unique_train_{compare_split}_shared_scaffold_count": int(len(shared)),
        f"{compare_split}_scaffold_seen_fraction": float(seen_count) / float(len(compare_rows)) if compare_rows else 0.0,
        f"{compare_split}_seen_scaffold_count": int(seen_count),
        f"{compare_split}_unseen_scaffold_count": int(unseen_count),
    }


def _nearest_neighbor_rows(
    *,
    train_rows: list[dict[str, Any]],
    compare_rows: list[dict[str, Any]],
    compare_split: str,
    near_duplicate_threshold: float,
    strong_near_duplicate_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    train_fps = [row.get("_fingerprint") for row in train_rows]
    nn_rows: list[dict[str, Any]] = []
    similarities: list[float] = []
    cross_source_rows: list[dict[str, Any]] = []
    pair_counter = Counter()
    for compare_row in compare_rows:
        fp = compare_row.get("_fingerprint")
        if fp is None or not train_fps:
            similarity = 0.0
            best_train = None
        else:
            values = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            if not values:
                similarity = 0.0
                best_train = None
            else:
                best_index = max(range(len(values)), key=lambda idx: float(values[idx]))
                similarity = float(values[best_index])
                best_train = train_rows[int(best_index)]
        similarities.append(float(similarity))
        bucket = _bucket_similarity(float(similarity))
        nn_row = {
            "compare_split": str(compare_split),
            "dataset_index": int(compare_row["_audit_original_index"]),
            "id": str(compare_row.get("id") or ""),
            "name": str(compare_row.get("name") or ""),
            "source": str(compare_row.get("source") or ""),
            "canonical_smiles": str(compare_row.get("canonical_smiles") or ""),
            "murcko_scaffold": str(compare_row.get("murcko_scaffold") or ""),
            "nearest_train_similarity": float(similarity),
            "nearest_train_similarity_bucket": str(bucket),
            "nearest_train_id": str(best_train.get("id") or "") if best_train is not None else "",
            "nearest_train_name": str(best_train.get("name") or "") if best_train is not None else "",
            "nearest_train_source": str(best_train.get("source") or "") if best_train is not None else "",
            "nearest_train_canonical_smiles": str(best_train.get("canonical_smiles") or "") if best_train is not None else "",
            "nearest_train_murcko_scaffold": str(best_train.get("murcko_scaffold") or "") if best_train is not None else "",
            "nearest_train_dataset_index": int(best_train["_audit_original_index"]) if best_train is not None else None,
            "cross_source_near_duplicate": bool(
                best_train is not None
                and str(best_train.get("source") or "") != str(compare_row.get("source") or "")
                and float(similarity) >= float(near_duplicate_threshold)
            ),
            "cross_source_strong_near_duplicate": bool(
                best_train is not None
                and str(best_train.get("source") or "") != str(compare_row.get("source") or "")
                and float(similarity) >= float(strong_near_duplicate_threshold)
            ),
        }
        nn_rows.append(nn_row)
        if nn_row["cross_source_near_duplicate"]:
            pair = _pair_key(str(compare_row.get("source") or ""), str(best_train.get("source") or ""))
            pair_counter[pair] += 1
            cross_source_rows.append(
                {
                    "compare_split": str(compare_split),
                    "leakage_type": "near_duplicate",
                    "similarity": float(similarity),
                    "strong_near_duplicate": bool(nn_row["cross_source_strong_near_duplicate"]),
                    "compare_dataset_index": int(compare_row["_audit_original_index"]),
                    "compare_id": str(compare_row.get("id") or ""),
                    "compare_name": str(compare_row.get("name") or ""),
                    "compare_source": str(compare_row.get("source") or ""),
                    "compare_smiles": str(compare_row.get("canonical_smiles") or ""),
                    "train_dataset_index": int(best_train["_audit_original_index"]),
                    "train_id": str(best_train.get("id") or ""),
                    "train_name": str(best_train.get("name") or ""),
                    "train_source": str(best_train.get("source") or ""),
                    "train_smiles": str(best_train.get("canonical_smiles") or ""),
                    "source_pair": str(pair),
                }
            )
    stats = _stats(similarities)
    bucket_counts = dict(sorted(Counter(_bucket_similarity(value) for value in similarities).items()))
    return nn_rows, {
        f"{compare_split}_nearest_neighbor_similarity_mean": stats["mean"],
        f"{compare_split}_nearest_neighbor_similarity_median": stats["median"],
        f"{compare_split}_nearest_neighbor_similarity_max": stats["max"],
        f"{compare_split}_nearest_neighbor_similarity_min": stats["min"],
        f"{compare_split}_nearest_neighbor_similarity_bucket_counts": bucket_counts,
        f"{compare_split}_near_duplicate_cross_source_count": int(len(cross_source_rows)),
        f"{compare_split}_near_duplicate_cross_source_counts_by_source_pair": dict(sorted(pair_counter.items())),
    }, cross_source_rows


def _classify_benchmark(
    *,
    exact_train_test_overlap_count: int,
    test_scaffold_seen_fraction: float,
    test_nearest_neighbor_similarity_median: float | None,
    test_nearest_neighbor_similarity_bucket_counts: dict[str, int],
) -> tuple[str, str]:
    high_similarity_fraction = 0.0
    total = sum(int(v) for v in test_nearest_neighbor_similarity_bucket_counts.values())
    if total > 0:
        high_similarity_fraction = float(test_nearest_neighbor_similarity_bucket_counts.get(">0.80", 0)) / float(total)
    median = float(test_nearest_neighbor_similarity_median) if test_nearest_neighbor_similarity_median is not None else 0.0
    if (
        int(exact_train_test_overlap_count) == 0
        and float(test_scaffold_seen_fraction) <= 0.25
        and float(high_similarity_fraction) <= 0.15
        and float(median) < 0.60
    ):
        return (
            "strong_novelty_benchmark",
            "No exact train/test overlap, low scaffold reuse, and mostly moderate-or-lower nearest-train similarity.",
        )
    if (
        int(exact_train_test_overlap_count) == 0
        and float(test_scaffold_seen_fraction) <= 0.75
        and float(high_similarity_fraction) <= 0.50
    ):
        return (
            "scaffold_novel_partial",
            "No exact train/test overlap, but a meaningful fraction of test molecules still reuse train scaffolds or close analog neighborhoods.",
        )
    return (
        "held_out_only",
        "The benchmark is held out by split, but scaffold reuse and/or close train analogs remain substantial.",
    )


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_sources = {_canonical_bucket(value) for value in _parse_source_csv(args.include_sources)}
    exclude_sources = {_canonical_bucket(value) for value in _parse_source_csv(args.exclude_sources)}
    raw_rows = _load_dataset(dataset_path)
    filtered_rows, filter_summary = _filter_rows(
        raw_rows,
        target_cyp=str(args.target_cyp or "").strip(),
        include_sources=include_sources,
        exclude_sources=exclude_sources,
    )
    filtered_rows = _prepare_rows(filtered_rows, radius=int(args.fp_radius), n_bits=int(args.fp_bits))

    train_rows, val_rows, test_rows = split_drugs(
        filtered_rows,
        seed=int(args.seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        mode=str(args.split_mode),
    )
    train_rows = _annotate_split(train_rows, "train")
    val_rows = _annotate_split(val_rows, "val")
    test_rows = _annotate_split(test_rows, "test")

    exact_rows_test, exact_test_summary = _exact_overlap_rows(
        train_rows=train_rows,
        compare_rows=test_rows,
        compare_split="test",
    )
    exact_rows_val, exact_val_summary = _exact_overlap_rows(
        train_rows=train_rows,
        compare_rows=val_rows,
        compare_split="val",
    )
    exact_rows = sorted(
        exact_rows_test + exact_rows_val,
        key=lambda row: (str(row["compare_split"]), str(row["compare_source"]), str(row["compare_id"]), str(row["train_id"])),
    )

    scaffold_rows_test, scaffold_test_summary = _scaffold_overlap_rows(
        train_rows=train_rows,
        compare_rows=test_rows,
        compare_split="test",
    )
    scaffold_rows_val, scaffold_val_summary = _scaffold_overlap_rows(
        train_rows=train_rows,
        compare_rows=val_rows,
        compare_split="val",
    )
    scaffold_rows = sorted(
        scaffold_rows_test + scaffold_rows_val,
        key=lambda row: (str(row["compare_split"]), str(row["source"]), int(row["dataset_index"])),
    )

    nn_rows_test, nn_test_summary, cross_source_rows_test = _nearest_neighbor_rows(
        train_rows=train_rows,
        compare_rows=test_rows,
        compare_split="test",
        near_duplicate_threshold=float(args.near_duplicate_threshold),
        strong_near_duplicate_threshold=float(args.strong_near_duplicate_threshold),
    )
    nn_rows_val, nn_val_summary, cross_source_rows_val = _nearest_neighbor_rows(
        train_rows=train_rows,
        compare_rows=val_rows,
        compare_split="val",
        near_duplicate_threshold=float(args.near_duplicate_threshold),
        strong_near_duplicate_threshold=float(args.strong_near_duplicate_threshold),
    )
    nn_rows = sorted(
        nn_rows_test + nn_rows_val,
        key=lambda row: (str(row["compare_split"]), str(row["source"]), -float(row["nearest_train_similarity"])),
    )

    exact_cross_source_rows = [
        dict(row, leakage_type="exact_duplicate", similarity=1.0, source_pair=_pair_key(str(row["compare_source"]), str(row["train_source"])))
        for row in exact_rows
        if bool(row["cross_source"])
    ]
    cross_source_rows = sorted(
        exact_cross_source_rows + cross_source_rows_test + cross_source_rows_val,
        key=lambda row: (str(row["compare_split"]), str(row["leakage_type"]), -float(row["similarity"])),
    )
    exact_cross_source_counts = Counter(str(row["source_pair"]) for row in exact_cross_source_rows)
    near_cross_source_counts = Counter(
        str(row["source_pair"]) for row in cross_source_rows if str(row["leakage_type"]) == "near_duplicate"
    )

    unique_train_scaffolds = {str(row.get("murcko_scaffold") or "") for row in train_rows if str(row.get("murcko_scaffold") or "")}
    unique_test_scaffolds = {str(row.get("murcko_scaffold") or "") for row in test_rows if str(row.get("murcko_scaffold") or "")}
    novelty_class, interpretation = _classify_benchmark(
        exact_train_test_overlap_count=int(exact_test_summary["exact_train_test_overlap_count"]),
        test_scaffold_seen_fraction=float(scaffold_test_summary["test_scaffold_seen_fraction"]),
        test_nearest_neighbor_similarity_median=nn_test_summary["test_nearest_neighbor_similarity_median"],
        test_nearest_neighbor_similarity_bucket_counts=nn_test_summary["test_nearest_neighbor_similarity_bucket_counts"],
    )

    summary = {
        "dataset_path_used": str(dataset_path),
        "filtering_applied_on_the_fly": True,
        "benchmark_definition": {
            "base_dataset": str(dataset_path),
            "target_cyp": str(args.target_cyp),
            "included_sources": sorted(include_sources),
            "excluded_sources": sorted(exclude_sources),
            "split_mode": str(args.split_mode),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
        },
        "filter_summary": filter_summary,
        "train_count": int(len(train_rows)),
        "val_count": int(len(val_rows)),
        "test_count": int(len(test_rows)),
        "exact_train_test_overlap_count": int(exact_test_summary["exact_train_test_overlap_count"]),
        "exact_train_val_overlap_count": int(exact_val_summary["exact_train_val_overlap_count"]),
        "canonical_smiles_train_test_overlap_count": int(exact_test_summary["canonical_smiles_train_test_overlap_count"]),
        "canonical_smiles_train_val_overlap_count": int(exact_val_summary["canonical_smiles_train_val_overlap_count"]),
        "inchikey_train_test_overlap_count": int(exact_test_summary["inchikey_train_test_overlap_count"]),
        "inchikey_train_val_overlap_count": int(exact_val_summary["inchikey_train_val_overlap_count"]),
        "unique_train_scaffold_count": int(len(unique_train_scaffolds)),
        "unique_test_scaffold_count": int(len(unique_test_scaffolds)),
        "shared_train_test_scaffold_count": int(len(unique_train_scaffolds & unique_test_scaffolds)),
        "test_scaffold_seen_fraction": float(scaffold_test_summary["test_scaffold_seen_fraction"]),
        "val_scaffold_seen_fraction": float(scaffold_val_summary["val_scaffold_seen_fraction"]),
        "nearest_neighbor_similarity_mean": nn_test_summary["test_nearest_neighbor_similarity_mean"],
        "nearest_neighbor_similarity_median": nn_test_summary["test_nearest_neighbor_similarity_median"],
        "nearest_neighbor_similarity_max": nn_test_summary["test_nearest_neighbor_similarity_max"],
        "nearest_neighbor_similarity_min": nn_test_summary["test_nearest_neighbor_similarity_min"],
        "nearest_neighbor_similarity_bucket_counts": nn_test_summary["test_nearest_neighbor_similarity_bucket_counts"],
        "val_nearest_neighbor_similarity_bucket_counts": nn_val_summary["val_nearest_neighbor_similarity_bucket_counts"],
        "exact_cross_source_duplicate_count": int(len(exact_cross_source_rows)),
        "near_duplicate_cross_source_count": int(sum(1 for row in cross_source_rows if str(row["leakage_type"]) == "near_duplicate")),
        "exact_cross_source_counts_by_source_pair": dict(sorted(exact_cross_source_counts.items())),
        "near_duplicate_cross_source_counts_by_source_pair": dict(sorted(near_cross_source_counts.items())),
        "benchmark_novelty_class": str(novelty_class),
        "interpretation": str(interpretation),
        "near_duplicate_similarity_threshold": float(args.near_duplicate_threshold),
        "strong_near_duplicate_similarity_threshold": float(args.strong_near_duplicate_threshold),
        "outputs": {
            "summary_json": str(output_dir / "benchmark_novelty_audit_summary.json"),
            "exact_overlap_csv": str(output_dir / "benchmark_exact_overlap.csv"),
            "scaffold_overlap_csv": str(output_dir / "benchmark_scaffold_overlap.csv"),
            "nearest_neighbor_csv": str(output_dir / "benchmark_nearest_neighbor_similarity.csv"),
            "cross_source_leakage_csv": str(output_dir / "benchmark_cross_source_leakage.csv"),
        },
    }

    _write_csv(
        output_dir / "benchmark_exact_overlap.csv",
        exact_rows,
        fieldnames=[
            "compare_split",
            "match_type",
            "train_dataset_index",
            "compare_dataset_index",
            "train_id",
            "compare_id",
            "train_name",
            "compare_name",
            "train_source",
            "compare_source",
            "train_smiles",
            "compare_smiles",
            "train_inchikey",
            "compare_inchikey",
            "cross_source",
        ],
    )
    _write_csv(
        output_dir / "benchmark_scaffold_overlap.csv",
        scaffold_rows,
        fieldnames=[
            "compare_split",
            "dataset_index",
            "id",
            "name",
            "source",
            "canonical_smiles",
            "murcko_scaffold",
            "murcko_scaffold_generic",
            "train_scaffold_seen",
        ],
    )
    _write_csv(
        output_dir / "benchmark_nearest_neighbor_similarity.csv",
        nn_rows,
        fieldnames=[
            "compare_split",
            "dataset_index",
            "id",
            "name",
            "source",
            "canonical_smiles",
            "murcko_scaffold",
            "nearest_train_similarity",
            "nearest_train_similarity_bucket",
            "nearest_train_id",
            "nearest_train_name",
            "nearest_train_source",
            "nearest_train_canonical_smiles",
            "nearest_train_murcko_scaffold",
            "nearest_train_dataset_index",
            "cross_source_near_duplicate",
            "cross_source_strong_near_duplicate",
        ],
    )
    _write_csv(
        output_dir / "benchmark_cross_source_leakage.csv",
        cross_source_rows,
        fieldnames=[
            "compare_split",
            "leakage_type",
            "similarity",
            "strong_near_duplicate",
            "compare_dataset_index",
            "compare_id",
            "compare_name",
            "compare_source",
            "compare_smiles",
            "train_dataset_index",
            "train_id",
            "train_name",
            "train_source",
            "train_smiles",
            "source_pair",
            "match_type",
            "cross_source",
            "compare_inchikey",
            "train_inchikey",
        ],
    )
    (output_dir / "benchmark_novelty_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Benchmark novelty audit complete | "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} | "
        f"exact_train_test_overlap={summary['exact_train_test_overlap_count']} | "
        f"shared_scaffold_fraction={summary['test_scaffold_seen_fraction']:.4f} | "
        f"nn_median={summary['nearest_neighbor_similarity_median'] if summary['nearest_neighbor_similarity_median'] is not None else 'NA'} | "
        f"class={summary['benchmark_novelty_class']}",
        flush=True,
    )
    print(f"Summary JSON: {output_dir / 'benchmark_novelty_audit_summary.json'}", flush=True)
    print(f"Exact overlap CSV: {output_dir / 'benchmark_exact_overlap.csv'}", flush=True)
    print(f"Scaffold overlap CSV: {output_dir / 'benchmark_scaffold_overlap.csv'}", flush=True)
    print(f"Nearest-neighbor CSV: {output_dir / 'benchmark_nearest_neighbor_similarity.csv'}", flush=True)
    print(f"Cross-source leakage CSV: {output_dir / 'benchmark_cross_source_leakage.csv'}", flush=True)


if __name__ == "__main__":
    main()
