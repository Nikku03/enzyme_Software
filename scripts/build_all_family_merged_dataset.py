from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from build_cyp3a4_downstream_subsets import _strict_exact_clean_allowed
from build_cyp3a4_merged_dataset import (
    _dedupe_sorted,
    _infer_label_regime,
    _normalize_cyp,
    _normalize_token,
    _parse_attnsom_record,
    _parse_generic_record,
    _safe_mol_prop,
    _safe_property_keys,
    _scan_sdf_properties,
)


FAMILY_TO_STEM = {
    "CYP1A2": "1A2",
    "CYP2A6": "2A6",
    "CYP2B6": "2B6",
    "CYP2C8": "2C8",
    "CYP2C9": "2C9",
    "CYP2C19": "2C19",
    "CYP2D6": "2D6",
    "CYP2E1": "2E1",
    "CYP3A4": "3A4",
}
STEM_TO_FAMILY = {stem: family for family, stem in FAMILY_TO_STEM.items()}

DEFAULT_METXBIO_CONFIG = {
    "id_fields": ["InChIKey", "PubChem_CID", "_Name"],
    "name_fields": ["_Name", "InChIKey", "PubChem_CID"],
    "citation_fields": ["References"],
    "doi_fields": ["DOI"],
    "source_detail_fields": ["PubChem_CID", "InChIKey"],
    "flag_comments_field": "FLAG_COMMENTS",
    "index_base": 0,
    "confidence_override": "medium",
}
DEFAULT_PENG_CONFIG = {
    "fixed_cyp": "CYP3A4",
    "site_fields": ["PRIMARY_SOM"],
    "id_fields": ["Name", "ID", "_Name"],
    "name_fields": ["Name", "_Name", "ID"],
    "citation_fields": ["Citation"],
    "doi_fields": ["DOI"],
    "source_detail_fields": ["PotentialSOMs", "PredictSOMs"],
    "index_base": 1,
    "confidence_override": "medium",
}
DEFAULT_RUDIK_CONFIG = {
    "fixed_cyp": "CYP3A4",
    "site_fields": ["PRIMARY_SOM"],
    "id_fields": ["ID", "_Name"],
    "name_fields": ["ID", "_Name"],
    "citation_fields": ["Citation"],
    "doi_fields": ["DOI"],
    "source_detail_fields": [],
    "index_base": 1,
    "confidence_override": "medium",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a provenance-preserving merged master dataset across all supported enzyme families."
    )
    parser.add_argument(
        "--output-dir",
        default="data/prepared_training/enzyme_family_merged_master",
        help="Directory where the family-aware merged dataset and audit files will be written.",
    )
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--property-scan-limit", type=int, default=100)
    parser.add_argument(
        "--family-allowlist",
        default="",
        help="Optional comma-separated family allowlist, e.g. CYP3A4,CYP2D6",
    )
    parser.add_argument("--attnsom-root", default="data/ATTNSOM/cyp_dataset")
    parser.add_argument("--cyp-dbs-root", default="CYP_DBs")
    parser.add_argument("--metxbio-sdf", default="metxbio data.cleaned.sdf")
    parser.add_argument("--peng-sdf", default="suppl_info/Peng_external_dataset/Peng_external_dataset.sdf")
    parser.add_argument("--rudik-sdf", default="suppl_info/Rudik_external_dataset/Rudik_external_dataset.sdf")
    parser.add_argument("--trainable-min-rows", type=int, default=50)
    parser.add_argument("--benchmarkable-min-exact-rows", type=int, default=30)
    parser.add_argument("--benchmarkable-min-tiered-rows", type=int, default=20)
    return parser.parse_args()


def _csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in str(raw or "").replace(";", ",").split(",") if token.strip()]


def _source_family_tag_for_name(source_name: Any) -> str:
    token = _normalize_token(source_name)
    if token in {"attnsom", "cyp_dbs_external"}:
        return "attnsom_family"
    if token == "metxbiodb":
        return "metxbiodb_family"
    if token == "peng_external":
        return "peng_family"
    if token == "rudik_external":
        return "rudik_family"
    if token:
        return f"{token}_family"
    return "unknown_family"


def _enzyme_superfamily(family: str) -> str:
    family = str(family or "").strip().upper()
    if family.startswith("CYP"):
        return "CYP"
    return family.split("_", 1)[0] if family else "unknown"


def _hash_group_id(canonical_smiles: str, target_family: str) -> str:
    digest = hashlib.sha1(f"{canonical_smiles}::{target_family}".encode("utf-8")).hexdigest()[:16]
    return f"dup_{digest}"


def _signature_key(annotation: dict[str, Any]) -> str:
    return json.dumps(
        {
            "label_regime": str(annotation.get("label_regime") or ""),
            "target_family": str(annotation.get("target_family") or ""),
            "primary_site_atoms": list(annotation.get("primary_site_atoms") or []),
            "secondary_site_atoms": list(annotation.get("secondary_site_atoms") or []),
            "tertiary_site_atoms": list(annotation.get("tertiary_site_atoms") or []),
            "all_labeled_site_atoms": list(annotation.get("all_labeled_site_atoms") or []),
            "broad_region_annotations": list(annotation.get("broad_region_annotations") or []),
        },
        sort_keys=True,
    )


def _source_detail_strings(annotation: dict[str, Any]) -> list[str]:
    values = [str(annotation.get("site_source") or "")]
    values.extend(str(v) for v in list(annotation.get("source_details") or []) if str(v).strip())
    values.append(str(annotation.get("source_file") or ""))
    return [value for value in values if str(value).strip()]


def _group_conflict_flags(annotations: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    label_regimes = {str(annotation.get("label_regime") or "") for annotation in annotations}
    if len(label_regimes) > 1:
        flags.append("label_regime_mismatch")
    site_sets = [set(int(v) for v in list(annotation.get("all_labeled_site_atoms") or [])) for annotation in annotations]
    nonempty_sets = [values for values in site_sets if values]
    unique_site_sets = {tuple(sorted(values)) for values in nonempty_sets}
    if len(unique_site_sets) > 1:
        flags.append("site_set_mismatch")
    if len(unique_site_sets) > 1 and all(nonempty_sets) and any(not (left & right) for idx, left in enumerate(nonempty_sets) for right in nonempty_sets[idx + 1 :]):
        flags.append("site_set_disjoint")
    if any(str(annotation.get("label_regime") or "") == "broad_region" for annotation in annotations):
        flags.append("contains_broad_region_label")
    if any(list(annotation.get("secondary_site_atoms") or []) or list(annotation.get("tertiary_site_atoms") or []) for annotation in annotations):
        if len({tuple(annotation.get("primary_site_atoms") or []) for annotation in annotations}) > 1:
            flags.append("tier_assignment_mismatch")
    return _dedupe_sorted(flags)


def _preferred_annotation(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    regime_rank = {
        "tiered_multisite": 4,
        "multi_exact": 3,
        "single_exact": 2,
        "broad_region": 1,
        "unknown": 0,
    }
    confidence_rank = {
        "validated_gold": 4,
        "validated": 3,
        "high": 2,
        "medium": 1,
        "low": 0,
    }
    ordered = sorted(
        annotations,
        key=lambda row: (
            -regime_rank.get(str(row.get("label_regime") or ""), 0),
            -confidence_rank.get(str(row.get("confidence") or "").lower(), 0),
            -int(bool(str(row.get("doi") or "").strip())),
            -int(bool(str(row.get("citation") or "").strip())),
            str(row.get("source") or ""),
            str(row.get("source_record_id") or ""),
        ),
    )
    return dict(ordered[0])


def _build_row_from_annotations(
    annotations: list[dict[str, Any]],
    *,
    duplicate_group_id: str,
    merge_policy_used: str,
    conflict_flags: list[str],
    all_families_for_molecule: list[str],
) -> dict[str, Any]:
    preferred = _preferred_annotation(annotations)
    union_primary = sorted({int(v) for annotation in annotations for v in list(annotation.get("primary_site_atoms") or [])})
    union_secondary = sorted({int(v) for annotation in annotations for v in list(annotation.get("secondary_site_atoms") or [])})
    union_tertiary = sorted({int(v) for annotation in annotations for v in list(annotation.get("tertiary_site_atoms") or [])})
    union_sites = sorted({int(v) for annotation in annotations for v in list(annotation.get("all_labeled_site_atoms") or [])})
    broad_region_annotations = _dedupe_sorted(
        [str(v) for annotation in annotations for v in list(annotation.get("broad_region_annotations") or []) if str(v).strip()]
    )
    citations = _dedupe_sorted([str(annotation.get("citation") or "") for annotation in annotations if str(annotation.get("citation") or "").strip()])
    dois = _dedupe_sorted([str(annotation.get("doi") or "") for annotation in annotations if str(annotation.get("doi") or "").strip()])
    merged_from_sources = sorted({str(annotation.get("site_source") or annotation.get("source") or "") for annotation in annotations if str(annotation.get("site_source") or annotation.get("source") or "").strip()})
    source_details = _dedupe_sorted([value for annotation in annotations for value in _source_detail_strings(annotation)])
    target_family = str(preferred.get("target_family") or preferred.get("enzyme_family") or "")
    label_regime = _infer_label_regime(
        primary=union_primary,
        secondary=union_secondary,
        tertiary=union_tertiary,
        all_atoms=union_sites,
        broad_region_annotations=broad_region_annotations,
    )
    if merge_policy_used == "conflict_preserved":
        label_regime = str(preferred.get("label_regime") or label_regime)
    site_source_value = str(preferred.get("site_source") or preferred.get("source") or "")
    if len(merged_from_sources) > 1 and merge_policy_used != "conflict_preserved":
        site_source_value = "merged_multi_source"
    effective_sites = union_sites if merge_policy_used != "conflict_preserved" else list(preferred.get("site_atoms") or [])
    row = {
        "id": str(preferred.get("source_record_id") or preferred.get("molecule_name") or preferred.get("canonical_smiles") or duplicate_group_id),
        "molecule_name": str(preferred.get("molecule_name") or preferred.get("name") or ""),
        "name": str(preferred.get("name") or preferred.get("molecule_name") or ""),
        "canonical_smiles": str(preferred.get("canonical_smiles") or ""),
        "smiles": str(preferred.get("canonical_smiles") or ""),
        "inchi_key": str(preferred.get("inchi_key") or ""),
        "formula": str(preferred.get("formula") or ""),
        "atom_count": int(preferred.get("atom_count") or 0),
        "enzyme_family": str(target_family),
        "primary_family": str(target_family),
        "all_families": list(all_families_for_molecule),
        "target_family": str(target_family),
        "family_confidence": str(preferred.get("family_confidence") or "medium"),
        "enzyme_superfamily": str(preferred.get("enzyme_superfamily") or _enzyme_superfamily(target_family)),
        "source_family_tag": str(preferred.get("source_family_tag") or _source_family_tag_for_name(preferred.get("source"))),
        "source_family": str(preferred.get("source_family") or preferred.get("source_family_tag") or _source_family_tag_for_name(preferred.get("source"))),
        "target_cyp": str(target_family),
        "primary_cyp": str(target_family),
        "all_cyps": list(all_families_for_molecule),
        "site_atoms": list(effective_sites),
        "metabolism_sites": list(effective_sites),
        "som": [{"atom_idx": int(idx)} for idx in effective_sites],
        "site_count": int(len(effective_sites)),
        "is_multisite": bool(len(effective_sites) > 1),
        "molecule_source": str(preferred.get("source") or ""),
        "source": str(preferred.get("source") or ""),
        "site_source": str(site_source_value),
        "source_record_id": str(preferred.get("source_record_id") or ""),
        "source_file": str(preferred.get("source_file") or ""),
        "source_details": source_details,
        "citation": " | ".join(citations),
        "doi": " | ".join(dois),
        "primary_site_atoms": list(union_primary if merge_policy_used != "conflict_preserved" else list(preferred.get("primary_site_atoms") or [])),
        "secondary_site_atoms": list(union_secondary if merge_policy_used != "conflict_preserved" else list(preferred.get("secondary_site_atoms") or [])),
        "tertiary_site_atoms": list(union_tertiary if merge_policy_used != "conflict_preserved" else list(preferred.get("tertiary_site_atoms") or [])),
        "all_labeled_site_atoms": list(union_sites if merge_policy_used != "conflict_preserved" else list(preferred.get("all_labeled_site_atoms") or [])),
        "label_regime": str(label_regime),
        "broad_region_annotations": list(broad_region_annotations if merge_policy_used != "conflict_preserved" else list(preferred.get("broad_region_annotations") or [])),
        "merged_from_sources": list(merged_from_sources),
        "merged_from_source_families": sorted({_source_family_tag_for_name(value) for value in merged_from_sources}),
        "duplicate_group_id": str(duplicate_group_id),
        "merge_policy_used": str(merge_policy_used),
        "conflict_flags": list(conflict_flags),
        "per_source_site_annotations": [
            {
                "site_source": str(annotation.get("site_source") or annotation.get("source") or ""),
                "source_record_id": str(annotation.get("source_record_id") or ""),
                "source_file": str(annotation.get("source_file") or ""),
                "citation": str(annotation.get("citation") or ""),
                "doi": str(annotation.get("doi") or ""),
                "label_regime": str(annotation.get("label_regime") or ""),
                "enzyme_family": str(annotation.get("enzyme_family") or annotation.get("target_family") or ""),
                "target_family": str(annotation.get("target_family") or annotation.get("enzyme_family") or ""),
                "family_confidence": str(annotation.get("family_confidence") or ""),
                "primary_site_atoms": list(annotation.get("primary_site_atoms") or []),
                "secondary_site_atoms": list(annotation.get("secondary_site_atoms") or []),
                "tertiary_site_atoms": list(annotation.get("tertiary_site_atoms") or []),
                "all_labeled_site_atoms": list(annotation.get("all_labeled_site_atoms") or []),
                "broad_region_annotations": list(annotation.get("broad_region_annotations") or []),
            }
            for annotation in annotations
        ],
        "full_xtb_status": "unknown",
        "confidence": str(preferred.get("confidence") or "medium"),
        "citation_available": bool(citations),
        "doi_available": bool(dois),
        "site_provenance_preserved": True,
        "reactions": [],
        "symmetry_expanded": False,
        "symmetry_expanded_added_atoms": [],
        "symmetry_expanded_groups": [],
    }
    return row


def _merge_duplicate_group(
    annotations: list[dict[str, Any]],
    *,
    all_families_for_molecule: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    duplicate_group_id = _hash_group_id(str(annotations[0].get("canonical_smiles") or ""), str(annotations[0].get("target_family") or ""))
    sources = sorted({str(annotation.get("source") or "") for annotation in annotations})
    signatures = {_signature_key(annotation) for annotation in annotations}
    conflict_flags = _group_conflict_flags(annotations)
    if len(annotations) == 1:
        policy = "single_source"
        rows = [_build_row_from_annotations(annotations, duplicate_group_id=duplicate_group_id, merge_policy_used=policy, conflict_flags=[], all_families_for_molecule=all_families_for_molecule)]
    elif len(signatures) == 1:
        policy = "exact_agreement"
        rows = [_build_row_from_annotations(annotations, duplicate_group_id=duplicate_group_id, merge_policy_used=policy, conflict_flags=[], all_families_for_molecule=all_families_for_molecule)]
    elif "contains_broad_region_label" not in conflict_flags and "site_set_disjoint" not in conflict_flags:
        policy = "partial_agreement_union"
        rows = [_build_row_from_annotations(annotations, duplicate_group_id=duplicate_group_id, merge_policy_used=policy, conflict_flags=conflict_flags, all_families_for_molecule=all_families_for_molecule)]
    else:
        policy = "conflict_preserved"
        rows = [
            _build_row_from_annotations([annotation], duplicate_group_id=duplicate_group_id, merge_policy_used=policy, conflict_flags=conflict_flags, all_families_for_molecule=all_families_for_molecule)
            for annotation in sorted(annotations, key=lambda row: (str(row.get("source") or ""), str(row.get("source_record_id") or "")))
        ]
    audit = {
        "duplicate_group_id": str(duplicate_group_id),
        "group_size": int(len(annotations)),
        "output_row_count": int(len(rows)),
        "merge_policy_used": str(policy),
        "sources": list(sources),
        "target_family": str(annotations[0].get("target_family") or ""),
        "all_families_for_molecule": list(all_families_for_molecule),
        "conflict_flags": list(conflict_flags),
        "canonical_smiles": str(annotations[0].get("canonical_smiles") or ""),
        "inchi_keys": sorted({str(annotation.get("inchi_key") or "") for annotation in annotations if str(annotation.get("inchi_key") or "").strip()}),
        "label_regimes": sorted({str(annotation.get("label_regime") or "") for annotation in annotations}),
        "site_sets": [list(annotation.get("all_labeled_site_atoms") or []) for annotation in annotations],
        "source_record_ids": [str(annotation.get("source_record_id") or "") for annotation in annotations],
    }
    return rows, audit


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    fieldnames = sorted({key for row in normalized_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(row)


def _family_allowed(family: str, allowlist: set[str]) -> bool:
    return not allowlist or str(family) in allowlist


def _safe_scan_sdf_properties(path: Path, *, limit: int) -> dict[str, Any]:
    try:
        return _scan_sdf_properties(path, limit=limit)
    except UnicodeDecodeError:
        prop_counts = Counter()
        mol_count = 0
        valid_count = 0
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        for mol in supplier:
            mol_count += 1
            if mol is None:
                continue
            valid_count += 1
            for key in _safe_property_keys(mol):
                prop_counts[str(key)] += 1
            if mol_count >= int(limit):
                break
        return {
            "scanned_records": int(mol_count),
            "valid_records": int(valid_count),
            "property_counts": dict(sorted(prop_counts.items())),
            "scan_fallback": "safe_property_keys_after_unicode_decode_error",
        }


def _enrich_record(
    record: dict[str, Any],
    *,
    target_family: str,
    source_name: str,
    family_confidence: str,
) -> dict[str, Any]:
    out = dict(record)
    out["enzyme_family"] = str(target_family)
    out["primary_family"] = str(target_family)
    out["all_families"] = [str(target_family)]
    out["target_family"] = str(target_family)
    out["target_cyp"] = str(target_family)
    out["primary_cyp"] = str(target_family)
    out["all_cyps"] = [str(target_family)]
    out["family_confidence"] = str(family_confidence)
    out["enzyme_superfamily"] = _enzyme_superfamily(target_family)
    out["source_family_tag"] = _source_family_tag_for_name(source_name)
    out["source_family"] = out["source_family_tag"]
    return out


def _attnsom_specs(root: Path, source_name: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for stem, family in sorted(STEM_TO_FAMILY.items()):
        path = root / f"{stem}.sdf"
        if path.exists():
            specs.append(
                {
                    "source_name": source_name,
                    "source_file": path,
                    "parser_family": "attnsom_tiered",
                    "target_family": family,
                    "family_confidence": "fixed_filename",
                    "config": {},
                }
            )
    return specs


def _build_source_specs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowlist = {_normalize_cyp(token) for token in _csv_tokens(args.family_allowlist)}
    specs: list[dict[str, Any]] = []
    skipped_specs: list[dict[str, Any]] = []

    attnsom_root = (ROOT / args.attnsom_root).resolve()
    cyp_dbs_root = (ROOT / args.cyp_dbs_root).resolve()
    metxbio_path = (ROOT / args.metxbio_sdf).resolve()
    peng_path = (ROOT / args.peng_sdf).resolve()
    rudik_path = (ROOT / args.rudik_sdf).resolve()

    for spec in _attnsom_specs(attnsom_root, "ATTNSOM"):
        if _family_allowed(str(spec["target_family"]), allowlist):
            specs.append(spec)
    for spec in _attnsom_specs(cyp_dbs_root, "CYP_DBs_external"):
        if _family_allowed(str(spec["target_family"]), allowlist):
            specs.append(spec)

    if metxbio_path.exists():
        specs.append(
            {
                "source_name": "MetXBioDB",
                "source_file": metxbio_path,
                "parser_family": "metxbio_multi_bom",
                "config": dict(DEFAULT_METXBIO_CONFIG),
            }
        )
    if peng_path.exists() and _family_allowed("CYP3A4", allowlist):
        specs.append(
            {
                "source_name": "Peng_external",
                "source_file": peng_path,
                "parser_family": "flat_atom_list",
                "target_family": "CYP3A4",
                "family_confidence": "fixed_source",
                "config": dict(DEFAULT_PENG_CONFIG),
            }
        )
    if rudik_path.exists() and _family_allowed("CYP3A4", allowlist):
        specs.append(
            {
                "source_name": "Rudik_external",
                "source_file": rudik_path,
                "parser_family": "flat_atom_list",
                "target_family": "CYP3A4",
                "family_confidence": "fixed_source",
                "config": dict(DEFAULT_RUDIK_CONFIG),
            }
        )

    hlm_path = cyp_dbs_root / "HLM.sdf"
    if hlm_path.exists():
        skipped_specs.append(
            {
                "source_name": "CYP_DBs_external",
                "source_file": str(hlm_path),
                "reason": "unsupported_ambiguous_family",
                "details": "HLM.sdf is present but not mapped to a specific enzyme family.",
            }
        )
    return specs, skipped_specs


def _parse_metxbio_multi_family_records(
    mol: Chem.Mol,
    *,
    source_name: str,
    source_file: Path,
    allowlist: set[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family, stem in sorted(FAMILY_TO_STEM.items()):
        if not _family_allowed(family, allowlist):
            continue
        bom_field = f"BOM_{stem}"
        if not str(_safe_mol_prop(mol, bom_field)).strip():
            continue
        family_config = dict(config)
        family_config["fixed_cyp"] = family
        family_config["bom_field"] = bom_field
        record = _parse_generic_record(
            mol,
            source_name=source_name,
            source_file=source_file,
            target_cyp=family,
            parser_family="metxbio_bom",
            config=family_config,
        )
        if record is not None:
            records.append(
                _enrich_record(
                    record,
                    target_family=family,
                    source_name=source_name,
                    family_confidence="field_specific",
                )
            )
    return records


def _basic_family_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_breakdown = Counter(str(row.get("molecule_source") or "unknown") for row in rows)
    source_family_breakdown = Counter(str(row.get("source_family_tag") or "unknown") for row in rows)
    label_regime_breakdown = Counter(str(row.get("label_regime") or "unknown") for row in rows)
    merge_policy_breakdown = Counter(str(row.get("merge_policy_used") or "unknown") for row in rows)
    return {
        "retained_row_count": int(len(rows)),
        "source_breakdown": dict(sorted(source_breakdown.items(), key=lambda item: (-item[1], item[0]))),
        "source_family_breakdown": dict(sorted(source_family_breakdown.items(), key=lambda item: (-item[1], item[0]))),
        "label_regime_breakdown": dict(sorted(label_regime_breakdown.items(), key=lambda item: (-item[1], item[0]))),
        "merge_policy_breakdown": dict(sorted(merge_policy_breakdown.items(), key=lambda item: (-item[1], item[0]))),
        "citation_count": int(sum(1 for row in rows if bool(str(row.get("citation") or "").strip()))),
        "doi_count": int(sum(1 for row in rows if bool(str(row.get("doi") or "").strip()))),
        "multisite_count": int(sum(1 for row in rows if bool(row.get("is_multisite")))),
        "duplicate_group_count": int(len({str(row.get("duplicate_group_id") or "") for row in rows if str(row.get("duplicate_group_id") or "").strip()})),
        "conflict_count": int(sum(1 for row in rows if list(row.get("conflict_flags") or []))),
    }


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowlist = {_normalize_cyp(token) for token in _csv_tokens(args.family_allowlist)}
    source_specs, skipped_specs = _build_source_specs(args)

    parsed_annotations: list[dict[str, Any]] = []
    per_source_summary: list[dict[str, Any]] = []
    property_scans: dict[str, Any] = {}

    for spec in source_specs:
        input_sdf = Path(spec["source_file"]).resolve()
        if not input_sdf.exists():
            raise FileNotFoundError(f"Input SDF not found: {input_sdf}")
        source_name = str(spec["source_name"])
        parser_family = str(spec["parser_family"])
        config = dict(spec.get("config") or {})
        spec_key = f"{source_name}::{input_sdf.name}"
        property_scans[spec_key] = _safe_scan_sdf_properties(input_sdf, limit=int(args.property_scan_limit))
        parsed_count = 0
        retained_annotation_count = 0
        retained_unique_molecule_count = 0
        skipped_invalid = 0
        skipped_non_target_family = 0
        supplier = Chem.SDMolSupplier(str(input_sdf), removeHs=False)
        for mol in supplier:
            parsed_count += 1
            if mol is None:
                skipped_invalid += 1
                continue
            records: list[dict[str, Any]] = []
            if parser_family == "attnsom_tiered":
                target_family = str(spec["target_family"])
                record = _parse_attnsom_record(
                    mol,
                    source_name=source_name,
                    source_file=input_sdf,
                    target_cyp=target_family,
                )
                if record is not None:
                    records = [
                        _enrich_record(
                            record,
                            target_family=target_family,
                            source_name=source_name,
                            family_confidence=str(spec.get("family_confidence") or "fixed_filename"),
                        )
                    ]
            elif parser_family == "flat_atom_list":
                target_family = str(spec["target_family"])
                record = _parse_generic_record(
                    mol,
                    source_name=source_name,
                    source_file=input_sdf,
                    target_cyp=target_family,
                    parser_family="flat_atom_list",
                    config=config,
                )
                if record is not None:
                    records = [
                        _enrich_record(
                            record,
                            target_family=target_family,
                            source_name=source_name,
                            family_confidence=str(spec.get("family_confidence") or "fixed_source"),
                        )
                    ]
            elif parser_family == "metxbio_multi_bom":
                records = _parse_metxbio_multi_family_records(
                    mol,
                    source_name=source_name,
                    source_file=input_sdf,
                    allowlist=allowlist,
                    config=config,
                )
            else:
                raise ValueError(f"Unsupported parser_family={parser_family}")
            if not records:
                if parser_family == "metxbio_multi_bom":
                    skipped_non_target_family += 1
                else:
                    skipped_invalid += 1
                continue
            retained_unique_molecule_count += 1
            retained_annotation_count += len(records)
            parsed_annotations.extend(records)
        per_source_summary.append(
            {
                "source_name": source_name,
                "source_file": str(input_sdf),
                "parser_family": parser_family,
                "parsed_count": int(parsed_count),
                "retained_annotation_count": int(retained_annotation_count),
                "retained_unique_molecule_count": int(retained_unique_molecule_count),
                "excluded_non_target_family_count": int(skipped_non_target_family),
                "excluded_invalid_or_missing_label_count": int(skipped_invalid),
                "fixed_family": str(spec.get("target_family") or ""),
            }
        )

    molecule_family_map: dict[str, set[str]] = defaultdict(set)
    for annotation in parsed_annotations:
        canonical_smiles = str(annotation.get("canonical_smiles") or "")
        target_family = str(annotation.get("target_family") or "")
        if canonical_smiles and target_family:
            molecule_family_map[canonical_smiles].add(target_family)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for annotation in parsed_annotations:
        key = (str(annotation.get("canonical_smiles") or ""), str(annotation.get("target_family") or ""))
        grouped[key].append(annotation)

    merged_rows: list[dict[str, Any]] = []
    duplicate_audit_rows: list[dict[str, Any]] = []
    conflict_audit_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    merge_policy_counts = Counter()

    for key in sorted(grouped):
        canonical_smiles, target_family = key
        all_families_for_molecule = sorted(molecule_family_map.get(canonical_smiles, {target_family}))
        rows, audit = _merge_duplicate_group(
            grouped[key],
            all_families_for_molecule=all_families_for_molecule,
        )
        if len(all_families_for_molecule) > 1:
            for row in rows:
                conflict_flags = set(str(v) for v in list(row.get("conflict_flags") or []))
                conflict_flags.add("multi_family_assignment")
                row["conflict_flags"] = sorted(conflict_flags)
        merged_rows.extend(rows)
        duplicate_audit_rows.append(audit)
        merge_policy_counts[str(audit["merge_policy_used"])] += 1
        if list(audit.get("conflict_flags") or []):
            conflict_audit_rows.append(audit)
        for row in rows:
            for annotation in list(row.get("per_source_site_annotations") or []):
                provenance_rows.append(
                    {
                        "output_row_id": str(row.get("id") or ""),
                        "duplicate_group_id": str(row.get("duplicate_group_id") or ""),
                        "merge_policy_used": str(row.get("merge_policy_used") or ""),
                        "output_label_regime": str(row.get("label_regime") or ""),
                        "enzyme_family": str(row.get("enzyme_family") or ""),
                        "site_source": str(annotation.get("site_source") or ""),
                        "source_record_id": str(annotation.get("source_record_id") or ""),
                        "source_file": str(annotation.get("source_file") or ""),
                        "annotation_label_regime": str(annotation.get("label_regime") or ""),
                        "annotation_family": str(annotation.get("target_family") or annotation.get("enzyme_family") or ""),
                        "primary_site_atoms": list(annotation.get("primary_site_atoms") or []),
                        "secondary_site_atoms": list(annotation.get("secondary_site_atoms") or []),
                        "tertiary_site_atoms": list(annotation.get("tertiary_site_atoms") or []),
                        "all_labeled_site_atoms": list(annotation.get("all_labeled_site_atoms") or []),
                        "citation": str(annotation.get("citation") or ""),
                        "doi": str(annotation.get("doi") or ""),
                    }
                )

    merged_rows = sorted(
        merged_rows,
        key=lambda row: (
            str(row.get("target_family") or ""),
            str(row.get("canonical_smiles") or ""),
            str(row.get("source") or ""),
            str(row.get("id") or ""),
        ),
    )
    duplicate_group_count = sum(1 for values in grouped.values() if len(values) > 1)
    label_regime_counts = Counter(str(row.get("label_regime") or "unknown") for row in merged_rows)
    family_counts = Counter(str(row.get("target_family") or "unknown") for row in merged_rows)
    citation_count = sum(1 for row in merged_rows if bool(str(row.get("citation") or "").strip()))
    doi_count = sum(1 for row in merged_rows if bool(str(row.get("doi") or "").strip()))
    site_provenance_count = sum(1 for row in merged_rows if bool(row.get("per_source_site_annotations")))

    per_family_summary: dict[str, Any] = {}
    trainable_families: list[str] = []
    benchmarkable_families: list[str] = []
    audit_only_families: list[str] = []
    family_count_rows: list[dict[str, Any]] = []
    for family in sorted(family_counts):
        family_rows = [row for row in merged_rows if str(row.get("target_family") or "") == family]
        strict_exact_count = sum(
            1
            for row in family_rows
            if _strict_exact_clean_allowed(
                row,
                include_partial_agreement_union=True,
            )[0]
        )
        tiered_count = sum(1 for row in family_rows if str(row.get("label_regime") or "") == "tiered_multisite")
        broad_count = sum(1 for row in family_rows if str(row.get("label_regime") or "") == "broad_region")
        trainable = (strict_exact_count + tiered_count) >= int(args.trainable_min_rows)
        benchmarkable = strict_exact_count >= int(args.benchmarkable_min_exact_rows) or tiered_count >= int(args.benchmarkable_min_tiered_rows)
        if trainable:
            trainable_families.append(family)
        if benchmarkable:
            benchmarkable_families.append(family)
        if not trainable and not benchmarkable:
            audit_only_families.append(family)
        per_family_summary[family] = {
            **_basic_family_stats(family_rows),
            "strict_exact_clean_candidate_count": int(strict_exact_count),
            "tiered_multisite_count": int(tiered_count),
            "broad_region_count": int(broad_count),
            "trainable": bool(trainable),
            "benchmarkable": bool(benchmarkable),
        }
        family_count_rows.append(
            {
                "family": family,
                "retained_row_count": int(len(family_rows)),
                "strict_exact_clean_candidate_count": int(strict_exact_count),
                "tiered_multisite_count": int(tiered_count),
                "broad_region_count": int(broad_count),
                "trainable": int(trainable),
                "benchmarkable": int(benchmarkable),
            }
        )

    dataset_payload = {
        "n_drugs": int(len(merged_rows)),
        "n_site_labeled": int(sum(1 for row in merged_rows if list(row.get("all_labeled_site_atoms") or []))),
        "summary": {
            "total_input_sources": int(len(source_specs)),
            "total_parsed_rows": int(sum(item["parsed_count"] for item in per_source_summary)),
            "total_retained_rows": int(len(parsed_annotations)),
            "total_final_merged_rows": int(len(merged_rows)),
            "discovered_families": sorted(family_counts),
            "label_regime_counts": dict(sorted(label_regime_counts.items())),
            "merge_policy_counts": dict(sorted(merge_policy_counts.items())),
        },
        "build_stats": {
            "family_allowlist": sorted(allowlist),
            "source_specs": [
                {
                    "source_name": spec["source_name"],
                    "source_file": str(spec["source_file"]),
                    "parser_family": spec["parser_family"],
                    "fixed_family": str(spec.get("target_family") or ""),
                    "config_keys": sorted(list((spec.get("config") or {}).keys())),
                }
                for spec in source_specs
            ],
            "skipped_source_specs": skipped_specs,
            "per_source_summary": per_source_summary,
            "property_scans": property_scans,
            "duplicate_group_count": int(duplicate_group_count),
            "exact_agreement_merge_count": int(merge_policy_counts.get("exact_agreement", 0)),
            "partial_agreement_merge_count": int(merge_policy_counts.get("partial_agreement_union", 0)),
            "conflict_preserved_count": int(merge_policy_counts.get("conflict_preserved", 0)),
            "single_source_count": int(merge_policy_counts.get("single_source", 0)),
            "citation_count": int(citation_count),
            "doi_count": int(doi_count),
            "site_provenance_preserved_count": int(site_provenance_count),
        },
        "drugs": merged_rows,
    }
    summary = {
        "total_input_sources": int(len(source_specs)),
        "total_parsed_rows": int(sum(item["parsed_count"] for item in per_source_summary)),
        "total_retained_rows": int(len(parsed_annotations)),
        "total_final_merged_rows": int(len(merged_rows)),
        "discovered_families": sorted(family_counts),
        "dataset_output": str(output_dir / "enzyme_family_merged_master_dataset.json"),
        "duplicate_group_count": int(duplicate_group_count),
        "exact_agreement_merge_count": int(merge_policy_counts.get("exact_agreement", 0)),
        "partial_agreement_merge_count": int(merge_policy_counts.get("partial_agreement_union", 0)),
        "conflict_preserved_count": int(merge_policy_counts.get("conflict_preserved", 0)),
        "single_source_count": int(merge_policy_counts.get("single_source", 0)),
        "label_regime_counts": dict(sorted(label_regime_counts.items())),
        "citation_count": int(citation_count),
        "doi_count": int(doi_count),
        "site_provenance_preserved_count": int(site_provenance_count),
        "trainable_family_count": int(len(trainable_families)),
        "benchmarkable_family_count": int(len(benchmarkable_families)),
        "insufficient_data_family_count": int(len(audit_only_families)),
        "trainable_families": sorted(trainable_families),
        "benchmarkable_families": sorted(benchmarkable_families),
        "audit_only_families": sorted(audit_only_families),
        "per_source_summary": per_source_summary,
        "per_family_summary": per_family_summary,
        "merge_policy": {
            "exact_agreement": "Merged to one family-scoped row when all source-specific label signatures matched exactly within that family.",
            "partial_agreement_union": "Merged to one family-scoped row with union site set and preserved per-source annotations when site sets differed but were not disjoint/broad-region conflicts.",
            "conflict_preserved": "Kept source-specific family-scoped rows separate with shared duplicate_group_id when label regimes conflicted, broad-region labels were involved, or site sets were disjoint.",
        },
        "family_detection_policy": {
            "attnsom_and_cyp_dbs": "Family is derived from the fixed SDF filename stem such as 1A2.sdf -> CYP1A2.",
            "metxbiodb": "Family is derived from explicit BOM_* fields such as BOM_2D6 or BOM_3A4; a record can emit multiple family-scoped annotations.",
            "peng_external": "Current file is treated as fixed CYP3A4 because no broader family field is present.",
            "rudik_external": "Current file is treated as fixed CYP3A4 because no broader family field is present.",
        },
        "normalization_policy": {
            "sanitization": "RDKit SDF parsing + canonical SMILES generation via the existing canonicalization mapping workflow",
            "hydrogen_handling": "RemoveHs before canonicalization to stay consistent with existing repo builders",
            "salt_fragment_policy": "No extra salt stripping or charge neutralization beyond RDKit canonicalization",
            "atom_index_policy": "Site indices are remapped through the existing canonical atom-order mapping when needed",
        },
        "outputs": {
            "merged_dataset_json": str(output_dir / "enzyme_family_merged_master_dataset.json"),
            "summary_json": str(output_dir / "enzyme_family_merged_master_summary.json"),
            "source_breakdown_csv": str(output_dir / "enzyme_family_source_breakdown.csv"),
            "duplicate_audit_csv": str(output_dir / "enzyme_family_duplicate_audit.csv"),
            "conflict_audit_csv": str(output_dir / "enzyme_family_conflict_audit.csv"),
            "provenance_audit_csv": str(output_dir / "enzyme_family_provenance_audit.csv"),
            "family_counts_csv": str(output_dir / "enzyme_family_family_counts.csv"),
        },
    }

    _write_csv(output_dir / "enzyme_family_duplicate_audit.csv", duplicate_audit_rows)
    _write_csv(output_dir / "enzyme_family_conflict_audit.csv", conflict_audit_rows)
    _write_csv(output_dir / "enzyme_family_source_breakdown.csv", per_source_summary)
    _write_csv(output_dir / "enzyme_family_provenance_audit.csv", provenance_rows)
    _write_csv(output_dir / "enzyme_family_family_counts.csv", family_count_rows)
    (output_dir / "enzyme_family_merged_master_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.summary_only:
        (output_dir / "enzyme_family_merged_master_dataset.json").write_text(json.dumps(dataset_payload, indent=2), encoding="utf-8")

    print(
        "Family-aware merged master build complete | "
        f"sources={len(source_specs)} | retained_annotations={len(parsed_annotations)} | "
        f"final_rows={len(merged_rows)} | families={len(family_counts)}",
        flush=True,
    )
    print(f"Summary JSON: {output_dir / 'enzyme_family_merged_master_summary.json'}", flush=True)
    print(f"Merged dataset: {output_dir / 'enzyme_family_merged_master_dataset.json'}", flush=True)


if __name__ == "__main__":
    main()
