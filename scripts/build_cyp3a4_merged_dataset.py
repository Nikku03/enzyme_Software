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
from rdkit.Chem import rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a provenance-preserving merged CYP3A4 dataset from one or more source SDF files."
    )
    parser.add_argument("--input-sdf", action="append", default=[], help="Input SDF path. Repeat once per source.")
    parser.add_argument("--source-name", action="append", default=[], help="Logical source name. Repeat once per source.")
    parser.add_argument(
        "--parser-family",
        action="append",
        default=[],
        help="Parser family for each input: attnsom_tiered, flat_atom_list, tiered_atom_lists.",
    )
    parser.add_argument(
        "--source-config-json",
        action="append",
        default=[],
        help="Optional per-source JSON config. Repeat once per source or omit entirely.",
    )
    parser.add_argument("--target-cyp", default="CYP3A4")
    parser.add_argument("--output-dir", default="data/prepared_training/cyp3a4_merged_dataset")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--property-scan-limit", type=int, default=100)
    return parser.parse_args()


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_cyp(value: Any) -> str:
    token = _normalize_token(value).replace("cyp", "")
    token = token.replace("_", "")
    if token == "3a4":
        return "CYP3A4"
    return str(value or "").strip().upper()


def _safe_mol_prop(mol: Chem.Mol, name: str) -> str:
    if not mol.HasProp(name):
        return ""
    try:
        return str(mol.GetProp(name))
    except Exception:
        try:
            return str(mol.GetPropsAsDict(includePrivate=False, includeComputed=False).get(name, ""))
        except Exception:
            return ""


def _safe_property_keys(mol: Chem.Mol) -> list[str]:
    try:
        return sorted(str(key) for key in mol.GetPropsAsDict(includePrivate=False, includeComputed=False).keys())
    except Exception:
        out: list[str] = []
        for key in mol.GetPropNames(includePrivate=False, includeComputed=False):
            try:
                out.append(str(key))
            except Exception:
                continue
        return sorted(set(out))


def _first_nonempty_prop(mol: Chem.Mol, names: list[str]) -> str:
    for name in names:
        value = _safe_mol_prop(mol, name)
        if str(value).strip():
            return str(value).strip()
    return ""


def _parse_index_tokens(raw: str, *, index_base: int) -> list[int]:
    values: list[int] = []
    for token in str(raw or "").replace(",", " ").replace(";", " ").split():
        try:
            values.append(int(token) - int(index_base))
        except Exception:
            continue
    return sorted(set(int(v) for v in values if int(v) >= 0))


def _canonicalize_mol_with_mapping(mol: Chem.Mol) -> tuple[str | None, list[int] | None]:
    base = Chem.RemoveHs(mol)
    canonical = Chem.MolToSmiles(base, canonical=True)
    rebuilt = Chem.MolFromSmiles(canonical)
    if rebuilt is None:
        return None, None
    match = base.GetSubstructMatch(rebuilt)
    if match:
        return canonical, list(match)
    reverse = rebuilt.GetSubstructMatch(base)
    if reverse:
        inverted: list[int] = [0] * len(reverse)
        for rebuilt_idx, base_idx in enumerate(reverse):
            inverted[int(base_idx)] = int(rebuilt_idx)
        return canonical, inverted
    return canonical, None


def _remap_indices(indices: list[int], mapping: list[int] | None) -> list[int]:
    if not indices:
        return []
    if not mapping:
        return sorted(set(int(idx) for idx in indices if int(idx) >= 0))
    remapped = []
    for idx in indices:
        idx = int(idx)
        if 0 <= idx < len(mapping):
            remapped.append(int(mapping[idx]))
    return sorted(set(remapped))


def _canonical_identity_from_mol(mol: Chem.Mol) -> tuple[str | None, list[int] | None, Chem.Mol | None]:
    canonical_smiles, mapping = _canonicalize_mol_with_mapping(mol)
    if canonical_smiles is None:
        return None, None, None
    canonical_mol = Chem.MolFromSmiles(canonical_smiles)
    return canonical_smiles, mapping, canonical_mol


def _inchikey_from_mol(mol: Chem.Mol | None) -> str:
    if mol is None:
        return ""
    try:
        return str(Chem.MolToInchiKey(mol) or "")
    except Exception:
        return ""


def _formula_from_mol(mol: Chem.Mol | None) -> str:
    if mol is None:
        return ""
    try:
        return str(rdMolDescriptors.CalcMolFormula(mol) or "")
    except Exception:
        return ""


def _dedupe_sorted(values: list[Any]) -> list[Any]:
    out = []
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _infer_label_regime(*, primary: list[int], secondary: list[int], tertiary: list[int], all_atoms: list[int], broad_region_annotations: list[str]) -> str:
    if primary or secondary or tertiary:
        if secondary or tertiary:
            return "tiered_multisite"
        if len(primary) <= 1:
            return "single_exact"
        return "multi_exact"
    if all_atoms:
        if len(all_atoms) <= 1:
            return "single_exact"
        return "multi_exact"
    if broad_region_annotations:
        return "broad_region"
    return "unknown"


def _annotation_signature(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_regime": str(annotation.get("label_regime") or ""),
        "primary_site_atoms": list(annotation.get("primary_site_atoms") or []),
        "secondary_site_atoms": list(annotation.get("secondary_site_atoms") or []),
        "tertiary_site_atoms": list(annotation.get("tertiary_site_atoms") or []),
        "all_labeled_site_atoms": list(annotation.get("all_labeled_site_atoms") or []),
        "broad_region_annotations": list(annotation.get("broad_region_annotations") or []),
    }


def _source_detail_strings(annotation: dict[str, Any]) -> list[str]:
    values = [str(annotation.get("site_source") or "")]
    values.extend(str(v) for v in list(annotation.get("source_details") or []) if str(v).strip())
    values.append(str(annotation.get("source_file") or ""))
    return [value for value in values if str(value).strip()]


def _parse_metxbio_bom_annotation(raw: str) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if not (line.startswith("<") and ">" in line):
            continue
        body = line[1 : line.index(">")]
        parts = [part.strip() for part in body.split(";")]
        if len(parts) < 2:
            continue
        locus = parts[0]
        reaction = parts[1]
        locus_parts = [token.strip() for token in locus.split(",") if token.strip()]
        if not locus_parts:
            continue
        if len(locus_parts) >= 2 and all(part.isdigit() for part in locus_parts[:2]):
            atoms = [int(locus_parts[0]) - 1, int(locus_parts[1]) - 1]
            if min(atoms) >= 0:
                parsed.append({"kind": "bond", "atoms": atoms, "reaction": reaction})
            continue
        if locus_parts[0].isdigit():
            atom_idx = int(locus_parts[0]) - 1
            if atom_idx >= 0:
                parsed.append({"kind": "atom", "atom": atom_idx, "reaction": reaction})
    return parsed


def _parse_attnsom_record(mol: Chem.Mol, *, source_name: str, source_file: Path, target_cyp: str) -> dict[str, Any] | None:
    canonical_smiles, mapping, canonical_mol = _canonical_identity_from_mol(mol)
    if canonical_smiles is None or mapping is None:
        return None
    base_mol = Chem.MolFromSmiles(canonical_smiles)
    atom_count = int(base_mol.GetNumAtoms()) if base_mol is not None else 0
    if atom_count <= 0:
        return None
    primary = _remap_indices(_parse_index_tokens(_safe_mol_prop(mol, "PRIMARY_SOM"), index_base=1), mapping)
    secondary = _remap_indices(_parse_index_tokens(_safe_mol_prop(mol, "SECONDARY_SOM"), index_base=1), mapping)
    tertiary = _remap_indices(_parse_index_tokens(_safe_mol_prop(mol, "TERTIARY_SOM"), index_base=1), mapping)
    all_atoms = sorted(set(primary + secondary + tertiary))
    if not all_atoms:
        return None
    citation = _safe_mol_prop(mol, "Citation")
    doi = _safe_mol_prop(mol, "DOI")
    name = _first_nonempty_prop(mol, ["ID", "_Name"]) or source_name
    record_id = _first_nonempty_prop(mol, ["ID"]) or name
    return {
        "molecule_name": str(name),
        "name": str(name),
        "canonical_smiles": str(canonical_smiles),
        "smiles": str(canonical_smiles),
        "inchi_key": _inchikey_from_mol(canonical_mol),
        "formula": _formula_from_mol(canonical_mol),
        "atom_count": int(atom_count),
        "target_cyp": str(target_cyp),
        "primary_cyp": str(target_cyp),
        "all_cyps": [str(target_cyp)],
        "source": str(source_name),
        "molecule_source": str(source_name),
        "site_source": str(source_name),
        "source_record_id": str(record_id),
        "source_file": str(source_file),
        "source_details": [str(source_name)],
        "citation": str(citation),
        "doi": str(doi),
        "primary_site_atoms": list(primary),
        "secondary_site_atoms": list(secondary),
        "tertiary_site_atoms": list(tertiary),
        "all_labeled_site_atoms": list(all_atoms),
        "site_atoms": list(all_atoms),
        "metabolism_sites": list(all_atoms),
        "som": [{"atom_idx": int(idx)} for idx in all_atoms],
        "site_count": int(len(all_atoms)),
        "is_multisite": bool(len(all_atoms) > 1),
        "label_regime": _infer_label_regime(
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
            all_atoms=all_atoms,
            broad_region_annotations=[],
        ),
        "broad_region_annotations": [],
        "full_xtb_status": "unknown",
        "confidence": "validated" if primary and not secondary and not tertiary else "medium",
        "citation_available": bool(str(citation).strip()),
        "doi_available": bool(str(doi).strip()),
        "site_provenance_preserved": True,
        "raw_property_keys": _safe_property_keys(mol),
        "parser_family": "attnsom_tiered",
    }


def _resolve_target_cyp(mol: Chem.Mol, config: dict[str, Any], target_cyp: str) -> str:
    fixed_cyp = str(config.get("fixed_cyp") or "").strip()
    if fixed_cyp:
        return _normalize_cyp(fixed_cyp)
    cyp_field = str(config.get("cyp_field") or "").strip()
    if cyp_field:
        return _normalize_cyp(_safe_mol_prop(mol, cyp_field))
    return str(target_cyp)


def _parse_generic_record(
    mol: Chem.Mol,
    *,
    source_name: str,
    source_file: Path,
    target_cyp: str,
    parser_family: str,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    resolved_cyp = _resolve_target_cyp(mol, config, target_cyp)
    if _normalize_cyp(resolved_cyp) != _normalize_cyp(target_cyp):
        return None
    canonical_smiles, mapping, canonical_mol = _canonical_identity_from_mol(mol)
    if canonical_smiles is None or mapping is None:
        return None
    atom_count = int(canonical_mol.GetNumAtoms()) if canonical_mol is not None else 0
    if atom_count <= 0:
        return None
    index_base = int(config.get("index_base", 0))
    id_fields = list(config.get("id_fields") or ["ID", "DRUGBANK_ID", "COMPOUND_ID", "_Name"])
    name_fields = list(config.get("name_fields") or ["_Name", "NAME", "ID"])
    citation_fields = list(config.get("citation_fields") or ["Citation", "CITATION"])
    doi_fields = list(config.get("doi_fields") or ["DOI", "Doi"])
    broad_region_fields = list(config.get("broad_region_fields") or [])
    source_detail_fields = list(config.get("source_detail_fields") or [])
    primary: list[int] = []
    secondary: list[int] = []
    tertiary: list[int] = []
    all_atoms: list[int] = []
    if parser_family == "tiered_atom_lists":
        primary = _remap_indices(_parse_index_tokens(_safe_mol_prop(mol, str(config.get("primary_field") or "")), index_base=index_base), mapping)
        secondary = _remap_indices(_parse_index_tokens(_safe_mol_prop(mol, str(config.get("secondary_field") or "")), index_base=index_base), mapping)
        tertiary = _remap_indices(_parse_index_tokens(_safe_mol_prop(mol, str(config.get("tertiary_field") or "")), index_base=index_base), mapping)
        all_atoms = sorted(set(primary + secondary + tertiary))
    elif parser_family == "flat_atom_list":
        site_fields = list(config.get("site_fields") or [])
        tokens: list[int] = []
        for field_name in site_fields:
            tokens.extend(_parse_index_tokens(_safe_mol_prop(mol, str(field_name)), index_base=index_base))
        all_atoms = _remap_indices(tokens, mapping)
    elif parser_family == "metxbio_bom":
        raw = _safe_mol_prop(mol, str(config.get("bom_field") or "BOM_3A4"))
        parsed = _parse_metxbio_bom_annotation(raw)
        atom_sites = _remap_indices(
            sorted({int(item["atom"]) for item in parsed if item["kind"] == "atom"}),
            mapping,
        )
        bond_sites = _remap_indices(
            sorted({int(atom) for item in parsed if item["kind"] == "bond" for atom in item["atoms"]}),
            mapping,
        )
        all_atoms = sorted(set(atom_sites + bond_sites))
        flag_comments_field = str(config.get("flag_comments_field") or "FLAG_COMMENTS")
        broad_regions = [str(_safe_mol_prop(mol, flag_comments_field))] if str(_safe_mol_prop(mol, flag_comments_field)).strip() else []
        primary = list(atom_sites)
        secondary = []
        tertiary = []
        if atom_sites:
            label_regime_default = "multi_exact" if len(atom_sites) > 1 else "single_exact"
        elif bond_sites:
            label_regime_default = "broad_region"
        else:
            label_regime_default = "unknown"
    else:
        raise ValueError(f"Unsupported parser_family={parser_family}")
    if parser_family != "metxbio_bom":
        broad_regions = [str(_safe_mol_prop(mol, field)) for field in broad_region_fields if str(_safe_mol_prop(mol, field)).strip()]
        label_regime_default = _infer_label_regime(
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
            all_atoms=all_atoms,
            broad_region_annotations=broad_regions,
        )
    if not all_atoms and not broad_regions:
        return None
    record_id = _first_nonempty_prop(mol, [str(v) for v in id_fields if str(v).strip()]) or source_name
    name = _first_nonempty_prop(mol, [str(v) for v in name_fields if str(v).strip()]) or record_id or source_name
    citation = _first_nonempty_prop(mol, [str(v) for v in citation_fields if str(v).strip()])
    doi = _first_nonempty_prop(mol, [str(v) for v in doi_fields if str(v).strip()])
    source_details = [str(source_name)]
    for field in source_detail_fields:
        value = _safe_mol_prop(mol, str(field))
        if str(value).strip():
            source_details.append(f"{field}={value}")
    source_details = _dedupe_sorted([value for value in source_details if str(value).strip()])
    label_regime_override = str(config.get("label_regime_override") or "").strip()
    label_regime = label_regime_override or label_regime_default
    return {
        "molecule_name": str(name),
        "name": str(name),
        "canonical_smiles": str(canonical_smiles),
        "smiles": str(canonical_smiles),
        "inchi_key": _inchikey_from_mol(canonical_mol),
        "formula": _formula_from_mol(canonical_mol),
        "atom_count": int(atom_count),
        "target_cyp": str(target_cyp),
        "primary_cyp": str(target_cyp),
        "all_cyps": [str(target_cyp)],
        "source": str(source_name),
        "molecule_source": str(source_name),
        "site_source": str(source_name),
        "source_record_id": str(record_id),
        "source_file": str(source_file),
        "source_details": source_details,
        "citation": str(citation),
        "doi": str(doi),
        "primary_site_atoms": list(primary),
        "secondary_site_atoms": list(secondary),
        "tertiary_site_atoms": list(tertiary),
        "all_labeled_site_atoms": list(all_atoms),
        "site_atoms": list(all_atoms),
        "metabolism_sites": list(all_atoms),
        "som": [{"atom_idx": int(idx)} for idx in all_atoms],
        "site_count": int(len(all_atoms)),
        "is_multisite": bool(len(all_atoms) > 1),
        "label_regime": str(label_regime),
        "broad_region_annotations": list(broad_regions),
        "full_xtb_status": "unknown",
        "confidence": str(config.get("confidence_override") or ("medium" if all_atoms else "low")),
        "citation_available": bool(str(citation).strip()),
        "doi_available": bool(str(doi).strip()),
        "site_provenance_preserved": True,
        "raw_property_keys": _safe_property_keys(mol),
        "parser_family": str(parser_family),
    }


def _scan_sdf_properties(path: Path, *, limit: int) -> dict[str, Any]:
    prop_counts = Counter()
    mol_count = 0
    valid_count = 0
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in supplier:
        mol_count += 1
        if mol is None:
            continue
        valid_count += 1
        for key in mol.GetPropsAsDict(includePrivate=False, includeComputed=False).keys():
            prop_counts[str(key)] += 1
        if mol_count >= int(limit):
            break
    return {
        "scanned_records": int(mol_count),
        "valid_records": int(valid_count),
        "property_counts": dict(sorted(prop_counts.items())),
    }


def _load_json_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Source config must be a JSON object: {path}")
    return payload


def _default_parser_family(source_name: str) -> str:
    if _normalize_token(source_name) == "attnsom":
        return "attnsom_tiered"
    raise ValueError(f"No default parser_family for source '{source_name}'. Provide --parser-family or --source-config-json.")


def _build_source_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    input_paths = [Path(value).expanduser() for value in list(args.input_sdf or [])]
    if not input_paths:
        raise ValueError("Provide at least one --input-sdf")
    source_names = list(args.source_name or [])
    parser_families = list(args.parser_family or [])
    config_paths = [Path(value).expanduser() for value in list(args.source_config_json or [])]
    if source_names and len(source_names) != len(input_paths):
        raise ValueError("--source-name must be repeated exactly once per --input-sdf")
    if parser_families and len(parser_families) != len(input_paths):
        raise ValueError("--parser-family must be repeated exactly once per --input-sdf")
    if config_paths and len(config_paths) != len(input_paths):
        raise ValueError("--source-config-json must be repeated exactly once per --input-sdf")
    specs = []
    for index, input_path in enumerate(input_paths):
        if not input_path.is_absolute():
            input_path = (ROOT / input_path).resolve()
        source_name = source_names[index] if index < len(source_names) else input_path.stem
        parser_family = parser_families[index] if index < len(parser_families) and str(parser_families[index]).strip() else _default_parser_family(source_name)
        config = _load_json_config(config_paths[index] if index < len(config_paths) else None)
        specs.append(
            {
                "input_sdf": input_path,
                "source_name": str(source_name),
                "parser_family": str(parser_family),
                "config": dict(config),
            }
        )
    return specs


def _hash_group_id(canonical_smiles: str, target_cyp: str) -> str:
    digest = hashlib.sha1(f"{canonical_smiles}::{target_cyp}".encode("utf-8")).hexdigest()[:16]
    return f"dup_{digest}"


def _signature_key(annotation: dict[str, Any]) -> str:
    return json.dumps(_annotation_signature(annotation), sort_keys=True)


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
    row = {
        "id": str(preferred.get("source_record_id") or preferred.get("molecule_name") or preferred.get("canonical_smiles") or duplicate_group_id),
        "molecule_name": str(preferred.get("molecule_name") or preferred.get("name") or ""),
        "name": str(preferred.get("name") or preferred.get("molecule_name") or ""),
        "canonical_smiles": str(preferred.get("canonical_smiles") or ""),
        "smiles": str(preferred.get("canonical_smiles") or ""),
        "inchi_key": str(preferred.get("inchi_key") or ""),
        "formula": str(preferred.get("formula") or ""),
        "atom_count": int(preferred.get("atom_count") or 0),
        "target_cyp": str(preferred.get("target_cyp") or "CYP3A4"),
        "primary_cyp": str(preferred.get("target_cyp") or "CYP3A4"),
        "all_cyps": [str(preferred.get("target_cyp") or "CYP3A4")],
        "site_atoms": list(union_sites if merge_policy_used != "conflict_preserved" else list(preferred.get("site_atoms") or [])),
        "metabolism_sites": list(union_sites if merge_policy_used != "conflict_preserved" else list(preferred.get("site_atoms") or [])),
        "som": [{"atom_idx": int(idx)} for idx in (union_sites if merge_policy_used != "conflict_preserved" else list(preferred.get("site_atoms") or []))],
        "site_count": int(len(union_sites if merge_policy_used != "conflict_preserved" else list(preferred.get("site_atoms") or []))),
        "is_multisite": bool(len(union_sites if merge_policy_used != "conflict_preserved" else list(preferred.get("site_atoms") or [])) > 1),
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


def _merge_duplicate_group(annotations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    duplicate_group_id = _hash_group_id(str(annotations[0].get("canonical_smiles") or ""), str(annotations[0].get("target_cyp") or "CYP3A4"))
    sources = sorted({str(annotation.get("source") or "") for annotation in annotations})
    signatures = {_signature_key(annotation) for annotation in annotations}
    conflict_flags = _group_conflict_flags(annotations)
    if len(annotations) == 1:
        policy = "single_source"
        rows = [_build_row_from_annotations(annotations, duplicate_group_id=duplicate_group_id, merge_policy_used=policy, conflict_flags=[])]
    elif len(signatures) == 1:
        policy = "exact_agreement"
        rows = [_build_row_from_annotations(annotations, duplicate_group_id=duplicate_group_id, merge_policy_used=policy, conflict_flags=[])]
    elif "contains_broad_region_label" not in conflict_flags and "site_set_disjoint" not in conflict_flags:
        policy = "partial_agreement_union"
        rows = [_build_row_from_annotations(annotations, duplicate_group_id=duplicate_group_id, merge_policy_used=policy, conflict_flags=conflict_flags)]
    else:
        policy = "conflict_preserved"
        rows = [
            _build_row_from_annotations([annotation], duplicate_group_id=duplicate_group_id, merge_policy_used=policy, conflict_flags=conflict_flags)
            for annotation in sorted(annotations, key=lambda row: (str(row.get("source") or ""), str(row.get("source_record_id") or "")))
        ]
    audit = {
        "duplicate_group_id": str(duplicate_group_id),
        "group_size": int(len(annotations)),
        "output_row_count": int(len(rows)),
        "merge_policy_used": str(policy),
        "sources": list(sources),
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


def main() -> None:
    args = _parse_args()
    source_specs = _build_source_specs(args)
    target_cyp = _normalize_cyp(args.target_cyp)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_annotations: list[dict[str, Any]] = []
    per_source_summary: list[dict[str, Any]] = []
    property_scans: dict[str, Any] = {}

    for spec in source_specs:
        input_sdf = Path(spec["input_sdf"])
        if not input_sdf.exists():
            raise FileNotFoundError(f"Input SDF not found: {input_sdf}")
        source_name = str(spec["source_name"])
        parser_family = str(spec["parser_family"])
        config = dict(spec.get("config") or {})
        property_scans[source_name] = _scan_sdf_properties(input_sdf, limit=int(args.property_scan_limit))
        parsed_count = 0
        retained_count = 0
        skipped_non_target_cyp = 0
        skipped_invalid = 0
        supplier = Chem.SDMolSupplier(str(input_sdf), removeHs=False)
        for mol in supplier:
            parsed_count += 1
            if mol is None:
                skipped_invalid += 1
                continue
            record = None
            if parser_family == "attnsom_tiered":
                record = _parse_attnsom_record(
                    mol,
                    source_name=source_name,
                    source_file=input_sdf,
                    target_cyp=target_cyp,
                )
            elif parser_family in {"flat_atom_list", "tiered_atom_lists", "metxbio_bom"}:
                record = _parse_generic_record(
                    mol,
                    source_name=source_name,
                    source_file=input_sdf,
                    target_cyp=target_cyp,
                    parser_family=parser_family,
                    config=config,
                )
            else:
                raise ValueError(f"Unsupported parser_family={parser_family}")
            if record is None:
                resolved_cyp = _resolve_target_cyp(mol, config, target_cyp) if parser_family != "attnsom_tiered" else target_cyp
                if _normalize_cyp(resolved_cyp) != _normalize_cyp(target_cyp):
                    skipped_non_target_cyp += 1
                else:
                    skipped_invalid += 1
                continue
            parsed_annotations.append(record)
            retained_count += 1
        per_source_summary.append(
            {
                "source_name": source_name,
                "source_file": str(input_sdf),
                "parser_family": parser_family,
                "parsed_count": int(parsed_count),
                "retained_cyp3a4_count": int(retained_count),
                "excluded_non_target_cyp_count": int(skipped_non_target_cyp),
                "excluded_invalid_or_missing_label_count": int(skipped_invalid),
            }
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for annotation in parsed_annotations:
        key = (str(annotation.get("canonical_smiles") or ""), str(annotation.get("target_cyp") or target_cyp))
        grouped[key].append(annotation)

    merged_rows: list[dict[str, Any]] = []
    duplicate_audit_rows: list[dict[str, Any]] = []
    conflict_audit_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    merge_policy_counts = Counter()

    for key in sorted(grouped):
        rows, audit = _merge_duplicate_group(grouped[key])
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
                        "site_source": str(annotation.get("site_source") or ""),
                        "source_record_id": str(annotation.get("source_record_id") or ""),
                        "source_file": str(annotation.get("source_file") or ""),
                        "annotation_label_regime": str(annotation.get("label_regime") or ""),
                        "primary_site_atoms": list(annotation.get("primary_site_atoms") or []),
                        "secondary_site_atoms": list(annotation.get("secondary_site_atoms") or []),
                        "tertiary_site_atoms": list(annotation.get("tertiary_site_atoms") or []),
                        "all_labeled_site_atoms": list(annotation.get("all_labeled_site_atoms") or []),
                        "citation": str(annotation.get("citation") or ""),
                        "doi": str(annotation.get("doi") or ""),
                    }
                )

    merged_rows = sorted(merged_rows, key=lambda row: (str(row.get("canonical_smiles") or ""), str(row.get("source") or ""), str(row.get("id") or "")))
    duplicate_group_count = sum(1 for values in grouped.values() if len(values) > 1)
    label_regime_counts = Counter(str(row.get("label_regime") or "unknown") for row in merged_rows)
    citation_count = sum(1 for row in merged_rows if bool(str(row.get("citation") or "").strip()))
    doi_count = sum(1 for row in merged_rows if bool(str(row.get("doi") or "").strip()))
    site_provenance_count = sum(1 for row in merged_rows if bool(row.get("per_source_site_annotations")))

    dataset_payload = {
        "n_drugs": int(len(merged_rows)),
        "n_site_labeled": int(sum(1 for row in merged_rows if list(row.get("all_labeled_site_atoms") or []))),
        "summary": {
            "total_input_sources": int(len(source_specs)),
            "total_parsed_rows": int(sum(item["parsed_count"] for item in per_source_summary)),
            "total_retained_cyp3a4_rows": int(len(parsed_annotations)),
            "total_final_merged_rows": int(len(merged_rows)),
            "label_regime_counts": dict(sorted(label_regime_counts.items())),
            "merge_policy_counts": dict(sorted(merge_policy_counts.items())),
        },
        "build_stats": {
            "target_cyp": str(target_cyp),
            "source_specs": [
                {
                    "source_name": spec["source_name"],
                    "source_file": str(spec["input_sdf"]),
                    "parser_family": spec["parser_family"],
                    "config_keys": sorted(list((spec.get("config") or {}).keys())),
                }
                for spec in source_specs
            ],
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
        "total_retained_cyp3a4_rows": int(len(parsed_annotations)),
        "total_final_merged_rows": int(len(merged_rows)),
        "dataset_output": str(output_dir / "cyp3a4_merged_dataset.json"),
        "duplicate_group_count": int(duplicate_group_count),
        "exact_agreement_merge_count": int(merge_policy_counts.get("exact_agreement", 0)),
        "partial_agreement_merge_count": int(merge_policy_counts.get("partial_agreement_union", 0)),
        "conflict_preserved_count": int(merge_policy_counts.get("conflict_preserved", 0)),
        "single_exact_count": int(label_regime_counts.get("single_exact", 0)),
        "multi_exact_count": int(label_regime_counts.get("multi_exact", 0)),
        "tiered_multisite_count": int(label_regime_counts.get("tiered_multisite", 0)),
        "broad_region_count": int(label_regime_counts.get("broad_region", 0)),
        "unknown_count": int(label_regime_counts.get("unknown", 0)),
        "citation_count": int(citation_count),
        "doi_count": int(doi_count),
        "site_provenance_preserved_count": int(site_provenance_count),
        "per_source_summary": per_source_summary,
        "merge_policy": {
            "exact_agreement": "Merged to one row when all source-specific label signatures matched exactly.",
            "partial_agreement_union": "Merged to one row with union site set and preserved per-source annotations when site sets differed but were not disjoint/broad-region conflicts.",
            "conflict_preserved": "Kept source-specific rows separate with shared duplicate_group_id when label regimes conflicted, broad-region labels were involved, or site sets were disjoint.",
        },
        "normalization_policy": {
            "sanitization": "RDKit SDF parsing + canonical SMILES generation via canonicalize_mol_with_mapping",
            "hydrogen_handling": "RemoveHs before canonicalization to stay consistent with existing repo builders",
            "salt_fragment_policy": "No extra salt stripping or charge neutralization beyond RDKit canonicalization",
            "atom_index_policy": "Site indices are remapped through canonicalize_mol_with_mapping when canonical atom order changes",
        },
        "outputs": {
            "merged_dataset_json": str(output_dir / "cyp3a4_merged_dataset.json"),
            "summary_json": str(output_dir / "cyp3a4_merged_dataset_summary.json"),
            "duplicate_audit_csv": str(output_dir / "cyp3a4_duplicate_audit.csv"),
            "conflict_audit_csv": str(output_dir / "cyp3a4_label_conflict_audit.csv"),
            "source_breakdown_csv": str(output_dir / "cyp3a4_source_breakdown.csv"),
            "provenance_audit_csv": str(output_dir / "cyp3a4_provenance_audit.csv"),
        },
    }

    _write_csv(output_dir / "cyp3a4_duplicate_audit.csv", duplicate_audit_rows)
    _write_csv(output_dir / "cyp3a4_label_conflict_audit.csv", conflict_audit_rows)
    _write_csv(output_dir / "cyp3a4_source_breakdown.csv", per_source_summary)
    _write_csv(output_dir / "cyp3a4_provenance_audit.csv", provenance_rows)
    (output_dir / "cyp3a4_merged_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.summary_only:
        (output_dir / "cyp3a4_merged_dataset.json").write_text(json.dumps(dataset_payload, indent=2), encoding="utf-8")

    print(
        "CYP3A4 merged dataset build complete | "
        f"sources={len(source_specs)} | "
        f"retained={len(parsed_annotations)} | "
        f"final_rows={len(merged_rows)} | "
        f"duplicate_groups={duplicate_group_count}",
        flush=True,
    )
    print(f"Summary JSON: {output_dir / 'cyp3a4_merged_dataset_summary.json'}", flush=True)
    print(f"Merged dataset: {output_dir / 'cyp3a4_merged_dataset.json'}", flush=True)


if __name__ == "__main__":
    main()
