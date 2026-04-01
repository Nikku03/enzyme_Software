"""
build_main7_dataset.py
======================
Builds main7: the expanded training dataset from all recoverable data on disk.

Sources merged (in priority order for deduplication):
  1. main6  — 278 molecules (current training set, gold standard)
  2. expanded_metx_test — 477 site-labeled molecules not in main6, model CYPs only
  3. removed_multi_cyp_conflicts — 365 unique SMILES excluded by the single-CYP
     filter; we take the best (highest-confidence, most-sites) row per SMILES
  4. ATTNSOM SDFs — 2003 molecules across 9 CYPs, parsed from
     data/ATTNSOM/cyp_dataset/*.sdf using PRIMARY_SOM atom indices
  5. AZ 120 compounds — CSV with exact SoM atom indices, CYP3A4 only

Deduplication: by canonical SMILES (RDKit) if available, else raw SMILES.
Filtering:
  - primary_cyp must be in MODEL_CYPS (CYP1A2, 2C9, 2C19, 2D6, 3A4)
  - must have at least one site_atom
  - site_atoms must be integers in [0, n_atoms)

Output:
  data/prepared_training/main7_site_conservative_singlecyp_clean_symm.json
  data/prepared_training/main7_build_report.json

Usage:
  python scripts/build_main7_dataset.py [--output PATH] [--no-attnsom] [--no-az]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_CYPS = {"CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"}
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PREPARED_DIR = DATA_DIR / "prepared_training"

# Confidence ordering (higher index = higher priority when deduplicating)
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2, "validated": 3, "": 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smiles_key(smiles: str) -> str:
    """Canonical dedup key: try RDKit, fall back to stripped SMILES."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        pass
    return smiles.strip()


def _conf_rank(drug: dict) -> int:
    return CONFIDENCE_RANK.get(str(drug.get("confidence", "")).lower(), 0)


def _valid_site_atoms(site_atoms, n_atoms: int) -> list[int]:
    """Return valid integer site atom indices within molecule bounds."""
    result = []
    for a in (site_atoms or []):
        try:
            idx = int(a)
            if 0 <= idx < n_atoms:
                result.append(idx)
        except (ValueError, TypeError):
            pass
    return sorted(set(result))


def _smiles_n_atoms(smiles: str) -> int:
    """Count heavy atoms via RDKit, or approximate via regex."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol.GetNumAtoms()
    except Exception:
        pass
    # Rough fallback: count uppercase letters (atoms) minus H
    return len(re.findall(r"[A-Z]", smiles))


def _normalize_cyp(cyp_str: str) -> Optional[str]:
    """Extract the first model-supported CYP from a possibly-messy string."""
    if not cyp_str:
        return None
    for cyp in MODEL_CYPS:
        if cyp in cyp_str:
            return cyp
    return None


def _make_id(source_tag: str, smiles: str, cyp: str) -> str:
    h = hashlib.md5(smiles.encode()).hexdigest()[:12]
    return f"{source_tag}:{h}:{cyp}"


def _base_record(drug: dict, source_tag: str = "") -> dict:
    """Return a normalised drug record in main6 schema."""
    smiles = drug.get("canonical_smiles") or drug.get("smiles", "")
    cyp = drug.get("primary_cyp", "")
    return {
        "id": drug.get("id") or _make_id(source_tag, smiles, cyp),
        "name": drug.get("name", ""),
        "smiles": smiles,
        "primary_cyp": cyp,
        "all_cyps": drug.get("all_cyps", [cyp] if cyp else []),
        "reactions": drug.get("reactions", []),
        "site_atoms": drug.get("site_atoms", []),
        "metabolism_sites": drug.get("metabolism_sites", drug.get("site_atoms", [])),
        "source": drug.get("source", source_tag),
        "site_source": drug.get("site_source", drug.get("source", source_tag)),
        "confidence": drug.get("confidence", "medium"),
        "full_xtb_status": drug.get("full_xtb_status", "unknown"),
        "som": drug.get("som", []),
        "source_details": drug.get("source_details", [drug.get("source", source_tag)]),
        "symmetry_expanded": drug.get("symmetry_expanded", False),
        "symmetry_expanded_added_atoms": drug.get("symmetry_expanded_added_atoms", []),
        "symmetry_expanded_groups": drug.get("symmetry_expanded_groups", []),
    }


# ---------------------------------------------------------------------------
# Source 1: main6
# ---------------------------------------------------------------------------

def load_main6() -> list[dict]:
    path = PREPARED_DIR / "main6_site_conservative_singlecyp_clean_symm.json"
    d = json.loads(path.read_text())
    drugs = d.get("drugs", [])
    print(f"  [main6] loaded {len(drugs)} molecules")
    return [_base_record(drug, "main6") for drug in drugs]


# ---------------------------------------------------------------------------
# Source 2: expanded_metx_test
# ---------------------------------------------------------------------------

def load_expanded_metx_test() -> list[dict]:
    path = DATA_DIR / "expanded_metx_test" / "expanded_site_labeled.json"
    d = json.loads(path.read_text())
    drugs = d.get("drugs", [])
    valid = []
    for drug in drugs:
        cyp = drug.get("primary_cyp", "")
        # Only exact CYP strings in MODEL_CYPS
        if cyp not in MODEL_CYPS:
            continue
        site = drug.get("site_atoms", [])
        if not site:
            continue
        smiles = drug.get("canonical_smiles") or drug.get("smiles", "")
        if not smiles:
            continue
        valid.append(_base_record(drug, "expanded_metx_test"))
    print(f"  [expanded_metx_test] {len(valid)}/{len(drugs)} usable (model CYPs, site-labeled)")
    return valid


# ---------------------------------------------------------------------------
# Source 3: removed_multi_cyp_conflicts  (take best row per unique SMILES)
# ---------------------------------------------------------------------------

def load_multi_cyp_removed() -> list[dict]:
    path = PREPARED_DIR / "removed_multi_cyp_conflicts.json"
    d = json.loads(path.read_text())
    drugs = d.get("drugs", [])

    # Group by SMILES, keep best row per SMILES
    by_smiles: dict[str, list[dict]] = defaultdict(list)
    for drug in drugs:
        smiles = drug.get("canonical_smiles") or drug.get("smiles", "")
        site = drug.get("site_atoms", [])
        cyp = drug.get("primary_cyp", "")
        if not smiles or not site or cyp not in MODEL_CYPS:
            continue
        by_smiles[smiles].append(drug)

    records = []
    for smiles, candidates in by_smiles.items():
        # Pick highest confidence, then most site atoms
        best = max(candidates, key=lambda x: (_conf_rank(x), len(x.get("site_atoms", []))))
        records.append(_base_record(best, "removed_multicyp"))

    print(f"  [removed_multi_cyp] {len(records)} unique SMILES recovered (best primary CYP row)")
    return records


# ---------------------------------------------------------------------------
# Source 4: ATTNSOM SDFs
# ---------------------------------------------------------------------------

def _parse_attnsom_sdf(sdf_path: Path, cyp: str) -> list[dict]:
    """Parse an ATTNSOM SDF file, extract PRIMARY_SOM + SECONDARY_SOM."""
    with open(sdf_path, "rb") as f:
        content = f.read().decode("latin-1")

    records = []
    for mol_block in content.split("$$$$"):
        mol_block = mol_block.strip()
        if not mol_block:
            continue

        lines = mol_block.split("\n")

        # Molecule name is the first line
        mol_name = lines[0].strip() if lines else ""

        # Extract properties
        props: dict[str, list[str]] = {}
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("> <") or line.startswith(">  <"):
                # Property name
                prop_name = re.sub(r"[<>]", "", line).strip()
                values = []
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(">"):
                    val = lines[i].strip()
                    if val:
                        values.append(val)
                    i += 1
                props.setdefault(prop_name, []).extend(values)
            else:
                i += 1

        # Extract SMILES from mol block via RDKit if available
        smiles = ""
        try:
            from rdkit import Chem
            # Reconstruct mol block up to M  END
            mol_lines = []
            for line in lines:
                mol_lines.append(line)
                if "M  END" in line:
                    break
            mol_block_only = "\n".join(mol_lines)
            mol = Chem.MolFromMolBlock(mol_block_only, removeHs=True)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                n_atoms = mol.GetNumAtoms()
            else:
                continue
        except Exception:
            continue

        if not smiles:
            continue

        # Parse PRIMARY_SOM (0-based atom indices, space- or comma-separated)
        raw_primary = " ".join(props.get("PRIMARY_SOM", []))
        primary_atoms = []
        for tok in re.split(r"[\s,]+", raw_primary):
            try:
                primary_atoms.append(int(tok) - 1)  # ATTNSOM is 1-based
            except ValueError:
                pass

        # Parse SECONDARY_SOM
        raw_secondary = " ".join(props.get("SECONDARY_SOM", []))
        secondary_atoms = []
        for tok in re.split(r"[\s,]+", raw_secondary):
            try:
                secondary_atoms.append(int(tok) - 1)
            except ValueError:
                pass

        # Validate atom indices
        site_atoms = _valid_site_atoms(primary_atoms, n_atoms)
        if not site_atoms:
            # Try secondary if primary missing
            site_atoms = _valid_site_atoms(secondary_atoms, n_atoms)
        if not site_atoms:
            continue

        mol_id = props.get("ID", [mol_name])[0] if props.get("ID") else mol_name
        record = {
            "id": _make_id(f"attnsom_{cyp}", smiles, cyp),
            "name": mol_id or mol_name,
            "smiles": smiles,
            "primary_cyp": cyp,
            "all_cyps": [cyp],
            "reactions": ["hydroxylation"],
            "site_atoms": site_atoms,
            "metabolism_sites": site_atoms,
            "source": "ATTNSOM",
            "site_source": "ATTNSOM",
            "confidence": "medium",
            "full_xtb_status": "unknown",
            "som": [{"atom_idx": a} for a in site_atoms],
            "source_details": ["ATTNSOM"],
            "symmetry_expanded": False,
            "symmetry_expanded_added_atoms": [],
            "symmetry_expanded_groups": [],
        }
        records.append(record)

    return records


def load_attnsom() -> list[dict]:
    attn_dir = DATA_DIR / "ATTNSOM" / "cyp_dataset"
    # Map SDF filename stem → CYP name
    sdf_to_cyp = {
        "1A2": "CYP1A2",
        "2C9": "CYP2C9",
        "2C19": "CYP2C19",
        "2D6": "CYP2D6",
        "3A4": "CYP3A4",
        # Below are not in MODEL_CYPS but parse anyway for future
        # "2A6": "CYP2A6",
        # "2B6": "CYP2B6",
        # "2C8": "CYP2C8",
        # "2E1": "CYP2E1",
    }
    all_records = []
    for stem, cyp in sdf_to_cyp.items():
        sdf_path = attn_dir / f"{stem}.sdf"
        if not sdf_path.exists():
            print(f"  [ATTNSOM] {sdf_path.name} not found — skipping")
            continue
        records = _parse_attnsom_sdf(sdf_path, cyp)
        print(f"  [ATTNSOM] {sdf_path.name}: {len(records)} molecules with PRIMARY_SOM")
        all_records.extend(records)
    print(f"  [ATTNSOM] total: {len(all_records)} molecules")
    return all_records


# ---------------------------------------------------------------------------
# Source 5: AZ 120 compounds
# ---------------------------------------------------------------------------

def load_az_compounds() -> list[dict]:
    az_path = DATA_DIR / "ATTNSOM" / "cyp_dataset" / "az" / "az_120_compounds.csv"
    if not az_path.exists():
        print("  [AZ] az_120_compounds.csv not found — skipping")
        return []

    records = []
    with open(az_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get("SMILES", "").strip()
            if not smiles:
                continue

            # Parse SoMs: "['34', '30', '18,19,20,21,22,23,24,25,26,27,28']"
            raw_soms = row.get("SoMs grouped (numbers provided are atom indices)", "")
            exact_flags = row.get("Exact SoM annotation (1) or extended SoM annotation (0) per group", "")

            # Parse as Python-like list strings
            som_groups = re.findall(r"'([^']+)'", raw_soms)
            exact_list = re.findall(r"'([^']+)'", exact_flags)

            # Collect only EXACT SoM atoms (flag == 1) as primary; fall back to all
            site_atoms = []
            all_atoms = []
            for i, group in enumerate(som_groups):
                atoms_in_group = []
                for tok in re.split(r"[\s,]+", group.strip()):
                    try:
                        atoms_in_group.append(int(tok))
                    except ValueError:
                        pass
                all_atoms.extend(atoms_in_group)
                is_exact = (i < len(exact_list) and exact_list[i].strip() == "1")
                if is_exact:
                    site_atoms.extend(atoms_in_group)

            if not site_atoms:
                site_atoms = all_atoms
            if not site_atoms:
                continue

            n_atoms = _smiles_n_atoms(smiles)
            site_atoms = _valid_site_atoms(site_atoms, n_atoms)
            if not site_atoms:
                continue

            # AZ compounds are CYP3A4 (the dataset is from an AZ 3A4 study)
            cyp = "CYP3A4"
            compound_id = row.get("Compound ID", "").strip()
            records.append({
                "id": _make_id("az120", smiles, cyp),
                "name": compound_id,
                "smiles": smiles,
                "primary_cyp": cyp,
                "all_cyps": [cyp],
                "reactions": ["hydroxylation"],
                "site_atoms": site_atoms,
                "metabolism_sites": site_atoms,
                "source": "AZ120",
                "site_source": "AZ120",
                "confidence": "high",  # AZ internal data, typically high quality
                "full_xtb_status": "unknown",
                "som": [{"atom_idx": a} for a in site_atoms],
                "source_details": ["AZ120"],
                "symmetry_expanded": False,
                "symmetry_expanded_added_atoms": [],
                "symmetry_expanded_groups": [],
            })

    print(f"  [AZ120] {len(records)}/{120} molecules with valid site atoms")
    return records


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(
    all_records: list[dict],
    source_priority: list[str],
) -> tuple[list[dict], dict]:
    """
    Deduplicate by canonical SMILES.
    When two records have the same SMILES:
      - prefer the one from the higher-priority source
      - break ties by confidence rank, then number of site atoms
    Returns (deduped_list, stats).
    """
    source_rank = {src: i for i, src in enumerate(source_priority)}

    def record_key(r: dict) -> tuple:
        src = r.get("source", "")
        # Find the tag that matches source_priority (sources can differ from tag)
        src_rank = max(
            (source_rank.get(tag, -1) for tag in [src] + r.get("source_details", [])),
            default=-1,
        )
        return (src_rank, _conf_rank(r), len(r.get("site_atoms", [])))

    by_key: dict[str, list[dict]] = defaultdict(list)
    skipped_no_smiles = 0
    for rec in all_records:
        smiles = rec.get("smiles", "").strip()
        if not smiles:
            skipped_no_smiles += 1
            continue
        key = _smiles_key(smiles)
        by_key[key].append(rec)

    deduped = []
    n_conflicts = 0
    for key, candidates in by_key.items():
        if len(candidates) > 1:
            n_conflicts += 1
        best = max(candidates, key=record_key)
        deduped.append(best)

    stats = {
        "total_input": len(all_records),
        "unique_smiles": len(deduped),
        "duplicates_resolved": n_conflicts,
        "skipped_no_smiles": skipped_no_smiles,
    }
    return deduped, stats


# ---------------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------------

def validate_records(records: list[dict]) -> tuple[list[dict], dict]:
    valid = []
    stats = {"removed_no_site": 0, "removed_bad_cyp": 0, "removed_no_smiles": 0, "kept": 0}
    for rec in records:
        if not rec.get("smiles"):
            stats["removed_no_smiles"] += 1
            continue
        if rec.get("primary_cyp") not in MODEL_CYPS:
            stats["removed_bad_cyp"] += 1
            continue
        if not rec.get("site_atoms"):
            stats["removed_no_site"] += 1
            continue
        # Re-validate site atom indices against actual atom count
        n = _smiles_n_atoms(rec["smiles"])
        site = _valid_site_atoms(rec["site_atoms"], n)
        if not site:
            stats["removed_no_site"] += 1
            continue
        rec["site_atoms"] = site
        rec["metabolism_sites"] = site
        valid.append(rec)
        stats["kept"] += 1
    return valid, stats


# ---------------------------------------------------------------------------
# Build and save
# ---------------------------------------------------------------------------

def build_main7(args) -> None:
    print("=" * 60)
    print("Building main7 dataset")
    print("=" * 60)

    # Source priority for dedup (highest = most preferred)
    source_priority = ["main6", "expanded_metx_test", "removed_multicyp", "ATTNSOM", "AZ120"]

    all_records: list[dict] = []
    source_counts: dict[str, int] = {}

    # 1. main6
    print("\nLoading main6...")
    recs = load_main6()
    all_records.extend(recs)
    source_counts["main6"] = len(recs)

    # 2. expanded_metx_test
    print("Loading expanded_metx_test...")
    recs = load_expanded_metx_test()
    all_records.extend(recs)
    source_counts["expanded_metx_test"] = len(recs)

    # 3. removed multi-CYP
    print("Loading removed_multi_cyp_conflicts...")
    recs = load_multi_cyp_removed()
    all_records.extend(recs)
    source_counts["removed_multicyp"] = len(recs)

    # 4. ATTNSOM
    if not args.no_attnsom:
        print("Loading ATTNSOM SDFs...")
        try:
            recs = load_attnsom()
            all_records.extend(recs)
            source_counts["ATTNSOM"] = len(recs)
        except Exception as e:
            print(f"  [ATTNSOM] ERROR: {e} — skipping")
            source_counts["ATTNSOM"] = 0
    else:
        print("Skipping ATTNSOM (--no-attnsom)")
        source_counts["ATTNSOM"] = 0

    # 5. AZ 120
    if not args.no_az:
        print("Loading AZ 120...")
        try:
            recs = load_az_compounds()
            all_records.extend(recs)
            source_counts["AZ120"] = len(recs)
        except Exception as e:
            print(f"  [AZ120] ERROR: {e} — skipping")
            source_counts["AZ120"] = 0
    else:
        print("Skipping AZ 120 (--no-az)")
        source_counts["AZ120"] = 0

    print(f"\nTotal before dedup: {len(all_records)}")
    print(f"Per-source: {source_counts}")

    # Deduplicate
    print("\nDeduplicating by canonical SMILES...")
    deduped, dedup_stats = deduplicate(all_records, source_priority)
    print(f"  After dedup: {dedup_stats['unique_smiles']} unique molecules")
    print(f"  Conflicts resolved: {dedup_stats['duplicates_resolved']}")

    # Validate
    print("\nValidating records...")
    valid, val_stats = validate_records(deduped)
    print(f"  Kept: {val_stats['kept']}")
    print(f"  Removed (no site): {val_stats['removed_no_site']}")
    print(f"  Removed (bad CYP): {val_stats['removed_bad_cyp']}")
    print(f"  Removed (no SMILES): {val_stats['removed_no_smiles']}")

    # CYP distribution
    cyp_counts: dict[str, int] = defaultdict(int)
    final_source_counts: dict[str, int] = defaultdict(int)
    for rec in valid:
        cyp_counts[rec.get("primary_cyp", "?")] += 1
        final_source_counts[rec.get("source", "?")] += 1

    print(f"\nFinal dataset: {len(valid)} molecules")
    print(f"CYP distribution: {dict(sorted(cyp_counts.items(), key=lambda x:-x[1]))}")
    print(f"Source distribution: {dict(sorted(final_source_counts.items(), key=lambda x:-x[1]))}")
    print(f"Increase over main6: +{len(valid) - 278} molecules ({len(valid)/278:.1f}x)")

    # Build output
    output = {
        "n_drugs": len(valid),
        "n_site_labeled": len(valid),
        "summary": {
            "version": "main7",
            "description": "Expanded training set: main6 + expanded_metx_test + "
                           "recovered multi-CYP rows + ATTNSOM SDFs + AZ120",
            "cyp_counts": dict(cyp_counts),
            "source_counts": dict(final_source_counts),
            "model_cyps": sorted(MODEL_CYPS),
        },
        "build_stats": {
            "source_input_counts": source_counts,
            "dedup": dedup_stats,
            "validation": val_stats,
        },
        "drugs": valid,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    # Report
    report = {
        "output": str(out_path),
        "n_drugs": len(valid),
        "n_vs_main6": len(valid) - 278,
        "multiplier": round(len(valid) / 278, 2),
        "cyp_counts": dict(cyp_counts),
        "source_counts": dict(final_source_counts),
        "source_input_counts": source_counts,
        "dedup_stats": dedup_stats,
        "validation_stats": val_stats,
    }
    report_path = PREPARED_DIR / "main7_build_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    print("=" * 60)
    print("Done. Next step: update HYBRID_COLAB_DATASET in colab_train_hybrid_lnn.ipynb")
    print("  to: data/prepared_training/main7_site_conservative_singlecyp_clean_symm.json")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build main7 expanded training dataset")
    parser.add_argument(
        "--output",
        default=str(PREPARED_DIR / "main7_site_conservative_singlecyp_clean_symm.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--no-attnsom",
        action="store_true",
        help="Skip ATTNSOM SDF parsing (requires RDKit)",
    )
    parser.add_argument(
        "--no-az",
        action="store_true",
        help="Skip AZ 120 compounds",
    )
    args = parser.parse_args()
    build_main7(args)


if __name__ == "__main__":
    main()
