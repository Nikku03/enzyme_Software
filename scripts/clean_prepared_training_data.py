#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rdkit import Chem


def load_drugs(path: Path) -> List[Dict[str, object]]:
    payload = json.loads(path.read_text())
    return list(payload.get("drugs", payload))


def save_dataset(path: Path, drugs: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_drugs": len(drugs),
        "n_site_labeled": sum(1 for drug in drugs if drug.get("site_atoms")),
        "summary": summary,
        "drugs": drugs,
    }
    path.write_text(json.dumps(payload, indent=2))


def normalize_reaction_name(name: str) -> str:
    value = str(name or "").strip().lower()
    aliases = {
        "epoxidation": "epoxidation",
        "epoxidation": "epoxidation",
        "epoxidation": "epoxidation",
        "s oxidation": "s-oxidation",
        "s_oxidation": "s-oxidation",
        "soxidation": "s-oxidation",
        "s-oxidation": "s-oxidation",
    }
    if value in aliases:
        return aliases[value]
    return value


def normalize_reactions(reactions: Iterable[str]) -> List[str]:
    normalized = {normalize_reaction_name(reaction) for reaction in reactions if str(reaction or "").strip()}
    return sorted(normalized)


def _eligible_carbon_neighbor(atom: Chem.Atom, neighbor: Chem.Atom, bond: Chem.Bond) -> bool:
    if neighbor.GetSymbol() != "C":
        return False
    if bond.GetBondTypeAsDouble() != 1.0:
        return False
    if neighbor.GetIsAromatic():
        return False
    if neighbor.GetTotalNumHs() <= 0:
        return False
    return True


def remap_noncarbon_sites(mol: Chem.Mol, site_atoms: List[int], reactions: List[str]) -> Tuple[List[int], Counter]:
    cleaned: set[int] = set()
    stats = Counter()
    reaction_set = set(reactions)
    for idx in site_atoms:
        if idx < 0 or idx >= mol.GetNumAtoms():
            stats["dropped_out_of_range"] += 1
            continue
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        if symbol == "C":
            cleaned.add(idx)
            continue
        if symbol == "S" and "s-oxidation" in reaction_set:
            cleaned.add(idx)
            stats["kept_s_oxidation_sulfur"] += 1
            continue
        if symbol in {"N", "O", "P", "S"}:
            mapped = []
            for neighbor in atom.GetNeighbors():
                bond = mol.GetBondBetweenAtoms(idx, neighbor.GetIdx())
                if bond is not None and _eligible_carbon_neighbor(atom, neighbor, bond):
                    mapped.append(neighbor.GetIdx())
            if mapped:
                cleaned.update(mapped)
                stats[f"remapped_{symbol.lower()}_to_adjacent_carbon"] += len(set(mapped))
                continue
        stats[f"dropped_{symbol.lower()}"] += 1
    return sorted(cleaned), stats


def clean_row(row: Dict[str, object]) -> Tuple[Dict[str, object], Counter]:
    cleaned = dict(row)
    cleaned["reactions"] = normalize_reactions(row.get("reactions") or [])
    site_atoms = [int(idx) for idx in row.get("site_atoms", [])]
    stats = Counter()
    if not site_atoms:
        cleaned["site_atoms"] = []
        cleaned["metabolism_sites"] = []
        cleaned["som"] = []
        return cleaned, stats
    mol = Chem.MolFromSmiles(str(row.get("smiles", "")))
    if mol is None:
        cleaned["site_atoms"] = []
        cleaned["metabolism_sites"] = []
        cleaned["som"] = []
        stats["invalid_smiles"] += 1
        return cleaned, stats
    cleaned_sites, remap_stats = remap_noncarbon_sites(mol, site_atoms, cleaned["reactions"])
    stats.update(remap_stats)
    cleaned["site_atoms"] = cleaned_sites
    cleaned["metabolism_sites"] = list(cleaned_sites)
    cleaned["som"] = []
    return cleaned, stats


def stratified_subset(drugs: List[Dict[str, object]], size: int, seed: int) -> List[Dict[str, object]]:
    if len(drugs) <= size:
        return list(drugs)
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for drug in drugs:
        buckets[(str(drug.get("source", "unknown")), str(drug.get("primary_cyp", "")))].append(drug)
    selected: List[Dict[str, object]] = []
    remainders: List[Dict[str, object]] = []
    target_ratio = float(size) / float(len(drugs))
    for bucket in buckets.values():
        pool = list(bucket)
        rng.shuffle(pool)
        take = min(len(pool), max(1, int(round(len(pool) * target_ratio))))
        selected.extend(pool[:take])
        remainders.extend(pool[take:])
    if len(selected) > size:
        rng.shuffle(selected)
        selected = selected[:size]
    elif len(selected) < size:
        rng.shuffle(remainders)
        selected.extend(remainders[: size - len(selected)])
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean prepared single-label training data for current models")
    parser.add_argument("--input-all", default="data/prepared_training/main5_all_models_conservative.json")
    parser.add_argument("--output-dir", default="data/prepared_training")
    parser.add_argument("--subset-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_all)
    output_dir = Path(args.output_dir)
    rows = load_drugs(input_path)

    by_smiles: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_smiles[str(row.get("smiles", ""))].append(row)

    ambiguous_rows: List[Dict[str, object]] = []
    cleaned_rows: List[Dict[str, object]] = []
    cleaning_stats = Counter()
    for smiles, group in by_smiles.items():
        cyps = {str(row.get("primary_cyp", "")) for row in group}
        if len(cyps) > 1:
            ambiguous_rows.extend(group)
            continue
        for row in group:
            cleaned, row_stats = clean_row(row)
            cleaned_rows.append(cleaned)
            cleaning_stats.update(row_stats)

    site_rows = [row for row in cleaned_rows if row.get("site_atoms")]
    subset_all = stratified_subset(cleaned_rows, size=args.subset_size, seed=args.seed)
    subset_site = [row for row in subset_all if row.get("site_atoms")]

    removed_summary = {
        "rows_removed": len(ambiguous_rows),
        "unique_smiles_removed": len({row.get("smiles") for row in ambiguous_rows}),
        "source_counts": dict(Counter(str(row.get("source", "unknown")) for row in ambiguous_rows)),
        "cyp_counts": dict(Counter(str(row.get("primary_cyp", "")) for row in ambiguous_rows)),
    }
    clean_summary = {
        "input_rows": len(rows),
        "output_rows": len(cleaned_rows),
        "output_site_rows": len(site_rows),
        "subset_rows": len(subset_all),
        "subset_site_rows": len(subset_site),
        "removed_multi_cyp": removed_summary,
        "cleaning_stats": dict(cleaning_stats),
        "source_counts": dict(Counter(str(row.get("source", "unknown")) for row in cleaned_rows)),
        "cyp_counts": dict(Counter(str(row.get("primary_cyp", "")) for row in cleaned_rows)),
    }

    save_dataset(output_dir / "main5_all_models_conservative_singlecyp_clean.json", cleaned_rows, clean_summary)
    save_dataset(output_dir / "main5_site_conservative_singlecyp_clean.json", site_rows, clean_summary)
    save_dataset(output_dir / f"main5_all_models_conservative_singlecyp_clean_{args.subset_size}.json", subset_all, clean_summary)
    save_dataset(output_dir / f"main5_site_conservative_singlecyp_clean_{args.subset_size}.json", subset_site, clean_summary)
    save_dataset(output_dir / "removed_multi_cyp_conflicts.json", ambiguous_rows, removed_summary)
    (output_dir / "clean_singlecyp_summary.json").write_text(json.dumps(clean_summary, indent=2))

    print(json.dumps(clean_summary, indent=2))


if __name__ == "__main__":
    main()
