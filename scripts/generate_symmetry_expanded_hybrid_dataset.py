from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rdkit import Chem


def _load_payload(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "drugs" in payload:
        return payload, list(payload.get("drugs", []))
    if isinstance(payload, list):
        return None, list(payload)
    raise ValueError(f"Unsupported dataset payload in {path}")


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
        values: list[int] = []
        for item in row.get("som", []):
            atom_idx = item.get("atom_idx", item) if isinstance(item, dict) else item
            if isinstance(atom_idx, int):
                values.append(int(atom_idx))
        return sorted(set(values))
    for key in ("site_atoms", "site_atom_indices", "metabolism_sites"):
        if row.get(key):
            return sorted(set(int(v) for v in row.get(key, []) if isinstance(v, int)))
    return []


def _expanded_site_atoms(row: dict[str, Any]) -> tuple[list[int], list[int], list[list[int]]]:
    original = _site_atoms(row)
    smiles = str(row.get("smiles", ""))
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or not original:
        return original, [], []
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    rank_to_atoms: dict[int, list[int]] = {}
    for idx, rank in enumerate(ranks):
        rank_to_atoms.setdefault(int(rank), []).append(idx)
    expanded = set(original)
    added_groups: list[list[int]] = []
    for atom_idx in original:
        if atom_idx < 0 or atom_idx >= len(ranks):
            continue
        group = sorted(rank_to_atoms.get(int(ranks[atom_idx]), []))
        new_atoms = [idx for idx in group if idx not in expanded]
        if new_atoms:
            added_groups.append(group)
            expanded.update(new_atoms)
    expanded_list = sorted(expanded)
    added = sorted(idx for idx in expanded_list if idx not in set(original))
    return expanded_list, added, added_groups


def _rewrite_row(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    original_sites = _site_atoms(row)
    expanded_sites, added_sites, added_groups = _expanded_site_atoms(row)
    updated["original_site_atoms"] = list(original_sites)
    updated["symmetry_allowed_site_atoms"] = list(expanded_sites)
    updated["symmetry_expanded_added_atoms"] = list(added_sites)
    updated["symmetry_expanded_groups"] = [list(group) for group in added_groups]
    updated["label_expansion_strategy"] = "rdkit_symmetry_class"
    if not expanded_sites or expanded_sites == original_sites:
        updated["symmetry_expanded"] = False
        return updated

    if row.get("som"):
        expanded_som = []
        seen = set()
        original_map = {}
        for item in row.get("som", []):
            atom_idx = item.get("atom_idx", item) if isinstance(item, dict) else item
            if isinstance(atom_idx, int):
                original_map[int(atom_idx)] = item
        for atom_idx in expanded_sites:
            template = original_map.get(atom_idx)
            if isinstance(template, dict):
                expanded_som.append(dict(template))
                seen.add(atom_idx)
            elif atom_idx not in seen:
                expanded_som.append({"atom_idx": int(atom_idx), "symmetry_expanded": True})
                seen.add(atom_idx)
        updated["original_som"] = row.get("som")
        updated["symmetry_allowed_som"] = expanded_som

    updated["symmetry_expanded"] = True
    return updated


def generate_dataset(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    expanded_rows = 0
    added_atoms_total = 0
    examples: list[dict[str, Any]] = []
    for row in rows:
        new_row = _rewrite_row(row)
        rewritten.append(new_row)
        added_atoms = list(new_row.get("symmetry_expanded_added_atoms", []))
        if added_atoms:
            expanded_rows += 1
            added_atoms_total += len(added_atoms)
            if len(examples) < 20:
                examples.append(
                    {
                        "id": str(new_row.get("id", "")),
                        "name": str(new_row.get("name", "")),
                        "source": str(new_row.get("source", "")),
                        "canonical_smiles": _canonical_smiles(new_row.get("smiles", "")),
                        "original_site_atoms": list(new_row.get("original_site_atoms", [])),
                        "expanded_site_atoms": list(new_row.get("site_atoms", [])),
                        "added_atoms": added_atoms,
                        "groups": list(new_row.get("symmetry_expanded_groups", [])),
                    }
                )
    summary = {
        "total_rows": len(rewritten),
        "expanded_rows": expanded_rows,
        "expanded_fraction": (expanded_rows / float(len(rewritten))) if rewritten else 0.0,
        "added_atoms_total": added_atoms_total,
        "examples": examples,
    }
    return rewritten, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate symmetry-expanded hybrid training dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-summary", default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dataset = Path(args.output_dataset)
    output_summary = Path(args.output_summary) if args.output_summary else None

    payload, rows = _load_payload(dataset_path)
    rewritten, summary = generate_dataset(rows)

    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    if payload is None:
        output_dataset.write_text(json.dumps(rewritten, indent=2), encoding="utf-8")
    else:
        out_payload = dict(payload)
        out_payload["drugs"] = rewritten
        out_payload["symmetry_expansion_summary"] = summary
        output_dataset.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")

    print(f"Wrote dataset: {output_dataset}")
    print(f"Expanded rows: {summary['expanded_rows']}/{summary['total_rows']}")
    print(f"Added atoms: {summary['added_atoms_total']}")
    if output_summary is not None:
        output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary: {output_summary}")


if __name__ == "__main__":
    main()
