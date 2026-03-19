#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
SUPPL = ROOT / "suppl_info"


def load_xenosite_scores(path: Path) -> Dict[int, Dict[int, float]]:
    scores: Dict[int, Dict[int, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            raw_id = str(row.get("ID", "")).strip()
            raw_score = str(row.get("OUT1", "")).strip()
            if not raw_id or not raw_score:
                continue
            parts = raw_id.split(".")
            if len(parts) != 3:
                continue
            try:
                mol_idx = int(parts[0])
                atom_idx_1based = int(parts[2])
                score = float(raw_score)
            except ValueError:
                continue
            scores[mol_idx][atom_idx_1based - 1] = score
    return dict(scores)


def load_sdf_records(path: Path) -> List[Chem.Mol]:
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    return [mol for mol in suppl if mol is not None]


def canonical_smiles_without_atom_maps(mol: Chem.Mol) -> str:
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)


def iter_structure_files(directory: Path) -> Iterable[Path]:
    numeric = re.compile(r"^\d+(?:\.sdf)?\.txt$")
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if path.is_file() and numeric.match(path.name):
            yield path


def mol_to_entry(mol: Chem.Mol, *, source: str, source_path: str, mol_index: int, scores: Dict[int, float], task: str) -> Dict[str, object]:
    smiles = canonical_smiles_without_atom_maps(mol)
    dense_scores = [None] * mol.GetNumAtoms()
    for atom_idx, score in scores.items():
        if 0 <= atom_idx < len(dense_scores):
            dense_scores[atom_idx] = score
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return {
        "mol_index": mol_index,
        "source": source,
        "source_path": source_path,
        "task": task,
        "name": mol.GetProp("_Name") if mol.HasProp("_Name") else "",
        "smiles": smiles,
        "num_atoms": mol.GetNumAtoms(),
        "xenosite_score_pairs": [{"atom_index": idx, "score": score} for idx, score in ranked],
        "xenosite_dense_scores": dense_scores,
        "top_atoms": [idx for idx, _ in ranked[:5]],
    }


def extract_external_dataset(dataset_name: str, sdf_path: Path, score_path: Path) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    molecules = load_sdf_records(sdf_path)
    scores = load_xenosite_scores(score_path)
    rows: List[Dict[str, object]] = []
    covered = 0
    for idx, mol in enumerate(molecules, start=1):
        atom_scores = scores.get(idx, {})
        if atom_scores:
            covered += 1
        rows.append(
            mol_to_entry(
                mol,
                source=dataset_name,
                source_path=str(score_path.relative_to(ROOT)),
                mol_index=idx,
                scores=atom_scores,
                task="ugt_site_pseudolabel",
            )
        )
    summary = {
        "molecules": len(molecules),
        "covered_by_scores": covered,
        "score_rows": len(scores),
    }
    return rows, summary


def extract_difficult_set(dataset_name: str, structures_dir: Path, score_path: Path) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    scores = load_xenosite_scores(score_path)
    rows: List[Dict[str, object]] = []
    structure_count = 0
    covered = 0
    for structure_file in iter_structure_files(structures_dir):
        mol_idx = int(structure_file.name.split(".")[0])
        mols = load_sdf_records(structure_file)
        if not mols:
            continue
        structure_count += 1
        atom_scores = scores.get(mol_idx, {})
        if atom_scores:
            covered += 1
        rows.append(
            mol_to_entry(
                mols[0],
                source=dataset_name,
                source_path=str(structure_file.relative_to(ROOT)),
                mol_index=mol_idx,
                scores=atom_scores,
                task="ugt_site_pseudolabel",
            )
        )
    summary = {
        "structures": structure_count,
        "covered_by_scores": covered,
        "score_rows": len(scores),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/xenosite_suppl")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    manifest: Dict[str, object] = {"datasets": []}

    peng_rows, peng_summary = extract_external_dataset(
        "xenosite_peng_external",
        SUPPL / "Peng_external_dataset" / "Peng_external_dataset.sdf",
        SUPPL / "Peng_external_dataset" / "XenoSite_prediction_on_Peng_external_dataset.txt",
    )
    datasets.append(("peng_external_pseudolabels.json", peng_rows, peng_summary))

    rudik_rows, rudik_summary = extract_external_dataset(
        "xenosite_rudik_external",
        SUPPL / "Rudik_external_dataset" / "Rudik_external_dataset.sdf",
        SUPPL / "Rudik_external_dataset" / "XenoSite_prediction_on_external_dataset.txt",
    )
    datasets.append(("rudik_external_pseudolabels.json", rudik_rows, rudik_summary))

    difficult35_rows, difficult35_summary = extract_difficult_set(
        "xenosite_35_difficult",
        SUPPL / "35_difficult_molecules" / "SOMP_predictions",
        SUPPL / "35_difficult_molecules" / "XenoSite_predictions.txt",
    )
    datasets.append(("difficult35_pseudolabels.json", difficult35_rows, difficult35_summary))

    difficult49_rows, difficult49_summary = extract_difficult_set(
        "xenosite_49_difficult",
        SUPPL / "49_difficult_molecules" / "SOMP_predictions",
        SUPPL / "49_difficult_molecules" / "XenoSite_predictions.txt",
    )
    datasets.append(("difficult49_pseudolabels.json", difficult49_rows, difficult49_summary))

    for filename, rows, summary in datasets:
        payload = {
            "task": "ugt_site_pseudolabel",
            "count": len(rows),
            "entries": rows,
            "summary": summary,
        }
        (output_dir / filename).write_text(json.dumps(payload, indent=2))
        manifest["datasets"].append({"file": filename, **summary})

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
