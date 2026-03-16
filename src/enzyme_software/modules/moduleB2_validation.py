from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

from enzyme_software.calibration.drug_metabolism_db import DRUG_DATABASE
from enzyme_software.modules.moduleB2_drug_metabolism_predictor import predict_drug_metabolism


def _ground_truth_atoms(mol: Any, ground_truth: Dict[str, Any]) -> Set[int]:
    if Chem is None or mol is None:
        return set()
    smarts = str((ground_truth or {}).get("site_smarts") or "").strip()
    if not smarts:
        return set()
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return set()
    atoms: Set[int] = set()
    for match in mol.GetSubstructMatches(patt, uniquify=False):
        for idx in match:
            atoms.add(int(idx))
    return atoms


def _hit_at_k(ranked_sites: List[Dict[str, Any]], truth_atoms: Set[int], k: int) -> bool:
    k_eff = max(1, int(k))
    for site in ranked_sites[:k_eff]:
        anchors = set(int(v) for v in (site.get("anchor_atom_indices") or []))
        if anchors & truth_atoms:
            return True
    return False


def run_drug_metabolism_validation(
    drug_db: Optional[Dict[str, Dict[str, Any]]] = None,
    topk_list: Iterable[int] = (1, 3, 5),
) -> Dict[str, Any]:
    """Run Module B2 predictor against the curated drug metabolism DB."""
    use_db = dict(drug_db or DRUG_DATABASE)
    ks = sorted({max(1, int(k)) for k in topk_list})
    max_k = max(ks) if ks else 5

    per_drug: List[Dict[str, Any]] = []
    error_buckets: Dict[str, int] = {
        "site_enumeration_missing": 0,
        "route_mismatch": 0,
        "ranking_miss": 0,
        "ground_truth_ambiguous": 0,
    }
    hit_counts = {k: 0 for k in ks}
    evaluable = 0

    for key, drug in use_db.items():
        smiles = str(drug.get("smiles") or "")
        isoform = str(drug.get("primary_isoform") or drug.get("primary_cyp") or "")
        pred = predict_drug_metabolism(smiles=smiles, isoform_hint=isoform, topk=max_k)
        ranked = list(pred.get("ranked_sites") or [])

        row: Dict[str, Any] = {
            "drug_key": key,
            "drug_name": drug.get("name"),
            "primary_isoform": isoform,
            "ranked_count": len(ranked),
            "top_site": ranked[0] if ranked else None,
            "hits": {},
            "errors": [],
        }

        if not ranked:
            row["errors"].append("site_enumeration_missing")
            error_buckets["site_enumeration_missing"] += 1
            per_drug.append(row)
            continue

        if Chem is None:
            row["errors"].append("ground_truth_ambiguous")
            error_buckets["ground_truth_ambiguous"] += 1
            per_drug.append(row)
            continue

        mol = Chem.MolFromSmiles(smiles)
        truth_atoms = _ground_truth_atoms(mol, dict(drug.get("ground_truth") or {})) if mol is not None else set()
        if not truth_atoms:
            row["errors"].append("ground_truth_ambiguous")
            error_buckets["ground_truth_ambiguous"] += 1
            per_drug.append(row)
            continue

        evaluable += 1
        for k in ks:
            hit = _hit_at_k(ranked, truth_atoms, k)
            row["hits"][f"top{k}"] = hit
            if hit:
                hit_counts[k] += 1

        if not row["hits"].get(f"top{max_k}", False):
            row["errors"].append("ranking_miss")
            error_buckets["ranking_miss"] += 1

        top_family = str((ranked[0] or {}).get("predicted_family") or "").lower()
        top_route = str((ranked[0] or {}).get("predicted_route") or "").lower()
        if isoform.upper().startswith("CYP") and ("cytochrome" not in top_family and "p450" not in top_route):
            row["errors"].append("route_mismatch")
            error_buckets["route_mismatch"] += 1

        per_drug.append(row)

    metrics = {
        f"top{k}_acc": (float(hit_counts[k]) / float(evaluable) if evaluable > 0 else 0.0)
        for k in ks
    }

    return {
        "total_drugs": len(use_db),
        "evaluable": evaluable,
        "topk": ks,
        "metrics": metrics,
        "hit_counts": hit_counts,
        "error_buckets": error_buckets,
        "per_drug_results": per_drug,
    }
