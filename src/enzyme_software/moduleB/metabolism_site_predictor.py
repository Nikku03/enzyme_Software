from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

from enzyme_software.modules.module_minus1_reactivity_hub import run_module_minus1_reactivity_hub
from enzyme_software.modules.moduleB2_drug_metabolism_predictor import (
    predict_drug_metabolism as predict_cyp_like,
)

try:
    from enzyme_software.calibration.layer3_xtb import (
        compute_substrate_bde_with_safeguard_for_mol,
    )
except Exception:  # pragma: no cover
    compute_substrate_bde_with_safeguard_for_mol = None


REACTION_MAP: Dict[str, str] = {
    "ch__benzylic": "benzylic_hydroxylation",
    "ch__allylic": "allylic_hydroxylation",
    "ch__alpha_hetero": "o_or_n_demethylation",
    "ch__primary": "alkyl_hydroxylation",
    "ch__secondary": "alkyl_hydroxylation",
    "ch__tertiary": "alkyl_hydroxylation",
    "ch__aliphatic": "alkyl_hydroxylation",
    "ch__aryl": "aromatic_hydroxylation",
    "ch__vinyl": "alkenyl_hydroxylation",
    "nh__amine": "n_hydroxylation",
    "nh__amide": "n_hydroxylation",
    "oh__alcohol": "o_hydroxylation",
    "oh__phenol": "o_hydroxylation",
    "o_demethyl": "o_demethylation",
    "o_dealkyl": "o_dealkylation",
    "n_demethyl": "n_demethylation",
    "n_dealkyl": "n_dealkylation",
}

_DEFAULT_BDE_BY_CLASS: Dict[str, float] = {
    "ch__aliphatic": 410.0,
    "ch__primary": 423.0,
    "ch__secondary": 412.5,
    "ch__tertiary": 403.8,
    "ch__benzylic": 375.5,
    "ch__allylic": 371.5,
    "ch__alpha_hetero": 385.0,
    "ch__aryl": 472.2,
    "nh__amine": 386.0,
    "nh__amide": 440.0,
    "oh__alcohol": 435.7,
    "oh__phenol": 362.8,
    # Dealkylation proxies (alpha C-H context near heteroatom)
    "o_demethyl": 368.0,
    "o_dealkyl": 380.0,
    "n_demethyl": 390.0,
    "n_dealkyl": 398.0,
}

_XH_TARGETS: Tuple[str, ...] = ("C-H", "N-H", "O-H")


def _json_safe(obj: Any) -> Any:
    """Convert common non-JSON-safe structures into JSON-safe primitives."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    return str(obj)


def _stable_site_sort_key(site: Dict[str, Any]) -> Tuple[Any, ...]:
    """Deterministic sorting with explicit tie-breakers."""
    score = site.get("score_norm", site.get("score"))
    bde = site.get("bde")
    reaction = str(site.get("reaction_class") or "")
    bond_id = int(site.get("bond_id") or 10**9)
    atom_indices = site.get("atom_indices")
    if isinstance(atom_indices, tuple):
        atom_tuple = atom_indices
    elif isinstance(atom_indices, list):
        atom_tuple = tuple(atom_indices)
    else:
        atom_tuple = (10**9, 10**9)
    score_key = float(score) if isinstance(score, (int, float)) else 10**9
    bde_key = float(bde) if isinstance(bde, (int, float)) else 10**9
    return (score_key, bde_key, reaction, atom_tuple, bond_id)


def _reaction_class_from_bond_class(bond_class: str) -> str:
    cls = str(bond_class or "").lower()
    return REACTION_MAP.get(cls, "oxidation")


def _default_bde_for_class(bond_class: str, default: float = 410.0) -> float:
    cls = str(bond_class or "").lower()
    return float(_DEFAULT_BDE_BY_CLASS.get(cls, default))


def _first_h_neighbor_idx(mol_h: Any, heavy_idx: int) -> Optional[int]:
    if mol_h is None:
        return None
    if heavy_idx < 0 or heavy_idx >= mol_h.GetNumAtoms():
        return None
    atom = mol_h.GetAtomWithIdx(int(heavy_idx))
    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() == "H":
            return int(nbr.GetIdx())
    return None


def _normalize_atom_indices(
    mol_h: Any,
    candidate: Dict[str, Any],
    target_bond: str,
) -> Tuple[int, int]:
    atom_indices = candidate.get("atom_indices")
    if (
        isinstance(atom_indices, list)
        and len(atom_indices) == 2
        and all(isinstance(v, int) for v in atom_indices)
    ):
        return int(atom_indices[0]), int(atom_indices[1])

    heavy = candidate.get("heavy_atom_index")
    if isinstance(heavy, int) and str(target_bond).upper() in {"C-H", "N-H", "O-H"}:
        h_idx = _first_h_neighbor_idx(mol_h, int(heavy))
        if h_idx is not None:
            return int(heavy), int(h_idx)
    return -1, -1


def _collect_xh_sites(smiles: str, mol_h: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, str]] = set()
    for target in _XH_TARGETS:
        try:
            result = run_module_minus1_reactivity_hub(
                smiles=smiles,
                target_bond=target,
                requested_output=None,
                constraints={},
            )
        except Exception:
            result = {"resolved_target": {"candidate_bonds": []}, "cpt_scores": {}}
        resolved = (result or {}).get("resolved_target") or {}
        candidates = list(resolved.get("candidate_bonds") or [])
        for cand in candidates:
            a0, a1 = _normalize_atom_indices(mol_h, cand, target_bond=target)
            bond_class = str(cand.get("bond_class") or "").lower().strip()
            if not bond_class:
                if target == "C-H":
                    bond_class = "ch__aliphatic"
                elif target == "N-H":
                    bond_class = "nh__amine"
                else:
                    bond_class = "oh__alcohol"
            bde = cand.get("bde_kj_mol")
            if not isinstance(bde, (int, float)):
                cpt_bde = ((result or {}).get("cpt_scores") or {}).get("bde") or {}
                bde = cpt_bde.get("corrected_kj_mol")
            if not isinstance(bde, (int, float)):
                bde = _default_bde_for_class(bond_class)

            key = (int(a0), int(a1), bond_class)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "atom_indices": (int(a0), int(a1)),
                    "bond_class": bond_class,
                    "bde": float(bde),
                    "reaction_class": _reaction_class_from_bond_class(bond_class),
                    "source": "module_minus1",
                }
            )
    if not out:
        out.extend(_fallback_xh_sites(mol_h))
    return out


def _is_allylic(atom: Any) -> bool:
    if atom.GetIsAromatic():
        return False
    for nbr in atom.GetNeighbors():
        for bond in nbr.GetBonds():
            if float(bond.GetBondTypeAsDouble()) == 2.0:
                other = bond.GetOtherAtom(nbr)
                if other.GetIdx() != atom.GetIdx() and other.GetSymbol() == "C":
                    return True
    return False


def _ch_fallback_class(atom: Any) -> str:
    if atom.GetIsAromatic():
        return "ch__aryl"
    if any(n.GetIsAromatic() for n in atom.GetNeighbors()):
        return "ch__benzylic"
    if _is_allylic(atom):
        return "ch__allylic"
    if any(n.GetSymbol() in {"O", "N", "S"} for n in atom.GetNeighbors()):
        return "ch__alpha_hetero"
    carbon_neighbors = sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "C")
    if carbon_neighbors >= 3:
        return "ch__tertiary"
    if carbon_neighbors == 2:
        return "ch__secondary"
    return "ch__primary"


def _fallback_xh_sites(mol_h: Any) -> List[Dict[str, Any]]:
    fallback: List[Dict[str, Any]] = []
    for atom in mol_h.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in {"C", "N", "O"}:
            continue
        h_idx = _first_h_neighbor_idx(mol_h, int(atom.GetIdx()))
        if h_idx is None:
            continue
        if symbol == "C":
            cls = _ch_fallback_class(atom)
        elif symbol == "N":
            cls = "nh__amine"
        else:
            cls = "oh__alcohol"
        fallback.append(
            {
                "atom_indices": (int(atom.GetIdx()), int(h_idx)),
                "bond_class": cls,
                "bde": _default_bde_for_class(cls),
                "reaction_class": _reaction_class_from_bond_class(cls),
                "source": "fallback_heuristic",
            }
        )
    return fallback


def _dealkyl_class(hetero_symbol: str, carbon_atom: Any) -> str:
    total_h = int(carbon_atom.GetTotalNumHs())
    if hetero_symbol == "O" and total_h >= 3:
        return "o_demethyl"
    if hetero_symbol == "N" and total_h >= 3:
        return "n_demethyl"
    if hetero_symbol == "O":
        return "o_dealkyl"
    return "n_dealkyl"


def _collect_dealkylation_sites(mol: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int, str]] = set()
    for bond in mol.GetBonds():
        if int(bond.GetBondTypeAsDouble()) != 1:
            continue
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        syms = {a.GetSymbol(), b.GetSymbol()}
        if syms not in ({"C", "O"}, {"C", "N"}):
            continue

        carbon = a if a.GetSymbol() == "C" else b
        hetero = b if carbon.GetIdx() == a.GetIdx() else a
        if carbon.GetIsAromatic():
            continue
        if int(carbon.GetTotalNumHs()) < 1:
            continue

        cls = _dealkyl_class(hetero.GetSymbol(), carbon)
        bde = _default_bde_for_class(cls)
        c_idx = int(carbon.GetIdx())
        h_idx = int(hetero.GetIdx())
        key = (c_idx, h_idx, cls)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "atom_indices": (c_idx, h_idx),
                "bond_class": cls,
                "bde": float(bde),
                "reaction_class": _reaction_class_from_bond_class(cls),
                "source": "rdkit_dealkylation",
            }
        )
    return out


def _apply_xtb_refinement(
    mol_h: Any,
    sites: List[Dict[str, Any]],
) -> None:
    if compute_substrate_bde_with_safeguard_for_mol is None:
        return
    for site in sites:
        cls = str(site.get("bond_class") or "")
        if not (cls.startswith("ch__") or cls.startswith("nh__") or cls.startswith("oh__")):
            continue
        atom_indices = site.get("atom_indices") or (-1, -1)
        if (
            not isinstance(atom_indices, tuple)
            or len(atom_indices) != 2
            or not all(isinstance(v, int) for v in atom_indices)
        ):
            continue
        heavy_idx, h_idx = int(atom_indices[0]), int(atom_indices[1])
        if heavy_idx < 0 or h_idx < 0:
            continue
        try:
            xtb = compute_substrate_bde_with_safeguard_for_mol(
                mol_h,
                (heavy_idx, h_idx),
                cls,
            )
        except Exception:
            continue
        bde_val = xtb.get("bde_kj_mol")
        if isinstance(bde_val, (int, float)):
            site["bde"] = float(bde_val)
            site["bde_source"] = str(xtb.get("source") or "xtb")
            site["xtb_status"] = xtb.get("status")


def _score_sites_by_bde(sites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    numeric_bdes = [float(s["bde"]) for s in sites if isinstance(s.get("bde"), (int, float))]
    if not numeric_bdes:
        for s in sites:
            s["score"] = 0.0
        return sites
    min_bde = min(numeric_bdes)
    max_bde = max(numeric_bdes)
    span = max(1e-9, max_bde - min_bde)
    for s in sites:
        bde = s.get("bde")
        if isinstance(bde, (int, float)):
            s["score"] = round((max_bde - float(bde)) / span, 6)
        else:
            s["score"] = 0.0
    return sites


def predict_metabolism_sites(
    smiles: str,
    topk: int = 5,
    use_xtb: bool = False,
    isoform_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Predict vulnerable metabolism sites ranked by bond vulnerability."""
    if Chem is None:
        raise RuntimeError("rdkit_unavailable")

    mol = Chem.MolFromSmiles(str(smiles or ""))
    if mol is None:
        raise ValueError("invalid_smiles")

    mol_h = Chem.AddHs(Chem.Mol(mol))
    sites = _collect_xh_sites(smiles=smiles, mol_h=mol_h)
    sites.extend(_collect_dealkylation_sites(mol))

    if use_xtb:
        _apply_xtb_refinement(mol_h, sites)

    _score_sites_by_bde(sites)
    for site in sites:
        score = site.get("score")
        if isinstance(score, (int, float)):
            site["score_norm"] = round(1.0 - float(score), 6)
        else:
            site["score_norm"] = None

    ranked: List[Dict[str, Any]] = []
    for idx, site in enumerate(sites):
        ranked.append(
            {
                "bond_id": idx + 1,
                "atom_indices": tuple(site.get("atom_indices") or (-1, -1)),
                "bond_class": str(site.get("bond_class") or "unknown"),
                "bde": round(float(site.get("bde") or 0.0), 3),
                "reaction_class": str(site.get("reaction_class") or "oxidation"),
                "score": round(float(site.get("score") or 0.0), 6),
                "score_norm": (
                    round(float(site.get("score_norm")), 6)
                    if isinstance(site.get("score_norm"), (int, float))
                    else None
                ),
                "source": site.get("source"),
            }
        )

    ranked_sorted = sorted(ranked, key=_stable_site_sort_key)
    k = max(1, int(topk or 5))
    top_prediction = ranked_sorted[0] if ranked_sorted else None
    out = {
        "ranked_sites": ranked_sorted[:k],
        "all_ranked_sites": ranked_sorted,
        "top_prediction": top_prediction,
        "isoform_hint": isoform_hint,
    }
    return _json_safe(out)


def _infer_cyp_label(cyp_payload: Dict[str, Any], isoform_hint: Optional[str]) -> Optional[str]:
    if isoform_hint:
        return str(isoform_hint).upper()
    ranked = list((cyp_payload or {}).get("ranked_sites") or [])
    if not ranked:
        return None
    top = ranked[0]
    fam = str(top.get("predicted_family") or "").lower()
    route = str(top.get("predicted_route") or "").lower()
    if "cytochrome" in fam or "p450" in route:
        return "CYP3A4_like"
    if "non_heme" in fam:
        return "non_CYP_nonheme_iron_like"
    return None


def predict_drug_metabolism(
    smiles: str,
    topk: int = 5,
    use_xtb: bool = False,
    isoform_hint: str | None = None,
) -> Dict[str, Any]:
    """
    Combine Part 2 CYP-like prediction and Part 3 site vulnerability ranking.
    """
    cyp = predict_cyp_like(smiles=smiles, isoform_hint=isoform_hint, topk=topk)
    sites = predict_metabolism_sites(
        smiles=smiles,
        topk=topk,
        use_xtb=use_xtb,
        isoform_hint=isoform_hint,
    )
    out = {
        "drug": smiles,
        "predicted_cyp": _infer_cyp_label(cyp, isoform_hint=isoform_hint),
        "top_metabolism_site": sites.get("top_prediction"),
        "reaction_class": (sites.get("top_prediction") or {}).get("reaction_class"),
        "ranked_sites": list(sites.get("ranked_sites") or []),
        "cyp_prediction": cyp,
        "site_prediction": sites,
    }
    return _json_safe(out)
