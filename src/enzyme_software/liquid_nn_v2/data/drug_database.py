from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .training_drugs import TRAINING_DRUGS

try:
    from enzyme_software.calibration.drug_metabolism_db import get_drug
except Exception:  # pragma: no cover - optional import
    get_drug = None

CYP_CLASSES = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]


def _entry(name: str, fallback_smiles: str, primary_cyp: str, site_atoms: List[int], expected_bond_class: str) -> Dict[str, object]:
    ref = get_drug(name) if get_drug is not None else None
    smiles = ref.get("smiles") if isinstance(ref, dict) and ref.get("smiles") else fallback_smiles
    cyp = ref.get("primary_cyp") if isinstance(ref, dict) and ref.get("primary_cyp") else primary_cyp
    return {
        "name": name,
        "smiles": smiles,
        "primary_cyp": cyp,
        "site_atoms": list(site_atoms),
        "expected_bond_class": expected_bond_class,
    }


DRUG_DATABASE: Dict[str, Dict[str, object]] = {
    "ibuprofen": _entry("ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CYP2C9", [7], "benzylic"),
    "warfarin": _entry("warfarin", "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O", "CYP2C9", [11], "aryl"),
    "diclofenac": _entry("diclofenac", "OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl", "CYP2C9", [14], "aryl"),
    "tolbutamide": _entry("tolbutamide", "CCCC1=CC=C(C=C1)S(=O)(=O)NC(=O)NC", "CYP2C9", [0, 1, 2], "primary_CH"),
    "codeine": _entry("codeine", "COC1=CC2=C(C=C1)C3C4C=CC(C2C3O)N(C)CC4", "CYP2D6", [0], "alpha_hetero"),
    "dextromethorphan": _entry("dextromethorphan", "COC1=CC2=C(C=C1)C3C4C=CC2N(C)CC3C4", "CYP2D6", [0], "alpha_hetero"),
    "metoprolol": _entry("metoprolol", "CC(C)NCC(O)COC1=CC=C(C=C1)CCOC", "CYP2D6", [18], "alpha_hetero"),
    "omeprazole": _entry("omeprazole", "COC1=CC2=NC(=NC2=C(C=C1)C)CS(=O)C3=NC4=CC=CC=C4N3", "CYP2C19", [12], "aryl"),
    "clopidogrel": _entry("clopidogrel", "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=CC=CS3C2", "CYP2C19", [4], "alpha_hetero"),
    "midazolam": _entry("midazolam", "CC1=NC=C2N1C3=CC=C(Cl)C=C3C(=NC2)C4=CC=CC=C4F", "CYP3A4", [8, 14], "benzylic"),
    "testosterone": _entry("testosterone", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C", "CYP3A4", [6], "allylic"),
    "nifedipine": _entry("nifedipine", "COC(=O)C1=C(C)NC(=C(C1C2=CC=CC=C2[N+]([O-])=O)C(=O)OC)C", "CYP3A4", [7], "allylic"),
    "caffeine": _entry("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CYP1A2", [0, 10, 12], "alpha_hetero"),
    "theophylline": _entry("theophylline", "CN1C2=C(C(=O)N(C1=O)C)NC=N2", "CYP1A2", [0, 8], "alpha_hetero"),
}

EXTENDED_DRUGS = list(TRAINING_DRUGS)


def load_training_dataset(path: str = "data/cyp_metabolism_dataset.json") -> List[Dict[str, object]]:
    """Load the built CYP metabolism dataset JSON."""
    with Path(path).open() as handle:
        data = json.load(handle)
    return list(data.get("drugs", []))
