from __future__ import annotations

from collections import Counter
from typing import Dict, List

try:
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None


def _training_entry(
    name: str,
    smiles: str,
    primary_cyp: str,
    site_atom_indices: List[int],
    expected_bond_class: str,
    source: str = "DrugBank",
) -> Dict[str, object]:
    return {
        "name": name,
        "smiles": smiles,
        "primary_cyp": primary_cyp,
        "site_atom_indices": list(site_atom_indices),
        "expected_bond_class": expected_bond_class,
        "source": source,
    }


TRAINING_DRUGS: List[Dict[str, object]] = [
    # CYP1A2 (6)
    _training_entry(
        "Caffeine",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "CYP1A2",
        [0, 10, 12],
        "alpha_hetero",
    ),
    _training_entry(
        "Theophylline",
        "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
        "CYP1A2",
        [0, 8],
        "alpha_hetero",
    ),
    _training_entry(
        "Tacrine",
        "Nc1c2CCCCc2nc2ccccc12",
        "CYP1A2",
        [7],
        "benzylic",
    ),
    _training_entry(
        "Phenacetin",
        "CCOc1ccc(NC(C)=O)cc1",
        "CYP1A2",
        [0],
        "alpha_hetero",
    ),
    _training_entry(
        "Melatonin",
        "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",
        "CYP1A2",
        [0],
        "alpha_hetero",
    ),
    _training_entry(
        "Clozapine",
        "CN1CCN(CC1)C2=Nc3cc(Cl)ccc3Nc4ccccc24",
        "CYP1A2",
        [0],
        "alpha_hetero",
    ),
    # CYP2C9 (6)
    _training_entry(
        "Ibuprofen",
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "CYP2C9",
        [3],
        "benzylic",
    ),
    _training_entry(
        "Warfarin",
        "CC(=O)C[C@@H](c1ccccc1)c1c(O)c2ccccc2oc1=O",
        "CYP2C9",
        [11],
        "aryl",
    ),
    _training_entry(
        "Diclofenac",
        "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
        "CYP2C9",
        [9],
        "aryl",
    ),
    _training_entry(
        "Tolbutamide",
        "Cc1ccc(cc1)S(=O)(=O)NC(=O)NCCCC",
        "CYP2C9",
        [13],
        "primary_CH",
    ),
    _training_entry(
        "Flurbiprofen",
        "CC(C)c1cccc(c1)C(C)C(=O)O",
        "CYP2C9",
        [3],
        "benzylic",
    ),
    _training_entry(
        "Naproxen",
        "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
        "CYP2C9",
        [10],
        "benzylic",
    ),
    # CYP2C19 (6)
    _training_entry(
        "Omeprazole",
        "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
        "CYP2C19",
        [12],
        "aryl",
    ),
    _training_entry(
        "Clopidogrel",
        "COC(=O)[C@H](c1ccc(Cl)cc1)N1CCc2sccc2C1",
        "CYP2C19",
        [1],
        "alpha_hetero",
    ),
    _training_entry(
        "Lansoprazole",
        "COc1cc2ncc(CS(=O)c3ncc(C)c(OC)c3C)nc2cc1OC",
        "CYP2C19",
        [12],
        "aryl",
    ),
    _training_entry(
        "Pantoprazole",
        "COc1cc2ncc(CS(=O)c3nc(C)c(OC(F)F)cc3OC)nc2cc1OC",
        "CYP2C19",
        [12],
        "aryl",
    ),
    _training_entry(
        "Diazepam",
        "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
        "CYP2C19",
        [0],
        "alpha_hetero",
    ),
    _training_entry(
        "Mephenytoin",
        "CC1(c2ccccc2)NC(=O)NC1=O",
        "CYP2C19",
        [0],
        "alpha_hetero",
    ),
    # CYP2D6 (6)
    _training_entry(
        "Codeine",
        "COc1ccc2C[C@H]3N(C)CC[C@@]45[C@@H](Oc1c24)[C@H](O)C=C[C@@H]35",
        "CYP2D6",
        [0],
        "alpha_hetero",
    ),
    _training_entry(
        "Dextromethorphan",
        "COc1ccc2C[C@H]3N(C)CC[C@@]45[C@@H](Oc1c24)[C@@H](C=C[C@@H]35)O",
        "CYP2D6",
        [0],
        "alpha_hetero",
    ),
    _training_entry(
        "Metoprolol",
        "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        "CYP2D6",
        [0],
        "alpha_hetero",
    ),
    _training_entry(
        "Tramadol",
        "COc1cccc([C@](O)(CN(C)C)C)c1",
        "CYP2D6",
        [0],
        "alpha_hetero",
    ),
    _training_entry(
        "Propranolol",
        "CC(C)NCC(O)COc1cccc2ccccc12",
        "CYP2D6",
        [0],
        "alpha_hetero",
    ),
    _training_entry(
        "Venlafaxine",
        "COc1ccc(cc1)C(CN(C)C)C(O)c1ccccc1",
        "CYP2D6",
        [0],
        "alpha_hetero",
    ),
    # CYP3A4 (6)
    _training_entry(
        "Midazolam",
        "Clc1ccc2c(c1)C(=NC1=CN(C)C=N1)c1cc(F)ccc1-2",
        "CYP3A4",
        [8, 14],
        "benzylic",
    ),
    _training_entry(
        "Testosterone",
        "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@@]34C)[C@@H]1CC[C@@H]2O",
        "CYP3A4",
        [6],
        "allylic",
    ),
    _training_entry(
        "Nifedipine",
        "COC(=O)C1=C(C)NC(C)=C(C1c1ccccc1[N+](=O)[O-])C(=O)OC",
        "CYP3A4",
        [7],
        "allylic",
    ),
    _training_entry(
        "Lidocaine",
        "CCN(CC)C(=O)c1ccccc1N(C)C",
        "CYP3A4",
        [0],
        "alpha_hetero",
    ),
    _training_entry(
        "Tamoxifen",
        "CC/C(=C(/c1ccccc1)c1ccc(OCCN(C)C)cc1)/c1ccccc1",
        "CYP3A4",
        [0],
        "alpha_hetero",
    ),
    _training_entry(
        "Simvastatin",
        "CCC(C)(C)C(=O)O[C@H]1C[C@@H](O)[C@H](C=C[C@@H]2[C@@H](CC[C@H]3CC(=O)OC23)C1)OC(=O)C(C)C",
        "CYP3A4",
        [16],
        "allylic",
    ),
]


TRAINING_DRUGS_BY_CYP: Dict[str, List[Dict[str, object]]] = {}
for entry in TRAINING_DRUGS:
    TRAINING_DRUGS_BY_CYP.setdefault(str(entry["primary_cyp"]), []).append(entry)


TRAINING_DRUG_COUNTS = dict(Counter(str(entry["primary_cyp"]) for entry in TRAINING_DRUGS))


__all__ = ["TRAINING_DRUGS", "TRAINING_DRUGS_BY_CYP", "TRAINING_DRUG_COUNTS", "validate_training_drugs"]


def validate_training_drugs(verbose: bool = False) -> bool:
    """Validate that all curated training drugs have usable SoM annotations."""
    issues = []
    if Chem is None:
        if verbose:
            print("RDKit unavailable; skipping training-drug validation")
        return True
    for drug in TRAINING_DRUGS:
        smiles = str(drug.get("smiles", ""))
        sites = list(drug.get("site_atom_indices", []))
        if not sites:
            issues.append(f"{drug['name']}: No site_atom_indices")
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            issues.append(f"{drug['name']}: Invalid SMILES")
            continue
        num_atoms = mol.GetNumAtoms()
        for idx in sites:
            if int(idx) >= num_atoms or int(idx) < 0:
                issues.append(f"{drug['name']}: site index {idx} out of range for num_atoms={num_atoms}")
    if verbose:
        if issues:
            print("VALIDATION ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("All training drugs validated OK")
    return len(issues) == 0
