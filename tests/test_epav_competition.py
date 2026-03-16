import math

from rdkit import Chem

from enzyme_software.modules.sre_atr import detect_groups, GroupRole
from enzyme_software.cpt.geometric_cpts import ElectronicPropertiesAttackValidationCPT


def _role_to_idx(smiles: str, group_type: str):
    mol = Chem.MolFromSmiles(smiles)
    groups = [g for g in detect_groups(smiles).groups if g.group_type == group_type]
    assert groups, f"No group {group_type} found for {smiles}"
    g = groups[0]
    role_to_idx = {
        "carbonyl_c": g.roles[GroupRole.CARBONYL_C].original_index,
        "carbonyl_o": g.roles[GroupRole.CARBONYL_O].original_index,
        "hetero_attach": g.roles[GroupRole.HETERO_ATTACH].original_index,
    }
    return mol, g, role_to_idx


def test_epav_single_site():
    smiles = "CC(=O)OCC"
    mol, g, role_to_idx = _role_to_idx(smiles, "ester")
    epav = ElectronicPropertiesAttackValidationCPT()
    res = epav.run(mol, role_to_idx, group_type=g.group_type)

    comp = res.data["competition"]
    assert comp["best_other_idx"] is None
    assert comp["best_other_score"] == 0.0
    assert comp["gap"] == 1.0
    assert res.breakdown["competition"] == 1.0
    assert len(res.data["candidate_site_scores"]) == 1


def test_epav_equivalent_sites_malonate():
    smiles = "COC(=O)CC(=O)OC"
    mol, g, role_to_idx = _role_to_idx(smiles, "ester")
    epav = ElectronicPropertiesAttackValidationCPT(
        competition_mode="allow_equivalents",
        equivalent_gap_eps=0.02,
    )
    res = epav.run(mol, role_to_idx, group_type=g.group_type)

    comp = res.data["competition"]
    assert len(res.data["candidate_site_scores"]) == 2

    gap = comp["gap"]
    if comp.get("equivalent_sites_detected"):
        assert math.isclose(res.breakdown["competition"], 1.0, rel_tol=1e-6)
        assert "equivalent_sites_detected" in res.warnings
        assert abs(gap) < 0.02
    else:
        assert 0.0 <= res.breakdown["competition"] < 0.2


def test_epav_mixed_competition():
    smiles = "O=CCOC(=O)C"
    mol, g, role_to_idx = _role_to_idx(smiles, "ester")
    epav = ElectronicPropertiesAttackValidationCPT()
    res = epav.run(mol, role_to_idx, group_type=g.group_type)

    comp = res.data["competition"]
    assert comp["best_other_idx"] is not None
    assert 0.0 < res.breakdown["competition"] < 1.0
    assert comp["gap"] > 0.05
