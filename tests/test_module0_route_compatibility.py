from __future__ import annotations

import pytest

from enzyme_software.context import OperationalConstraints
from enzyme_software.modules import module0_strategy_router as m0


def _condition_profile():
    constraints = OperationalConstraints(ph_min=6.5, ph_max=8.0, temperature_c=30.0)
    return m0._condition_profile_from_constraints(constraints)


def _structure_summary(mol):
    return {
        "atom_count": mol.GetNumAtoms(),
        "hetero_atoms": sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)),
        "ring_count": mol.GetRingInfo().NumRings(),
    }


def test_camphor_routes_to_p450_prior() -> None:
    if m0.Chem is None:
        pytest.skip("RDKit not available")

    smiles = "CC1(C)C2CCC1(C)C(=O)C2"
    mol = m0.Chem.MolFromSmiles(smiles)
    assert mol is not None

    assert m0.classify_substrate_shape(mol) == "hydrophobic_organic"
    assert m0.get_route_prior(mol, "P450") < m0.get_route_prior(mol, "NHI")

    bond_context = {"bond_class": "C-H", "bond_type": "C-H"}
    bond_context.update(m0._detect_substrate_shape_features(mol))
    priors = m0._physics_route_priors(
        route_candidates=["P450", "non_heme_iron", "radical_SAM"],
        smiles=smiles,
        target_bond="C-H",
        bond_context=bond_context,
        structure_summary=_structure_summary(mol),
        condition_profile=_condition_profile(),
    )

    assert priors["P450"]["prior_feasibility"] > priors["non_heme_iron"]["prior_feasibility"]


def test_taurine_routes_to_nhi_prior() -> None:
    if m0.Chem is None:
        pytest.skip("RDKit not available")

    smiles = "NCCS(=O)(=O)O"
    mol = m0.Chem.MolFromSmiles(smiles)
    assert mol is not None

    assert m0.classify_substrate_shape(mol) == "polar_amino_acid_deriv"
    assert m0.get_route_prior(mol, "NHI") < m0.get_route_prior(mol, "P450")

    bond_context = {"bond_class": "C-H", "bond_type": "C-H"}
    bond_context.update(m0._detect_substrate_shape_features(mol))
    priors = m0._physics_route_priors(
        route_candidates=["P450", "non_heme_iron", "radical_SAM"],
        smiles=smiles,
        target_bond="C-H",
        bond_context=bond_context,
        structure_summary=_structure_summary(mol),
        condition_profile=_condition_profile(),
    )

    assert priors["non_heme_iron"]["prior_feasibility"] > priors["P450"]["prior_feasibility"]


def test_taurine_bond_class_and_reference_bde() -> None:
    if m0.Chem is None:
        pytest.skip("RDKit not available")

    mol = m0.Chem.AddHs(m0.Chem.MolFromSmiles("NCCS(=O)(=O)O"))
    assert mol is not None

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        if not any(neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()):
            continue
        if any(neighbor.GetAtomicNum() == 7 for neighbor in atom.GetNeighbors()):
            assert m0.classify_ch_bond_class(mol, atom.GetIdx()) == "alpha_amino_CH"
            assert m0.get_nhi_reference_bde(mol, atom.GetIdx()) == 385.0
            return

    pytest.fail("Could not find taurine alpha-amino carbon")


def test_camphor_bond_class_is_aliphatic() -> None:
    if m0.Chem is None:
        pytest.skip("RDKit not available")

    mol = m0.Chem.AddHs(m0.Chem.MolFromSmiles("CC1(C)C2CCC1(C)C(=O)C2"))
    assert mol is not None

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and any(
            neighbor.GetAtomicNum() == 1 for neighbor in atom.GetNeighbors()
        ):
            bond_class = m0.classify_ch_bond_class(mol, atom.GetIdx())
            assert bond_class in {"aliphatic_CH", "default"}
            return

    pytest.fail("Could not find camphor aliphatic carbon with hydrogen")


def test_non_heme_reference_bde_lowers_taurine_kcat() -> None:
    taurine = m0._predict_kcat_brenda_anchored(
        route_name="non_heme_iron",
        bond_class="C-H",
        track="radical_hat",
        bde_kj_mol=400.0,
        substrate_smiles="NCCS(=O)(=O)O",
        temperature_K=298.15,
    )
    camphor = m0._predict_kcat_brenda_anchored(
        route_name="non_heme_iron",
        bond_class="C-H",
        track="radical_hat",
        bde_kj_mol=400.0,
        substrate_smiles="CC1(C)C2CCC1(C)C(=O)C2",
        temperature_K=298.15,
    )

    assert taurine["components"]["reference_bde_class"] == "alpha_amino_CH"
    assert camphor["components"]["reference_bde_class"] in {"aliphatic_CH", "default"}
    assert taurine["predicted_kcat_s_inv"] < camphor["predicted_kcat_s_inv"]
