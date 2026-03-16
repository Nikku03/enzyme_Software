import os
import sys

from rdkit import Chem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from enzyme_software.modules.sre_atr import (
    AtomicTruthRegistry,
    AtomRef,
    BondRef,
    GroupRole,
    detect_groups,
    resolve_bond,
)


def _groups_of_type(result, group_type: str):
    return [g for g in result.groups if g.group_type == group_type]


def test_atomref_and_bondref_sanity():
    atr = AtomicTruthRegistry.from_smiles("CCO")
    a_ref = AtomRef(atr, atr.atom_id_from_parent_index(0))
    b_ref = AtomRef(atr, atr.atom_id_from_parent_index(1))

    print("\n[atomref] a_ref:", a_ref.atom_id, a_ref.element, a_ref.original_index)
    print("[atomref] b_ref:", b_ref.atom_id, b_ref.element, b_ref.original_index)

    assert a_ref.element == "C"
    assert isinstance(a_ref.formal_charge, int)
    assert isinstance(a_ref.original_index, int)

    bond = atr.parent_mol.GetBondBetweenAtoms(a_ref.original_index, b_ref.original_index)
    assert bond is not None
    bond_ref = BondRef(
        a=a_ref,
        b=b_ref,
        bond_order=float(bond.GetBondTypeAsDouble()),
        is_aromatic=bool(bond.GetIsAromatic()),
        rdkit_bond_type=str(bond.GetBondType()),
    )
    print("[bondref] order:", bond_ref.bond_order, "aromatic:", bond_ref.is_aromatic)
    bond_ref.validate_connected(atr.parent_mol)


def test_ester_detection():
    result = detect_groups("CC(=O)OCC")
    esters = _groups_of_type(result, "ester")
    print("\n[ester] total groups:", len(result.groups))
    for g in esters:
        print("[ester] roles:", {r.value: a.element for r, a in g.roles.items()})
        print("[ester] evidence:", g.evidence)
    assert esters, "Expected at least one ester group"
    group = esters[0]
    assert GroupRole.CARBONYL_C in group.roles
    assert GroupRole.CARBONYL_O in group.roles
    assert GroupRole.HETERO_ATTACH in group.roles


def test_amide_detection():
    result = detect_groups("CC(=O)NC")
    amides = _groups_of_type(result, "amide")
    print("\n[amide] total groups:", len(result.groups))
    for g in amides:
        print("[amide] roles:", {r.value: a.element for r, a in g.roles.items()})
        print("[amide] evidence:", g.evidence)
    assert amides, "Expected at least one amide group"
    group = amides[0]
    assert GroupRole.CARBONYL_C in group.roles
    assert GroupRole.CARBONYL_O in group.roles
    assert GroupRole.HETERO_ATTACH in group.roles


def test_aryl_halide_detection():
    result = detect_groups("c1ccccc1Cl")
    aryls = _groups_of_type(result, "aryl_halide")
    print("\n[aryl_halide] total groups:", len(result.groups))
    for g in aryls:
        print("[aryl_halide] roles:", {r.value: a.element for r, a in g.roles.items()})
        print("[aryl_halide] evidence:", g.evidence)
    assert aryls, "Expected at least one aryl halide group"
    group = aryls[0]
    assert GroupRole.HALOGEN in group.roles
    assert GroupRole.ARYL_C in group.roles


def test_epoxide_detection():
    result = detect_groups("CC1OC1")
    epoxides = _groups_of_type(result, "epoxide")
    print("\n[epoxide] total groups:", len(result.groups))
    for g in epoxides:
        print("[epoxide] roles:", {r.value: a.element for r, a in g.roles.items()})
        print("[epoxide] evidence:", g.evidence)
    assert epoxides, "Expected at least one epoxide group"
    group = epoxides[0]
    assert GroupRole.EPOXIDE_O in group.roles
    assert GroupRole.EPOXIDE_C1 in group.roles
    assert GroupRole.EPOXIDE_C2 in group.roles


def test_negative_no_ester_in_ether():
    result = detect_groups("CCOCC")
    esters = _groups_of_type(result, "ester")
    print("\n[ether] ester groups found:", len(esters))
    assert not esters, "Did not expect ester group in diethyl ether"


def test_bond_resolution_by_indices():
    smiles = "CC(=O)OCC"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    atr = AtomicTruthRegistry.from_smiles(smiles)

    carbonyl_c = None
    hetero_o = None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        for bond in atom.GetBonds():
            if bond.GetBondTypeAsDouble() == 2.0 and bond.GetOtherAtom(atom).GetSymbol() == "O":
                carbonyl_c = atom.GetIdx()
            if bond.GetBondTypeAsDouble() == 1.0 and bond.GetOtherAtom(atom).GetSymbol() == "O":
                hetero_o = bond.GetOtherAtom(atom).GetIdx()
        if carbonyl_c is not None and hetero_o is not None:
            break
    assert carbonyl_c is not None and hetero_o is not None

    result = resolve_bond(mol, atr=atr, target_indices=(carbonyl_c, hetero_o))
    print("\n[bond indices] selected:", result.selected)
    assert result.ok
    assert result.selected is not None


def test_bond_resolution_by_smarts():
    smiles = "CC(=O)OCC"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    atr = AtomicTruthRegistry.from_smiles(smiles)

    result = resolve_bond(
        mol,
        atr=atr,
        target_smarts="[CX3](=O)[OX2][#6]",
        target_smarts_bond=(0, 2),
    )
    print("\n[bond smarts] candidates:", len(result.candidates), "selected:", result.selected)
    assert result.ok
    assert result.selected is not None


def test_bond_resolution_by_roles():
    smiles = "CC(=O)OCC"
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    atr = AtomicTruthRegistry.from_smiles(smiles)

    result = resolve_bond(
        mol,
        atr=atr,
        target_group_type="ester",
        target_role_pair=(GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH),
    )
    print("\n[bond roles] candidates:", len(result.candidates), "selected:", result.selected)
    assert result.ok
    assert result.selected is not None


def test_ambiguity_two_esters():
    result = detect_groups("CC(=O)OCC(=O)OCC")
    esters = _groups_of_type(result, "ester")
    print("\n[ambiguity esters] ester groups:", len(esters))
    assert len(esters) >= 2

    atr = AtomicTruthRegistry.from_smiles("CC(=O)OCC(=O)OCC")
    mol = Chem.MolFromSmiles("CC(=O)OCC(=O)OCC")
    resolved = resolve_bond(
        mol,
        atr=atr,
        target_group_type="ester",
        target_role_pair=(GroupRole.CARBONYL_C, GroupRole.HETERO_ATTACH),
    )
    print("[ambiguity esters] candidates:", len(resolved.candidates), "selected:", resolved.selected)
    assert resolved.ok
    assert len(resolved.candidates) >= 2


def test_ambiguity_multiple_aryl_halides():
    result = detect_groups("Clc1cccc(Cl)c1")
    aryls = _groups_of_type(result, "aryl_halide")
    print("\n[ambiguity aryl halides] groups:", len(aryls))
    assert len(aryls) >= 2


def test_ambiguity_ester_and_amide():
    result = detect_groups("CC(=O)OCC(=O)N")
    esters = _groups_of_type(result, "ester")
    amides = _groups_of_type(result, "amide")
    print("\n[ambiguity ester+amide] esters:", len(esters), "amides:", len(amides))
    assert esters
    assert amides


def test_aromatic_bond_typing_sanity():
    smiles = "c1ccccc1Cl"
    atr = AtomicTruthRegistry.from_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    result = resolve_bond(
        mol,
        atr=atr,
        target_group_type="aryl_halide",
        target_role_pair=(GroupRole.ARYL_C, GroupRole.HALOGEN),
    )
    assert result.ok
    assert result.selected is not None
    print("\n[aromatic bond] is_aromatic:", result.selected.is_aromatic)
    assert result.selected.is_aromatic
