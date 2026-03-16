import os
import sys

from rdkit import Chem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from enzyme_software.modules.sre_atr import detect_groups, GroupRole
from enzyme_software.modules.sre_fragment_builder import (
    ChemicallyAwareFragmentBuilder,
    CapStrategy,
)


def _get_first_group(smiles: str, group_type: str):
    result = detect_groups(smiles)
    groups = [g for g in result.groups if g.group_type == group_type]
    assert groups, f"Expected group_type={group_type}"
    return groups[0], result


def _print_fragment_details(smiles, group, frag):
    atr = group.atoms[0].atr
    print("\n=== Fragment Build ===")
    print("SMILES:", smiles)
    print("Group type:", group.group_type)
    print("Parent SMILES:", frag.parent_smiles)
    print("Fragment SMILES:", frag.frag_smiles)
    print("Group roles:")
    for role, atom in group.roles.items():
        truth = atr.get_by_uuid(atom.atom_id)
        print(
            f"  - {role.value}: {atom.atom_id} ({truth.element}) parent_idx={truth.parent_index}"
        )
    print("Kept parent UUIDs:", len(frag.kept_parent_uuids))
    print("Warnings:", frag.warnings)
    print("Build notes:", frag.build_notes)
    print("Context metrics:", frag.context_metrics)
    print("Cut bonds:")
    for cut in frag.cut_bonds:
        print(
            f"  - bond_idx={cut.parent_bond_idx} atoms={cut.parent_atom_indices} "
            f"uuids={cut.parent_atom_uuids} order={cut.bond_order} aromatic={cut.is_aromatic}"
        )
    print("Cap records:")
    for cap in frag.cap_records:
        print(
            f"  - kept={cap.kept_atom_uuid} removed={cap.removed_atom_uuid} "
            f"cap={cap.cap_atom_symbol} strategy={cap.strategy.value} note={cap.notes}"
        )


def test_fragment_builder_methyl_acetate_ester():
    smiles = "CC(=O)OCC"
    group, result = _get_first_group(smiles, "ester")
    atr = result.groups[0].atoms[0].atr
    mol = Chem.MolFromSmiles(smiles)
    builder = ChemicallyAwareFragmentBuilder(max_heavy_atoms=20)
    frag = builder.build_from_group(atr, mol, group, cap_strategy=CapStrategy.HYDROGEN)
    _print_fragment_details(smiles, group, frag)
    assert frag.frag_smiles

    Chem.SanitizeMol(frag.frag_mol)
    required = [
        group.roles[GroupRole.CARBONYL_C].atom_id,
        group.roles[GroupRole.CARBONYL_O].atom_id,
        group.roles[GroupRole.HETERO_ATTACH].atom_id,
    ]
    for rid in required:
        assert rid in frag.parent_uuid_to_frag_idx
    assert frag.frag_mol.GetNumHeavyAtoms() <= 20
    assert not any(atom.GetAtomicNum() == 0 for atom in frag.frag_mol.GetAtoms())


def test_fragment_builder_aspirin_preserves_aromatic_and_acid():
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    group, result = _get_first_group(smiles, "ester")
    atr = result.groups[0].atoms[0].atr
    mol = Chem.MolFromSmiles(smiles)

    builder = ChemicallyAwareFragmentBuilder(max_heavy_atoms=30)
    frag = builder.build_from_group(atr, mol, group)
    _print_fragment_details(smiles, group, frag)

    aromatic_count = sum(1 for atom in frag.frag_mol.GetAtoms() if atom.GetIsAromatic())
    assert aromatic_count >= 6

    patt = Chem.MolFromSmarts("[CX3](=O)[OX1,OX2H,OX2-]")
    assert patt is not None
    matches = mol.GetSubstructMatches(patt)
    assert matches
    acid_atoms = set(matches[0])
    acid_uuids = {atr.atom_id_from_parent_index(i) for i in acid_atoms}
    assert acid_uuids.issubset(frag.kept_parent_uuids)


def test_fragment_builder_aryl_halide_preserves_ring():
    smiles = "Clc1ccccc1"
    group, result = _get_first_group(smiles, "aryl_halide")
    atr = result.groups[0].atoms[0].atr
    mol = Chem.MolFromSmiles(smiles)

    builder = ChemicallyAwareFragmentBuilder(max_heavy_atoms=15)
    frag = builder.build_from_group(atr, mol, group)
    _print_fragment_details(smiles, group, frag)
    if not frag.cut_bonds:
        assert any("no_capping_needed" in note for note in frag.build_notes)
    aromatic_count = sum(1 for atom in frag.frag_mol.GetAtoms() if atom.GetIsAromatic())
    assert aromatic_count >= 6
    assert any(atom.GetSymbol() == "Cl" for atom in frag.frag_mol.GetAtoms())


def test_fragment_builder_epoxide_keeps_ring():
    smiles = "CC1OC1"
    group, result = _get_first_group(smiles, "epoxide")
    atr = result.groups[0].atoms[0].atr
    mol = Chem.MolFromSmiles(smiles)

    builder = ChemicallyAwareFragmentBuilder(max_heavy_atoms=10)
    frag = builder.build_from_group(atr, mol, group)
    _print_fragment_details(smiles, group, frag)
    ring_info = frag.frag_mol.GetRingInfo()
    ring_sizes = [len(r) for r in ring_info.AtomRings()]
    assert 3 in ring_sizes


def test_fragment_builder_graceful_degradation_max_atoms():
    smiles = "CCCCCCCCCC(=O)OCCCCCCCCCC"
    group, result = _get_first_group(smiles, "ester")
    atr = result.groups[0].atoms[0].atr
    mol = Chem.MolFromSmiles(smiles)

    builder = ChemicallyAwareFragmentBuilder(max_heavy_atoms=12, default_radius=1)
    frag = builder.build_from_group(atr, mol, group)
    _print_fragment_details(smiles, group, frag)
    assert any("truncation" in w for w in frag.warnings)
    assert frag.frag_mol.GetNumHeavyAtoms() <= builder.max_heavy_atoms
    assert frag.context_metrics.get("kept_heavy_atoms") == builder.max_heavy_atoms


def test_fragment_builder_mapping_is_bijective():
    smiles = "CC(=O)OCC"
    group, result = _get_first_group(smiles, "ester")
    atr = result.groups[0].atoms[0].atr
    mol = Chem.MolFromSmiles(smiles)

    builder = ChemicallyAwareFragmentBuilder(max_heavy_atoms=20)
    frag = builder.build_from_group(atr, mol, group)
    _print_fragment_details(smiles, group, frag)

    assert len(frag.parent_uuid_to_frag_idx) == len(frag.frag_idx_to_parent_uuid)
    mapped = {
        frag.frag_mol.GetAtomWithIdx(i).GetProp("parent_uuid")
        for i in frag.frag_idx_to_parent_uuid.keys()
    }
    assert len(mapped) == len(frag.parent_uuid_to_frag_idx)
