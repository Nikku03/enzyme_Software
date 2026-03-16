from __future__ import annotations

from computational_engines import (
    OpenMMEngine,
    PreparedMolecule,
    VinaEngine,
    XTBEngine,
    check_engines,
    prepare_molecule,
)


def test_engine_status_has_expected_keys():
    status = check_engines()
    for key in [
        "rdkit",
        "xtb",
        "xtb_path",
        "vina",
        "openmm",
        "pdbfixer",
        "meeko",
        "engines_available",
    ]:
        assert key in status


def test_prepare_molecule_invalid_smiles_returns_warning():
    prepared = prepare_molecule("NOT_A_SMILES")
    assert prepared.mol is None
    assert prepared.warnings


def test_xtb_engine_unavailable_behaviour():
    xtb = XTBEngine(xtb_path="/definitely/not/a/real/xtb")
    assert xtb.is_available() is False
    row = xtb.compute_bde(PreparedMolecule(smiles="C"), heavy_idx=0, h_idx=1)
    assert row["bde_kj_mol"] is None
    assert "error" in row

    # API compatibility with spec example:
    row2 = xtb.compute_bde(smiles="C", heavy_idx=0, h_idx=1)
    assert row2["bde_kj_mol"] is None
    assert "error" in row2


def test_vina_engine_interface_runs_without_vina_dependency():
    vina = VinaEngine()
    if not vina.is_available():
        out = vina.dock("missing_receptor.pdbqt", "CC")
        assert out.get("binding_energy") is None
        assert "error" in out

        # API compatibility with spec example:
        out2 = vina.dock(pdb_path="missing_receptor.pdb", smiles="CC")
        assert out2.get("binding_energy") is None
        assert "error" in out2


def test_openmm_engine_has_binding_stability_alias():
    omm = OpenMMEngine()
    assert hasattr(omm, "binding_stability")
    if not omm.is_available():
        out = omm.binding_stability("missing.pdb", n_steps=100)
        assert out.get("error") == "OpenMM not available"
