from __future__ import annotations

from enzyme_software.calibration import layer3_vina


def test_cyp_isoform_to_pdb_mapping():
    assert layer3_vina.CYP_ISOFORM_TO_PDB.get("CYP3A4") == "1TQN"
    assert layer3_vina.CYP_ISOFORM_TO_PDB.get("CYP2D6") == "2F9Q"


def test_dock_substrate_disabled_graceful(monkeypatch):
    monkeypatch.setenv("LAYER3_VINA_ENABLED", "0")
    result = layer3_vina.dock_substrate_in_cyp(smiles="CCO", cyp_isoform="CYP3A4")
    assert result["status"] == "disabled"


def test_dock_for_module2_extracts_smiles_and_cyp(monkeypatch):
    captured = {}

    def fake_dock_substrate_in_cyp(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "binding_energy_kcal": -6.2}

    monkeypatch.setattr(layer3_vina, "dock_substrate_in_cyp", fake_dock_substrate_in_cyp)
    job_card = {
        "smiles": "CCO",
        "cyp_prediction": {"primary_isoform": "CYP2D6"},
    }
    result = layer3_vina.dock_for_module2(job_card)
    assert result["status"] == "ok"
    assert captured["smiles"] == "CCO"
    assert captured["cyp_isoform"] == "CYP2D6"
