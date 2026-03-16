from __future__ import annotations

import pytest

from enzyme_software.calibration import layer3_xtb as l3


def test_layer3_bde_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ENZYME_LAYER3_XTB", raising=False)
    out = l3.compute_substrate_bde_xtb("CC", (0, 2))
    assert out.get("status") == "disabled"
    assert out.get("bde_kj_mol") is None


def test_layer3_strain_heuristic_fallback_when_xtb_missing(monkeypatch):
    monkeypatch.setenv("ENZYME_LAYER3_XTB", "1")
    monkeypatch.setattr(l3, "xtb_available", lambda xtb_path="xtb": False)
    out = l3.compute_substrate_strain("CCO")
    assert out.get("source") == "heuristic_fallback"
    assert isinstance(out.get("strain_kj_mol"), (int, float))


def test_layer3_bde_safeguard_uses_lookup_when_xtb_deviates(monkeypatch):
    monkeypatch.setenv("ENZYME_LAYER3_XTB", "1")
    monkeypatch.setattr(
        l3,
        "compute_substrate_bde_xtb_for_mol",
        lambda *args, **kwargs: {"status": "ok", "source": "layer3_xtb_gfn2", "bde_kj_mol": 550.0},
    )
    payload = l3.compute_substrate_bde_with_safeguard_for_mol(
        mol=None,  # mocked out
        bond_atom_indices=(0, 1),
        bond_class="ch__secondary",
    )
    assert payload.get("source") == "lookup_safeguard"
    assert payload.get("bde_kj_mol") == pytest.approx(412.5, abs=1e-6)
    assert isinstance(payload.get("deviation_kj_mol"), (int, float))


def test_layer3_bde_safeguard_accepts_xtb_when_close(monkeypatch):
    monkeypatch.setenv("ENZYME_LAYER3_XTB", "1")
    monkeypatch.setattr(
        l3,
        "compute_substrate_bde_xtb_for_mol",
        lambda *args, **kwargs: {"status": "ok", "source": "layer3_xtb_gfn2", "bde_kj_mol": 418.0},
    )
    payload = l3.compute_substrate_bde_with_safeguard_for_mol(
        mol=None,  # mocked out
        bond_atom_indices=(0, 1),
        bond_class="ch__secondary",
    )
    assert payload.get("source") == "xtb_validated"
    # v3: lookup + damped delta-xTB correction
    # lookup=412.5, xtb=418.0, correction=0.5*(+5.5)=+2.75 -> 415.25
    assert payload.get("bde_kj_mol") == pytest.approx(415.25, abs=1e-6)
    assert payload.get("delta_xtb_kj_mol") == pytest.approx(5.5, abs=1e-6)
    assert payload.get("correction_applied_kj_mol") == pytest.approx(2.75, abs=1e-6)


def test_module_minus1_radical_payload_includes_layer3_status(monkeypatch):
    pytest.importorskip("rdkit")
    from enzyme_software.modules.module_minus1_reactivity_hub import run_module_minus1_reactivity_hub

    monkeypatch.delenv("ENZYME_LAYER3_XTB", raising=False)
    out = run_module_minus1_reactivity_hub(
        smiles="CCCC",
        target_bond="C-H",
        requested_output=None,
        constraints={},
    )
    bde = (out.get("cpt_scores") or {}).get("bde") or {}
    assert "xtb_status" in bde
    assert "xtb_bde_kj_mol" in bde


def test_module2_strain_floor_metadata_present():
    pytest.importorskip("rdkit")
    from enzyme_software.modules.module2_active_site_refinement import _apply_variant_energy_scoring

    variants = [
        {
            "variant_id": "V_struct",
            "delta_deltaG_dagger_kJ_per_mol": 0.0,
            "requires_structural_localization": True,
            "strain_displacement_A": 0.0,
            "score": 0.0,
        }
    ]
    selected = {"delta_g_mean": 20.0}
    job_card = {
        "smiles": "CCO",
        "reaction_condition_field": {"given_conditions": {"temperature_c": 30.0}},
        "condition_profile": {},
    }
    scored = _apply_variant_energy_scoring(variants, selected, job_card)
    v = scored[0]
    assert isinstance(v.get("strain_energy_kJ_per_mol"), (int, float))
    assert v.get("strain_floor_source") is not None
    assert v.get("strain_floor_status") is not None
