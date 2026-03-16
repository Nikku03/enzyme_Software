from __future__ import annotations

from enzyme_software.calibration import layer3_openmm


def test_openmm_disabled_graceful(monkeypatch):
    monkeypatch.setenv("LAYER3_OPENMM_ENABLED", "0")
    result = layer3_openmm.compute_binding_stability("missing_complex.pdb")
    assert result["status"] == "disabled"


def test_stability_for_module2_skips_without_receptor():
    result = layer3_openmm.stability_for_module2({}, {})
    assert result["status"] == "skipped"
