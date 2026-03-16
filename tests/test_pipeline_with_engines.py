from __future__ import annotations

from enzyme_software.context import OperationalConstraints
from enzyme_software.modules import module2_active_site_refinement as m2
from enzyme_software.pipeline import run_pipeline


def test_pipeline_with_engine_outputs(monkeypatch):
    monkeypatch.setattr(m2, "l3_is_layer3_vina_enabled", lambda: True)
    monkeypatch.setattr(m2, "l3_is_layer3_openmm_enabled", lambda: True)
    monkeypatch.setattr(
        m2,
        "l3_dock_for_module2",
        lambda job_card, constraints=None: {
            "status": "ok",
            "binding_energy_kcal": -7.2,
            "binding_energy_kj": -30.1,
            "n_poses": 5,
            "topogate_scores": {
                "access_score": 1.0,
                "reach_score": 0.85,
                "retention_score": 0.72,
                "composite": 0.82,
            },
            "distance_to_center_A": 3.4,
            "receptor_source": "cyp_isoform:CYP3A4->1TQN",
            "pdb_id": "1TQN",
        },
    )
    monkeypatch.setattr(
        m2,
        "l3_stability_for_module2",
        lambda docking_result, job_card: {
            "status": "ok",
            "verdict": "STABLE",
            "energy_stable": True,
            "minimization": {"energy_before_kj": -100.0, "energy_after_kj": -120.0},
            "md": {"total_time_ps": 50.0, "energy_stable": True, "final_pe_kj": -118.0},
        },
    )

    ctx = run_pipeline(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        target_bond="acetyl_ester_C-O",
        constraints=OperationalConstraints(enable_vina=True, enable_openmm=True),
    )
    module2 = ctx.data.get("module2_active_site_refinement") or {}

    assert module2.get("docking", {}).get("status") == "ok"
    assert module2.get("md_stability", {}).get("status") == "ok"
    assert module2.get("computational_engines_used") == ["vina", "openmm"]
    assert module2.get("composite_binding_score") == 0.892
    assert module2.get("binding_score_adjustment") == 0.157
