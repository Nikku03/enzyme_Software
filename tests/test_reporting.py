from __future__ import annotations

from enzyme_software.reporting import render_debug_report, render_scientific_report


def _sample_result():
    return {
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "target_bond": "acetyl_ester_C-O",
        "data": {
            "job_card": {
                "decision": "GO",
                "mechanism_route": {"primary": "serine_hydrolase"},
                "confidence": {"route": 0.8},
            },
            "shared_state": {
                "physics": {"derived": {"energy_ledger": {"deltaG_dagger_kJ": 88.5, "k_eff_s_inv": 0.0012}}},
                "chem": {"derived": {"reaction_family": "hydrolysis", "leaving_group_score": 0.8}},
                "bio": {"derived": {"protonation": {"factor": 0.6}}},
            },
            "module3_experiment_designer": {
                "protocol_card": {
                    "arms": [
                        {"arm_id": "A1", "type": "baseline", "conditions": {"pH": 7.0}},
                        {"arm_id": "A2", "type": "improved_conditions", "conditions": {"pH": 7.5}},
                    ]
                }
            },
        },
    }


def test_scientific_report_has_sections():
    report = render_scientific_report(_sample_result())
    assert "Decision summary" in report
    assert "Next best actions" in report


def test_debug_report_has_raw_json():
    report = render_debug_report(_sample_result())
    assert "RAW JSON" in report
