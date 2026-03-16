from __future__ import annotations

from enzyme_software.calibration.layer4_validation import (
    VALIDATION_CASES,
    run_validation,
)


def test_layer4_template_mode_has_all_cases():
    report = run_validation(pipeline_function=None)
    assert report["total"] == len(VALIDATION_CASES)
    assert report["templates"] == len(VALIDATION_CASES)
    assert report["passed"] == 0


def test_layer4_runner_with_mock_pipeline():
    def _mock_pipeline(smiles: str, target_bond: str):
        # Return minimal shape expected by run_validation extractor
        if target_bond == "C-H" and "FC(F)F" in smiles:
            decision = "NO_GO"
            route = "metallo_transfer_CF3"
            k_eff = 1e-3
        elif target_bond in {"ester", "amide"}:
            decision = "LOW_CONF_GO"
            route = "serine_hydrolase"
            k_eff = 10.0
        elif target_bond == "C-Cl":
            decision = "LOW_CONF_GO"
            route = "haloalkane_dehalogenase"
            k_eff = 1.0
        else:
            decision = "LOW_CONF_GO"
            route = "P450"
            k_eff = 1.0
        return {
            "pipeline_summary": {"results": {"decision": decision, "route": route}},
            "job_card": {"energy_ledger": {"k_eff_s_inv": k_eff}},
        }

    report = run_validation(_mock_pipeline)
    assert report["total"] == len(VALIDATION_CASES)
    assert report["failed"] + report["passed"] + report["errors"] == len(VALIDATION_CASES)
    assert report["errors"] == 0
