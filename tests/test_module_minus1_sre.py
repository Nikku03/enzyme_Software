from enzyme_software.modules.module_minus1_sre import run_module_minus1, _RDKIT_AVAILABLE
from enzyme_software.pipeline import run_pipeline


def test_module_minus1_sre_outputs():
    result = run_module_minus1(
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "acetyl_ester_C-O",
        "salicylic acid",
        {"ph_min": 7.0, "ph_max": 8.0, "temperature_c": 30.0},
    )
    assert result["status"] in {"PASS", "FAIL"}
    assert "bond360_profile" in result
    assert "fragment" in result
    assert "cpt_scores" in result
    assert "route_bias" in result
    assert "mechanism_eligibility" in result
    if _RDKIT_AVAILABLE:
        cpt_scores = result["cpt_scores"]
        assert cpt_scores["status"] in {"ok", "no_cpts", "no_fragment"}
        mm_results = cpt_scores.get("mm_results") or {}
        if cpt_scores["status"] == "ok":
            assert any(key.startswith("serine_hydrolase__") for key in mm_results)


def test_pipeline_includes_module_minus1():
    ctx = run_pipeline(
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "acetyl_ester_C-O",
        requested_output="salicylic acid",
    )
    assert "module_minus1_sre" in ctx.data
    module_minus1 = ctx.data["module_minus1_sre"]
    assert module_minus1.get("bond360_profile") is not None
