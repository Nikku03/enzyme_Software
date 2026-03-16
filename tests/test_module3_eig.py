from enzyme_software.context import PipelineContext
from enzyme_software.modules.module3_experiment_designer import (
    Module3ExperimentDesigner,
    _expected_information_gain,
)


def _base_ctx() -> PipelineContext:
    ctx = PipelineContext(smiles="CCO", target_bond="C-O")
    ctx.data["job_card"] = {
        "chosen_route": "serine_hydrolase",
        "constraints": {"ph_min": 6.5, "ph_max": 8.0},
    }
    ctx.data["module1_topogate"] = {
        "module1_confidence": {"retention": 0.6},
    }
    ctx.data["module2_active_site_refinement"] = {
        "module3_handoff": {"best_variant": {"variant_id": "V1", "label": "Clamp"}},
        "variant_set": [
            {"variant_id": "V1", "label": "Clamp", "rank": 1, "score": 1.2},
            {"variant_id": "V0", "label": "Baseline", "rank": 2, "score": 1.18},
        ],
        "selected_scaffold": {"k_pred_mean": 1.2, "model_risk": 0.2},
    }
    ctx.data["shared_io"] = {
        "input": {"condition_profile": {"pH": 7.0, "temperature_C": 30.0}},
        "outputs": {"module2": {}},
    }
    return ctx


def test_eig_prefers_mid_probability():
    eig_mid, _ = _expected_information_gain(0.5, 10.0)
    eig_high, _ = _expected_information_gain(0.9, 10.0)
    assert eig_mid > eig_high


def test_negative_control_and_arm_count():
    ctx = _base_ctx()
    designer = Module3ExperimentDesigner()
    ctx = designer.run(ctx)
    protocol = ctx.data["module3_experiment_designer"]["protocol_card"]
    assert len(protocol["arms"]) == 5
    arm_types = {arm.get("type") for arm in protocol["arms"]}
    assert "negative_control" in arm_types
