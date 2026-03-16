from enzyme_software.context import PipelineContext
from enzyme_software.domain import ConditionProfile
from enzyme_software.mathcore.bayes_dag_router import BayesianDAGRouter
from enzyme_software.modules.module3_experiment_designer import Module3ExperimentDesigner


def _base_ctx() -> PipelineContext:
    ctx = PipelineContext(smiles="CCO", target_bond="C-O")
    ctx.data["job_card"] = {
        "chosen_route": "serine_hydrolase",
        "matched_bins": {
            "condition_bin": "ph_6_8|temp_295_305",
            "substrate_bin": "ester",
            "catalyst_family_bin": "serine_hydrolase",
        },
        "constraints": {"ph_min": 6.5, "ph_max": 8.0},
    }
    ctx.data["module2_active_site_refinement"] = {
        "module3_handoff": {"best_variant": {"variant_id": "V2", "label": "Clamp"}},
        "variant_set": [{"variant_id": "V0", "label": "Baseline"}],
    }
    ctx.data["shared_io"] = {
        "input": {"condition_profile": {"pH": 7.0, "temperature_C": 30.0}},
        "outputs": {"module2": {}},
    }
    return ctx


def _run_with_results(ctx: PipelineContext, wetlab_results: dict) -> PipelineContext:
    designer = Module3ExperimentDesigner()
    ctx = designer.run(ctx)
    protocol_card = ctx.data["module3_experiment_designer"]["protocol_card"]
    wetlab_results["batch_id"] = protocol_card["batch_id"]
    ctx.data["shared_io"]["input"]["wetlab_results"] = wetlab_results
    ctx = designer.run(ctx)
    return ctx


def test_negative_control_violation():
    ctx = _base_ctx()
    wetlab_results = {
        "batch_id": "placeholder",
        "arms": [
            {
                "arm_id": "A1",
                "observations": [
                    {"metric": "conversion", "value": 0.3},
                    {"metric": "conversion", "value": 0.32},
                ],
            },
            {
                "arm_id": "A4",
                "observations": [
                    {"metric": "conversion", "value": 0.2},
                    {"metric": "conversion", "value": 0.2},
                ],
            },
        ],
        "global_controls": {"blank_ok": True},
    }
    ctx = _run_with_results(ctx, wetlab_results)
    module3 = ctx.data["module3_experiment_designer"]
    assert module3["qc_result"]["status"] == "FAIL"
    assert module3["learning_update"]["status"] == "REJECTED_CONTROL_VIOLATION"


def test_baseline_high_variance():
    ctx = _base_ctx()
    wetlab_results = {
        "batch_id": "placeholder",
        "arms": [
            {
                "arm_id": "A1",
                "observations": [
                    {"metric": "conversion", "value": 0.0},
                    {"metric": "conversion", "value": 0.6},
                ],
            },
            {
                "arm_id": "A4",
                "observations": [
                    {"metric": "conversion", "value": 0.0},
                    {"metric": "conversion", "value": 0.0},
                ],
            },
        ],
        "global_controls": {"blank_ok": True},
    }
    ctx = _run_with_results(ctx, wetlab_results)
    module3 = ctx.data["module3_experiment_designer"]
    assert module3["qc_result"]["status"] == "FAIL"
    assert module3["learning_update"]["status"] == "REJECTED_HIGH_VARIANCE"


def test_sane_data_applies_router_update(tmp_path):
    ctx = _base_ctx()
    router = BayesianDAGRouter(state_path=tmp_path / "router_state.json")
    condition_bin = router._condition_bin(ConditionProfile(pH=7.0, temperature_C=30.0))
    ctx.data["job_card"]["matched_bins"]["condition_bin"] = condition_bin
    ctx.bayes_router = router

    wetlab_results = {
        "batch_id": "placeholder",
        "arms": [
            {
                "arm_id": "A1",
                "observations": [
                    {"metric": "conversion", "value": 0.3},
                    {"metric": "conversion", "value": 0.32},
                ],
            },
            {
                "arm_id": "A2",
                "observations": [
                    {"metric": "conversion", "value": 0.42},
                    {"metric": "conversion", "value": 0.4},
                ],
            },
            {
                "arm_id": "A3",
                "observations": [
                    {"metric": "conversion", "value": 0.25},
                    {"metric": "conversion", "value": 0.27},
                ],
            },
            {
                "arm_id": "A4",
                "observations": [
                    {"metric": "conversion", "value": 0.0},
                    {"metric": "conversion", "value": 0.0},
                ],
            },
            {
                "arm_id": "A5",
                "observations": [
                    {"metric": "conversion", "value": 0.15},
                    {"metric": "conversion", "value": 0.16},
                ],
            },
        ],
        "global_controls": {"blank_ok": True},
    }
    route = ctx.data["job_card"]["chosen_route"]
    matched_bins = ctx.data["job_card"]["matched_bins"]
    before_alpha, before_beta = router._bucket_alpha_beta(
        route,
        matched_bins["condition_bin"],
        matched_bins["substrate_bin"],
        matched_bins["catalyst_family_bin"],
    )

    ctx = _run_with_results(ctx, wetlab_results)
    module3 = ctx.data["module3_experiment_designer"]
    assert module3["qc_result"]["status"] == "PASS"
    assert module3["learning_update"]["status"] == "APPLIED"

    after_alpha, after_beta = router._bucket_alpha_beta(
        route,
        matched_bins["condition_bin"],
        matched_bins["substrate_bin"],
        matched_bins["catalyst_family_bin"],
    )
    assert (after_alpha, after_beta) != (before_alpha, before_beta)
