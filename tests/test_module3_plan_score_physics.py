from __future__ import annotations

from enzyme_software.context import PipelineContext
from enzyme_software.modules.module3_experiment_designer import _plan_score_physics


def _base_ctx(k_variant: float) -> PipelineContext:
    ctx = PipelineContext(smiles="CCO", target_bond="C-O")
    ctx.data["module2_active_site_refinement"] = {
        "module2_physics_audit": {"k_variant_s_inv": k_variant}
    }
    ctx.data["shared_io"] = {"input": {}, "outputs": {}}
    return ctx


def test_plan_score_physics_increases_with_signal():
    protocol = {"controls": {"negative_control_arm_id": "A4"}, "arms": []}
    ctx_low = _base_ctx(1e-6)
    score_low, _ = _plan_score_physics(ctx_low, protocol)
    ctx_high = _base_ctx(1e-4)
    score_high, _ = _plan_score_physics(ctx_high, protocol)
    assert score_high > score_low


def test_plan_score_physics_missing_physics_returns_zero():
    protocol = {"controls": {"negative_control_arm_id": "A4"}, "arms": []}
    ctx = PipelineContext(smiles="CCO", target_bond="C-O")
    ctx.data["shared_io"] = {"input": {}, "outputs": {}}
    score, audit = _plan_score_physics(ctx, protocol)
    assert score == 0.0
    assert audit.get("plan_phys", 0.0) == 0.0
