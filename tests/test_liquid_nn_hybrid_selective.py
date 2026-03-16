from __future__ import annotations

import pytest


def test_hybrid_selective_config_is_conservative():
    from enzyme_software.liquid_nn_v2 import ModelConfig

    config = ModelConfig.hybrid_selective()
    assert config.model_variant == "hybrid_selective"
    assert config.use_local_tunneling_bias is True
    assert config.use_output_refinement is True
    assert config.use_graph_tunneling is False
    assert config.use_energy_module is False
    assert config.use_deliberation_loop is False
    assert config.num_deliberation_steps == 0


def test_hybrid_selective_forward_outputs_and_diagnostics():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2 import ModelConfig
    from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2, SelectiveHybridLiquidMetabolismPredictor
    from enzyme_software.liquid_nn_v2.training.utils import create_dummy_batch

    config = ModelConfig.hybrid_selective(use_3d_branch=True, return_intermediate_stats=True)
    batch = create_dummy_batch(num_molecules=3, num_atoms=24, include_3d=True)
    for model in (LiquidMetabolismNetV2(config), SelectiveHybridLiquidMetabolismPredictor(config)):
        outputs = model(batch)
        assert tuple(outputs["site_logits"].shape) == (24, 1)
        assert tuple(outputs["cyp_logits"].shape) == (3, 5)
        assert outputs["diagnostics"]["model_variant"] == "hybrid_selective"
        hybrid_stats = outputs["diagnostics"]["hybrid_selective"]
        assert "tunnel_bias_mean" in hybrid_stats
        assert "refine_gate_mean" in hybrid_stats
        assert outputs["tunneling_outputs"]["tunnel_prob"].shape == (24, 1)
        assert outputs["hybrid_outputs"]["tunnel_bias"].shape == (24, 1)
        assert outputs["hybrid_outputs"]["refine_delta"].shape == (24, 1)
        assert outputs["hybrid_outputs"]["refine_gate"].shape == (24, 1)
        assert bool(torch.isfinite(outputs["site_logits"]).all())


def test_hybrid_selective_backward_pass_runs():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2 import ModelConfig, TrainingConfig
    from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
    from enzyme_software.liquid_nn_v2.training.trainer import Trainer
    from enzyme_software.liquid_nn_v2.training.utils import create_dummy_batch

    model = LiquidMetabolismNetV2(ModelConfig.hybrid_selective(use_3d_branch=True))
    trainer = Trainer(model=model, config=TrainingConfig(batch_size=2, max_grad_norm=1.0), device=torch.device("cpu"))
    stats = trainer.train_epoch(create_dummy_batch(num_molecules=2, num_atoms=12, include_3d=True))
    assert stats["total_loss"] > 0.0
    assert "tunnel_bias_mean" in stats
    assert "refine_gate_mean" in stats


@pytest.mark.parametrize(
    "config",
    [
        pytest.param("baseline", id="baseline"),
        pytest.param("light_advanced", id="light_advanced"),
        pytest.param("full_advanced", id="full_advanced"),
        pytest.param("hybrid_selective", id="hybrid_selective"),
    ],
)
def test_all_config_presets_still_run(config):
    pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2 import ModelConfig
    from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
    from enzyme_software.liquid_nn_v2.training.utils import create_dummy_batch

    factory = getattr(ModelConfig, config)
    model = LiquidMetabolismNetV2(factory(use_3d_branch=True, return_intermediate_stats=True))
    outputs = model(create_dummy_batch(num_molecules=2, num_atoms=16, include_3d=True))
    assert tuple(outputs["site_logits"].shape) == (16, 1)
    assert tuple(outputs["cyp_logits"].shape) == (2, 5)
