from __future__ import annotations

import pytest


def test_light_advanced_defaults_are_safe():
    from enzyme_software.liquid_nn_v2 import ModelConfig

    config = ModelConfig.light_advanced()
    assert config.use_physics_residual is True
    assert config.use_energy_module is False
    assert config.use_graph_tunneling is False
    assert config.use_deliberation_loop is False
    assert config.num_deliberation_steps == 0


def test_deliberation_loop_stays_finite():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.model.advanced_modules import DeliberationLoop

    loop = DeliberationLoop(
        atom_dim=32,
        mol_dim=24,
        num_cyp_classes=5,
        hidden_dim=16,
        num_steps=3,
        step_scale=0.05,
        max_state_norm=4.0,
    )
    atom_hidden = torch.randn(18, 32) * 5.0
    mol_hidden = torch.randn(3, 24) * 5.0
    batch = torch.repeat_interleave(torch.arange(3), 6)
    payload = loop(
        atom_hidden,
        mol_hidden,
        batch,
        node_energy=torch.randn(18, 1),
        tunnel_prob=torch.sigmoid(torch.randn(18, 1)),
    )
    assert bool(torch.isfinite(payload["atom_hidden"]).all())
    assert bool(torch.isfinite(payload["mol_hidden"]).all())
    assert payload["stats"]["num_steps"] == 3.0
    assert payload["stats"]["atom_hidden_norm_mean"] <= 4.5
    assert payload["stats"]["mol_hidden_norm_mean"] <= 4.5


def test_graph_tunneling_is_sparse_and_finite():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.model.advanced_modules import GraphTunneling

    module = GraphTunneling(hidden_dim=32, projection_dim=16, max_edges_per_node=3)
    hidden = torch.randn(20, 32)
    batch = torch.repeat_interleave(torch.arange(4), 5)
    tunnel_prob = torch.sigmoid(torch.randn(20, 1))
    payload = module(hidden, batch, tunnel_prob=tunnel_prob)
    assert bool(torch.isfinite(payload["message"]).all())
    assert bool(torch.isfinite(payload["edge_prob"]).all())
    assert payload["edge_index"].shape[1] <= hidden.shape[0] * 3
    assert payload["stats"]["tunneling_edge_count"] <= hidden.shape[0] * 3


def test_energy_loss_is_robust_to_extreme_values():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.training.loss import AdaptiveLossV2

    loss_fn = AdaptiveLossV2(energy_loss_weight=1.0, energy_loss_clip=1.5)
    site_logits = torch.randn(12, 1)
    cyp_logits = torch.randn(2, 5)
    site_labels = torch.randint(0, 2, (12, 1)).float()
    cyp_labels = torch.randint(0, 5, (2,))
    batch = torch.repeat_interleave(torch.arange(2), 6)
    huge_energy = torch.tensor([[1.0e6], [-1.0e6]] * 6, dtype=torch.float32)
    loss, stats = loss_fn(
        site_logits,
        cyp_logits,
        site_labels,
        cyp_labels,
        batch,
        energy_outputs={"node_energy": huge_energy},
    )
    assert bool(torch.isfinite(loss))
    assert stats["energy_loss"] <= 2.0


def test_trainer_finite_guard_raises_cleanly():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2 import ModelConfig, TrainingConfig
    from enzyme_software.liquid_nn_v2.training.trainer import Trainer
    from enzyme_software.liquid_nn_v2.training.utils import create_dummy_batch

    class BadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(1))
            self.config = ModelConfig.baseline(return_intermediate_stats=True, use_3d_branch=False)

        def forward(self, batch):
            num_atoms = batch["site_labels"].shape[0]
            num_mols = batch["cyp_labels"].shape[0]
            nan_atom = self.bias * torch.full((num_atoms, 1), float("nan"))
            nan_mol = self.bias * torch.full((num_mols, 5), float("nan"))
            return {
                "site_logits": nan_atom,
                "site_scores": nan_atom,
                "cyp_logits": nan_mol,
                "tau_history": [],
                "energy_outputs": {},
                "deliberation_outputs": {},
                "tau_stats": {},
                "diagnostics": {},
            }

    trainer = Trainer(
        model=BadModel(),
        config=TrainingConfig(batch_size=2, max_grad_norm=1.0),
        device=torch.device("cpu"),
    )
    batch = create_dummy_batch(num_molecules=2, num_atoms=12)
    with pytest.raises(FloatingPointError):
        trainer.train_epoch(batch)
