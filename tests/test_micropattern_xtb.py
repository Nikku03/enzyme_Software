from __future__ import annotations

import json
from pathlib import Path

import pytest

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    pytest.importorskip("rdkit")
    import torch

    from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
    from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.config import MicroPatternXTBConfig
    from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.model import (
        MicroPatternXTBHybridModel,
        load_base_hybrid_checkpoint,
    )
    from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.metrics import compute_reranker_metrics
    from enzyme_software.liquid_nn_v2.experiments.micropattern_xtb.reranker import MicroPatternReranker
    from enzyme_software.liquid_nn_v2.features.graph_builder import smiles_to_graph
    from enzyme_software.liquid_nn_v2.features.micropattern_features import build_candidate_local_descriptor, neighbors_within_radius
    from enzyme_software.liquid_nn_v2.features.xtb_features import (
        XTB_FEATURE_DIM,
        attach_xtb_features_to_graph,
        load_or_compute_xtb_features,
    )
    from enzyme_software.liquid_nn_v2.training.utils import collate_molecule_graphs


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_xtb_cache_missing_is_safe(tmp_path: Path):
    payload = load_or_compute_xtb_features("CCO", cache_dir=tmp_path, compute_if_missing=False)
    assert payload["xtb_valid"] is False
    assert payload["status"] == "missing"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_attach_xtb_features_to_graph(tmp_path: Path):
    graph = smiles_to_graph("CCO", site_atoms=[0])
    graph = attach_xtb_features_to_graph(graph, cache_dir=tmp_path, compute_if_missing=False)
    assert graph.xtb_atom_features.shape == (graph.num_atoms, XTB_FEATURE_DIM)
    assert graph.xtb_atom_valid_mask.shape == (graph.num_atoms, 1)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_micropattern_local_descriptor_shapes():
    graph = smiles_to_graph("CCO", site_atoms=[0])
    adj = neighbors_within_radius(graph.edge_index, graph.num_atoms, 0, 2)
    assert 0 in adj and 1 in adj and 2 in adj
    ring = aromatic = torch.zeros(graph.num_atoms).numpy()
    desc = build_candidate_local_descriptor(
        edge_index=graph.edge_index,
        num_atoms=graph.num_atoms,
        center_idx=0,
        radius=2,
        atom_embeddings=graph.x,
        manual_features=None,
        xtb_features=None,
        ring_flags=ring,
        aromatic_flags=aromatic,
        edge_attr=graph.edge_attr,
    )
    assert desc.ndim == 1
    assert desc.shape[0] > graph.x.shape[1]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_warm_start_checkpoint_loading_and_isolation(tmp_path: Path):
    base = HybridLNNModel(
        LiquidMetabolismNetV2(
            ModelConfig.light_advanced(
                use_manual_engine_priors=True,
                use_3d_branch=True,
                return_intermediate_stats=True,
            )
        )
    )
    checkpoint_path = tmp_path / "hybrid_lnn_latest.pt"
    torch.save({"model_state_dict": base.state_dict(), "config": {"base_model": base.base_lnn.config.__dict__}}, checkpoint_path)
    loaded = load_base_hybrid_checkpoint(checkpoint_path, device=torch.device("cpu"))
    first_key = next(iter(base.state_dict().keys()))
    assert torch.equal(base.state_dict()[first_key], loaded.state_dict()[first_key])
    exp_dir = tmp_path / "micropattern_xtb"
    exp_dir.mkdir()
    assert checkpoint_path.exists()
    assert not any(exp_dir.iterdir())


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_reranker_forward_and_metrics():
    reranker = MicroPatternReranker(hidden_dim=32)
    features = torch.randn(10, 24)
    xtb = torch.randn(10, 6)
    base_logits = torch.randn(10, 1)
    refined, payload = reranker(features, xtb, base_logits)
    assert refined.shape == (10, 1)
    assert torch.isfinite(refined).all()
    out = {
        "candidate_valid": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool),
        "candidate_positive": torch.tensor([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool),
        "reranked_candidate_scores": torch.tensor([[0.1, 0.4, 0.2, -1e9, -1e9], [0.3, 0.2, -1e9, -1e9, -1e9]]),
        "base_candidate_scores": torch.tensor([[0.2, 0.3, 0.1, -1e9, -1e9], [0.1, 0.3, -1e9, -1e9, -1e9]]),
    }
    metrics = compute_reranker_metrics(out, out)
    assert metrics["reranked_top1"] >= 0.5
    assert "base_top1" in metrics


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_experiment_model_forward():
    base = HybridLNNModel(
        LiquidMetabolismNetV2(
            ModelConfig.light_advanced(
                use_manual_engine_priors=True,
                use_3d_branch=True,
                return_intermediate_stats=True,
            )
        )
    )
    config = MicroPatternXTBConfig.default()
    model = MicroPatternXTBHybridModel(base, config)
    g1 = attach_xtb_features_to_graph(smiles_to_graph("CCO", site_atoms=[0]), cache_dir=Path("/tmp/micropattern_xtb_test"), compute_if_missing=False)
    g1.manual_engine_atom_features = torch.zeros(g1.num_atoms, 32).numpy()
    g1.manual_engine_mol_features = torch.zeros(1, 8).numpy()
    g1.manual_engine_atom_prior_logits = torch.zeros(g1.num_atoms, 1).numpy()
    g1.manual_engine_cyp_prior_logits = torch.zeros(1, len(config.model_config.cyp_names)).numpy()
    g1.manual_engine_route_prior = torch.full((1, len(config.model_config.cyp_names)), 1.0 / len(config.model_config.cyp_names)).numpy()
    batch = collate_molecule_graphs([g1])
    outputs = model(batch)
    assert "reranked_site_logits" in outputs
    assert outputs["reranked_site_logits"].shape == outputs["base_site_logits"].shape
