from __future__ import annotations

import pytest
from unittest.mock import patch


def test_route_prior_projection_matches_requested_classes():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.features.route_prior import route_posteriors_to_cyp_prior

    prior = route_posteriors_to_cyp_prior(
        {"p450": 0.8, "monooxygenase": 0.2},
        cyp_order=["CYP1A2", "CYP2D6", "CYP3A4"],
    )
    assert tuple(prior.shape) == (3,)
    assert bool(torch.isfinite(prior).all())
    assert pytest.approx(float(prior.sum().item()), rel=1e-5) == 1.0


def test_manual_feature_tensorization_is_finite():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.features.manual_engine_features import manual_features_to_tensor

    manual = {
        "candidate_sites": [
            {"heavy_atom_index": 0, "bde_kj_mol": 375.0, "score": 0.9, "bond_class": "ch__benzylic", "radical_stability": 0.7},
            {"heavy_atom_index": 3, "bde_kj_mol": 410.0, "score": 0.4, "bond_class": "ch__secondary", "radical_stability": 0.2},
        ],
        "bond360_profile": {"attack_sites": {"site_a": 0}},
        "mechanism_eligibility": {"p450_oxidation": "ELIGIBLE", "monooxygenase": "SUPPORTED"},
        "route_confidence": 0.8,
        "route_gap": 0.15,
        "ambiguity_flag": False,
        "fallback_used": False,
    }
    tensor = manual_features_to_tensor(manual, num_atoms=6)
    assert tuple(tensor.shape) == (6, 32)
    assert bool(torch.isfinite(tensor).all())
    assert float(tensor[0, 10].item()) > 0.0


def test_hybrid_arbitrator_prefers_agreement():
    from enzyme_software.liquid_nn_v2.arbitration.hybrid_arbitrator import HybridArbitrator

    pred = HybridArbitrator().arbitrate(
        {
            "site_atoms": [4, 2, 1],
            "site_scores": [0.9, 0.6, 0.4],
            "cyp": "CYP3A4",
            "cyp_confidence": 0.8,
        },
        {
            "predicted_sites": [4, 9, 10],
            "route": "p450",
            "route_confidence": 0.7,
            "primary_cyp": "CYP3A4",
        },
    )
    assert pred.site_source == "hybrid"
    assert pred.cyp_source == "hybrid"
    assert pred.agreement is True


def test_hybrid_lnn_wrapper_combines_prior():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.model.hybrid_model import HybridLNNModel

    class FakeBase(torch.nn.Module):
        def forward(self, batch):
            return {
                "site_logits": torch.zeros(4, 1),
                "site_scores": torch.full((4, 1), 0.5),
                "cyp_logits": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
                "diagnostics": {},
            }

    model = HybridLNNModel(FakeBase(), prior_weight_init=0.3)
    batch = {
        "manual_engine_route_prior": torch.tensor([[0.8, 0.1, 0.1]], dtype=torch.float32),
    }
    outputs = model(batch)
    assert "cyp_logits_base" in outputs
    assert "hybrid_manual_prior" in outputs
    assert tuple(outputs["cyp_logits"].shape) == (1, 3)
    assert bool(torch.isfinite(outputs["cyp_logits"]).all())


def test_sanitize_smiles_handles_bad_aromatic_input():
    pytest.importorskip("rdkit")
    from enzyme_software.liquid_nn_v2.utils.smiles_sanitizer import sanitize_smiles

    cleaned, error = sanitize_smiles("cC")
    assert error is None
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


def test_manual_feature_extraction_falls_back_on_unknown_atom_id():
    from enzyme_software.liquid_nn_v2.features.manual_engine_features import extract_module_minus1_features

    with patch(
        "enzyme_software.liquid_nn_v2.features.manual_engine_features._run_manual_engine",
        side_effect=KeyError("Unknown atom_id: fake_uuid_H35"),
    ):
        features = extract_module_minus1_features("CCO", cache_dir=None)

    assert features is not None
    assert features["fallback_used"] is True
    assert features["manual_feature_status"] == "fallback"
    assert len(features["candidate_sites"]) > 0


def test_graph_builder_uses_safe_parser_for_problem_smiles():
    pytest.importorskip("rdkit")
    from enzyme_software.liquid_nn_v2.features.graph_builder import smiles_to_graph

    graph = smiles_to_graph("cC")
    assert graph.num_atoms > 0
    assert graph.x.shape[0] == graph.num_atoms
