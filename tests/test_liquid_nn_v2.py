from __future__ import annotations

import pytest

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover - optional dependency
    Chem = None
    AllChem = None

from enzyme_software.liquid_nn_v2 import ModelConfig, smiles_to_graph
from enzyme_software.liquid_nn_v2.data.training_drugs import TRAINING_DRUGS, TRAINING_DRUG_COUNTS
from enzyme_software.liquid_nn_v2.data.smarts_patterns import FUNCTIONAL_GROUP_SMARTS
from enzyme_software.liquid_nn_v2.features.group_detector import detect_functional_groups
from enzyme_software.liquid_nn_v2.training.metrics import analyze_tau


def test_feature_extraction_shape():
    if Chem is None:
        pytest.skip("RDKit not available")
    graph = smiles_to_graph("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")
    assert graph.x.shape[1] == 140
    assert graph.edge_index.shape[0] == 2
    assert len(graph.tau_init) == graph.num_atoms


def test_graph_builder_emits_atom_3d_features_when_structure_present():
    if Chem is None or AllChem is None:
        pytest.skip("RDKit not available")
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        pytest.skip("3D embedding failed")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    structure_mol = Chem.RemoveHs(mol)
    graph = smiles_to_graph("CCO", structure_mol=structure_mol)
    assert graph.atom_3d_features is not None
    assert tuple(graph.atom_3d_features.shape) == (graph.num_atoms, 8)


def test_group_detection():
    if Chem is None:
        pytest.skip("RDKit not available")
    mol = Chem.MolFromSmiles("CC(=O)NC1=CC=CC=C1")
    groups = detect_functional_groups(mol)
    assert len(groups["amide"]) > 0
    assert len(groups["aromatic_ring"]) > 0
    assert len(groups) == len(FUNCTIONAL_GROUP_SMARTS)


def test_training_drug_dataset_shape_and_parseability():
    if Chem is None:
        pytest.skip("RDKit not available")
    assert len(TRAINING_DRUGS) == 30
    assert TRAINING_DRUG_COUNTS == {
        "CYP1A2": 6,
        "CYP2C9": 6,
        "CYP2C19": 6,
        "CYP2D6": 6,
        "CYP3A4": 6,
    }
    for entry in TRAINING_DRUGS:
        mol = Chem.MolFromSmiles(str(entry["smiles"]))
        assert mol is not None, entry["name"]
        for atom_idx in entry["site_atom_indices"]:
            assert 0 <= int(atom_idx) < mol.GetNumAtoms(), entry["name"]


def test_tau_analysis_numpy_path():
    summary = analyze_tau([
        [0.2, 0.4, 0.8],
        [0.25, 0.45, 0.85],
    ], [0.2, 0.4, 0.8], ["benzylic", "secondary_CH", "aryl"])
    assert "tau_by_class" in summary
    assert summary["tau_by_class"]["benzylic"] < summary["tau_by_class"]["aryl"]


def test_model_forward_or_skip():
    pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.model.model import BaselineLiquidMetabolismPredictor, LiquidMetabolismNetV2
    from enzyme_software.liquid_nn_v2.training.utils import create_dummy_batch

    model = LiquidMetabolismNetV2(ModelConfig.baseline())
    baseline_model = BaselineLiquidMetabolismPredictor(ModelConfig.baseline())
    batch = create_dummy_batch(num_molecules=2, num_atoms=20)
    for net in (model, baseline_model):
        outputs = net(batch)
        assert "site_logits" in outputs
        assert "atom_logits" in outputs
        assert "site_scores" in outputs
        assert "cyp_logits" in outputs
        assert "group_embeddings" in outputs
        assert "attention_weights" in outputs
        assert tuple(outputs["site_logits"].shape) == (20, 1)
        assert tuple(outputs["site_scores"].shape) == (20, 1)
        assert tuple(outputs["cyp_logits"].shape) == (2, 5)
        assert not outputs["site_logits"].isnan().any()
        assert all((tau > 0).all() for tau in outputs["tau_history"])
        assert outputs["diagnostics"]["model_variant"] == "baseline"


def test_model_forward_with_manual_engine_and_3d_or_skip():
    pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
    from enzyme_software.liquid_nn_v2.training.utils import create_dummy_batch

    model = LiquidMetabolismNetV2(ModelConfig())
    batch = create_dummy_batch(num_molecules=3, num_atoms=24, include_manual_engine=True, include_3d=True)
    outputs = model(batch)
    assert tuple(outputs["site_logits"].shape) == (24, 1)
    assert tuple(outputs["cyp_logits"].shape) == (3, 5)
    assert outputs["residual_stats"]["som"]["residual_abs_mean"] >= 0.0
    assert outputs["diagnostics"]["steric"]["steric_features_present"] == 1.0


def test_advanced_model_forward_with_all_modules_or_skip():
    pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.model.model import AdvancedLiquidMetabolismPredictor, LiquidMetabolismNetV2
    from enzyme_software.liquid_nn_v2.training.utils import create_dummy_batch

    config = ModelConfig.full_advanced()
    batch = create_dummy_batch(num_molecules=3, num_atoms=24, include_manual_engine=True, include_3d=True)
    for net in (LiquidMetabolismNetV2(config), AdvancedLiquidMetabolismPredictor(config)):
        outputs = net(batch)
        assert tuple(outputs["site_logits"].shape) == (24, 1)
        assert tuple(outputs["cyp_logits"].shape) == (3, 5)
        assert outputs["diagnostics"]["model_variant"] == "advanced"
        assert outputs["energy_outputs"]["node_energy"].shape == (24, 1)
        assert outputs["tunneling_outputs"]["barrier"].shape == (24, 1)
        assert outputs["tunneling_outputs"]["tunnel_prob"].min().item() >= 0.0
        assert outputs["tunneling_outputs"]["tunnel_prob"].max().item() <= 1.0
        assert outputs["graph_tunneling_outputs"]["message"].shape == outputs["site_logits"].expand(-1, config.som_branch_dim).shape
        assert outputs["phase_outputs"]["phase"].shape == (24, config.shared_hidden_dim)
        assert outputs["deliberation_outputs"]["stats"]["num_steps"] == float(config.num_deliberation_steps)
        assert not outputs["site_logits"].isnan().any()
        assert not outputs["cyp_logits"].isnan().any()


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig.light_advanced(use_phase_augmented_state=False, use_higher_order_coupling=False),
        ModelConfig.light_advanced(use_graph_tunneling=False),
        ModelConfig.baseline(),
    ],
)
def test_model_variants_run_cleanly(config):
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.model.model import LiquidMetabolismNetV2
    from enzyme_software.liquid_nn_v2.training.utils import create_dummy_batch

    batch = create_dummy_batch(num_molecules=2, num_atoms=16, include_3d=True)
    outputs = LiquidMetabolismNetV2(config)(batch)
    assert tuple(outputs["site_logits"].shape) == (16, 1)
    assert tuple(outputs["cyp_logits"].shape) == (2, 5)
    assert all(torch.isfinite(tau).all() for tau in outputs["tau_history"])
    if config.model_variant == "advanced" and config.use_tunneling_module:
        probs = outputs["tunneling_outputs"]["tunnel_prob"]
        assert bool(torch.isfinite(probs).all())
        assert float(probs.min().item()) >= 0.0
        assert float(probs.max().item()) <= 1.0


def test_hierarchical_pooling_handles_empty_groups():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.model.pooling import ChemistryHierarchicalPooling

    pooling = ChemistryHierarchicalPooling(atom_dim=16, hidden_dim=8)
    atom_embeddings = torch.rand(2, 4, 16)
    atom_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    group_membership = torch.zeros(2, 4, len(FUNCTIONAL_GROUP_SMARTS))
    outputs = pooling(atom_embeddings, group_membership, atom_mask)
    assert outputs["group_embeddings"].shape == (2, len(FUNCTIONAL_GROUP_SMARTS), 16)
    assert outputs["molecule_embedding"].shape == (2, 16)
    assert not outputs["molecule_embedding"].isnan().any()


def test_loss_or_skip():
    torch = pytest.importorskip("torch")
    from enzyme_software.liquid_nn_v2.data.cyp_classes import ALL_CYP_CLASSES
    from enzyme_software.liquid_nn_v2.training.loss import AdaptiveLossV2

    loss_fn = AdaptiveLossV2()
    site_logits = torch.rand(20, 1)
    cyp_logits = torch.rand(2, len(ALL_CYP_CLASSES))
    site_labels = torch.randint(0, 2, (20, 1)).float()
    cyp_labels = torch.randint(0, len(ALL_CYP_CLASSES), (2,))
    batch = torch.repeat_interleave(torch.arange(2), 10)
    energy_outputs = {"node_energy": torch.rand(20, 1)}
    deliberation_outputs = {
        "site_logits": [torch.rand(20, 1), torch.rand(20, 1)],
        "cyp_logits": [torch.rand(2, len(ALL_CYP_CLASSES)), torch.rand(2, len(ALL_CYP_CLASSES))],
    }
    loss, stats = loss_fn(site_logits, cyp_logits, site_labels, cyp_labels, batch, energy_outputs=energy_outputs, deliberation_outputs=deliberation_outputs)
    assert loss.item() > 0
    assert "site_loss" in stats
    assert "ranking_loss" in stats
    assert "energy_loss" in stats
    assert "deliberation_loss" in stats
