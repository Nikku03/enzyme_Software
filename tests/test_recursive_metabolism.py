from __future__ import annotations

from pathlib import Path

import pytest

from enzyme_software.liquid_nn_v2._compat import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch

    pytest.importorskip("rdkit")

    from enzyme_software.liquid_nn_v2 import HybridLNNModel, LiquidMetabolismNetV2, ModelConfig
    from enzyme_software.recursive_metabolism import (
        MetabolismSimulator,
        PathwayGenerator,
        RecursiveMetabolismConfig,
        RecursiveMetabolismDataset,
        RecursiveMetabolismModel,
        RecursiveMetabolismTrainer,
        collate_recursive_batch,
        load_base_hybrid_checkpoint,
    )
    from enzyme_software.recursive_metabolism.utils import initialized_state_dict


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_metabolism_simulator_hydroxylation():
    simulator = MetabolismSimulator()
    result = simulator.metabolize("CCO", 0)
    assert isinstance(result.success, bool)
    assert result.parent_smiles


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_pathway_generator_produces_steps():
    generator = PathwayGenerator(max_steps=3, min_heavy_atoms=2)
    pathway = generator.generate(smiles="CCN", drug_name="demo", ground_truth_sites=[0])
    assert pathway.drug_name == "demo"
    assert pathway.total_steps >= 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_recursive_dataset_and_collate(tmp_path: Path):
    pathways = [
        {
            "drug_smiles": "CCN",
            "drug_name": "demo",
            "steps": [
                {
                    "step_number": 0,
                    "parent_smiles": "CCN",
                    "metabolite_smiles": "CCNO",
                    "site_atom_idx": 0,
                    "metabolism_type": "hydroxylation",
                    "supervision_source": "ground_truth",
                    "source_weight": 1.0,
                }
            ],
            "total_steps": 1,
        }
    ]
    dataset = RecursiveMetabolismDataset(
        pathways,
        structure_sdf=None,
        include_manual_engine_features=False,
        include_xtb_features=False,
        drop_failed=True,
    )
    sample = dataset[0]
    assert sample["graph"] is not None
    batch = collate_recursive_batch([sample])
    assert batch is not None
    assert int(batch["graph_step_numbers"][0].item()) == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_recursive_dataset_ground_truth_only_filter():
    pathways = [
        {
            "drug_smiles": "CCN",
            "drug_name": "demo",
            "steps": [
                {
                    "step_number": 0,
                    "parent_smiles": "CCN",
                    "metabolite_smiles": "CCNO",
                    "site_atom_idx": 0,
                    "metabolism_type": "hydroxylation",
                    "source": "ground_truth",
                    "source_weight": 1.0,
                },
                {
                    "step_number": 1,
                    "parent_smiles": "CCNO",
                    "metabolite_smiles": "CCNOO",
                    "site_atom_idx": 1,
                    "metabolism_type": "hydroxylation",
                    "source": "heuristic",
                    "source_weight": 0.35,
                },
            ],
            "total_steps": 2,
        }
    ]
    dataset = RecursiveMetabolismDataset(
        pathways,
        structure_sdf=None,
        include_manual_engine_features=False,
        include_xtb_features=False,
        drop_failed=True,
        ground_truth_only=True,
    )
    assert len(dataset) == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_recursive_warm_start_loading(tmp_path: Path):
    base = HybridLNNModel(
        LiquidMetabolismNetV2(
            ModelConfig.light_advanced(
                use_manual_engine_priors=True,
                use_3d_branch=True,
                return_intermediate_stats=True,
            )
        )
    )
    checkpoint = tmp_path / "hybrid_lnn_latest.pt"
    torch.save(
        {
            "model_state_dict": initialized_state_dict(base),
            "config": {"base_model": base.base_lnn.config.__dict__},
        },
        checkpoint,
    )
    loaded = load_base_hybrid_checkpoint(checkpoint, device=torch.device("cpu"))
    first_key = next(iter(loaded.state_dict().keys()))
    assert torch.equal(loaded.state_dict()[first_key], base.state_dict()[first_key])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_recursive_model_forward(tmp_path: Path):
    base = HybridLNNModel(
        LiquidMetabolismNetV2(
            ModelConfig.light_advanced(
                use_manual_engine_priors=True,
                use_3d_branch=True,
                return_intermediate_stats=True,
            )
        )
    )
    config = RecursiveMetabolismConfig.default(
        freeze_base_model=True,
        include_xtb_features=False,
        include_manual_engine_features=False,
    )
    model = RecursiveMetabolismModel(base, config)
    pathways = [
        {
            "drug_smiles": "CCN",
            "drug_name": "demo",
            "steps": [
                {
                    "step_number": 0,
                    "parent_smiles": "CCN",
                    "metabolite_smiles": "CCNO",
                    "site_atom_idx": 0,
                    "metabolism_type": "hydroxylation",
                    "supervision_source": "ground_truth",
                    "source_weight": 1.0,
                }
            ],
            "total_steps": 1,
        }
    ]
    dataset = RecursiveMetabolismDataset(
        pathways,
        structure_sdf=None,
        include_manual_engine_features=False,
        include_xtb_features=False,
        drop_failed=True,
    )
    batch = collate_recursive_batch([dataset[0]])
    outputs = model(batch)
    assert "recursive_site_logits" in outputs
    assert outputs["recursive_site_logits"].shape == outputs["base_site_logits"].shape
