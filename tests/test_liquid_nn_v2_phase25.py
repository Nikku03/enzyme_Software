from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

from scripts.build_dataset.merge_curated_50 import merge_datasets
from enzyme_software.liquid_nn_v2.data.curated_50_drugs import CURATED_50_COUNTS, CURATED_50_DRUGS


def test_curated_50_dataset_shape_and_parseability():
    assert len(CURATED_50_DRUGS) == 50
    assert CURATED_50_COUNTS == {
        'CYP2D6': 15,
        'CYP2C19': 12,
        'CYP1A2': 8,
        'CYP2C9': 8,
        'CYP3A4': 7,
    }
    for entry in CURATED_50_DRUGS:
        mol = Chem.MolFromSmiles(entry['smiles'])
        assert mol is not None, entry['name']
        for atom_idx in entry['site_atom_indices']:
            assert 0 <= int(atom_idx) < mol.GetNumAtoms(), entry['name']
        assert entry['som'], entry['name']


def test_merge_curated_50_from_empty_dataset(tmp_path):
    existing = {'metadata': {'total_drugs': 0}, 'drugs': []}
    existing_path = tmp_path / 'training_dataset_530.json'
    output_path = tmp_path / 'training_dataset_580.json'
    existing_path.write_text(json.dumps(existing))
    merged = merge_datasets(str(existing_path), str(output_path))
    payload = json.loads(output_path.read_text())
    assert payload['metadata']['added_curated_drugs'] == 50
    assert payload['metadata']['total_drugs'] == 50
    assert len(payload['drugs']) == 50
    assert merged['metadata']['source_distribution']['curated'] == 50


def test_train_phase25_script_smoke(tmp_path):
    dataset = {
        'metadata': {'total_drugs': 3},
        'drugs': [
            {
                'name': 'Ibuprofen',
                'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
                'primary_cyp': 'CYP2C9',
                'cyp': 'CYP2C9',
                'som': [{'atom_idx': 7, 'bond_class': 'benzylic', 'confidence': 0.9}],
                'confidence': 'high',
                'source': 'curated',
            },
            {
                'name': 'Codeine',
                'smiles': 'COc1ccc2C[C@H]3N(C)CC[C@@]45[C@@H](Oc1c24)[C@H](O)C=C[C@@H]35',
                'primary_cyp': 'CYP2D6',
                'cyp': 'CYP2D6',
                'som': [{'atom_idx': 0, 'bond_class': 'alpha_hetero', 'confidence': 0.85}],
                'confidence': 'high',
                'source': 'curated',
            },
            {
                'name': 'Caffeine',
                'smiles': 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
                'primary_cyp': 'CYP1A2',
                'cyp': 'CYP1A2',
                'som': [{'atom_idx': 0, 'bond_class': 'alpha_hetero', 'confidence': 0.8}],
                'confidence': 'high',
                'source': 'curated',
            },
        ],
    }
    dataset_path = tmp_path / 'training_dataset_580.json'
    dataset_path.write_text(json.dumps(dataset))
    output_dir = tmp_path / 'checkpoints'
    env = dict(os.environ)
    env.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(Path.cwd() / 'src') + os.pathsep + str(Path.cwd())
    result = subprocess.run(
        [
            sys.executable,
            'scripts/train_phase2_5.py',
            '--dataset', str(dataset_path),
            '--epochs', '1',
            '--batch-size', '2',
            '--device', 'cpu',
            '--output-dir', str(output_dir),
        ],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'TEST SET EVALUATION' in result.stdout
    assert (output_dir / 'phase2_5_580drugs_latest.pt').exists()
