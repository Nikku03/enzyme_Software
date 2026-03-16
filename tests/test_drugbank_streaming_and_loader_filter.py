from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from scripts.build_dataset.parse_drugbank import parse_drugbank_xml
from enzyme_software.liquid_nn_v2.data.dataset_loader import CYPMetabolismDataset


def test_parse_drugbank_xml_streaming(tmp_path: Path):
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
    <drugbank xmlns="http://www.drugbank.ca">
      <drug>
        <drugbank-id primary="true">DBTEST1</drugbank-id>
        <name>Test Drug</name>
        <calculated-properties>
          <property><kind>SMILES</kind><value>CCO</value></property>
        </calculated-properties>
        <metabolism>CYP2D6 catalyzes hydroxylation.</metabolism>
      </drug>
      <drug>
        <drugbank-id primary="true">DBTEST2</drugbank-id>
        <name>No CYP</name>
        <calculated-properties>
          <property><kind>SMILES</kind><value>CCN</value></property>
        </calculated-properties>
        <metabolism>Renal elimination.</metabolism>
      </drug>
    </drugbank>
    '''
    path = tmp_path / 'drugbank.xml'
    path.write_text(xml)
    rows = parse_drugbank_xml(path)
    assert len(rows) == 1
    assert rows[0]['drugbank_id'] == 'DBTEST1'
    assert rows[0]['enzymes'] == ['CYP2D6']


def test_dataset_loader_filters_to_model_cyps(tmp_path: Path):
    data = {
        'drugs': [
            {'name': 'A', 'smiles': 'CCO', 'primary_cyp': 'CYP2D6', 'som': [{'atom_idx': 0, 'bond_class': 'primary_CH'}], 'confidence': 'validated'},
            {'name': 'B', 'smiles': 'CCN', 'primary_cyp': 'CYP2A6', 'som': [{'atom_idx': 0, 'bond_class': 'primary_CH'}], 'confidence': 'validated'},
        ]
    }
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps(data))
    ds = CYPMetabolismDataset(str(path), split='train', train_ratio=1.0, val_ratio=0.0)
    assert len(ds.drugs) == 1
    assert ds.drugs[0]['primary_cyp'] == 'CYP2D6'
