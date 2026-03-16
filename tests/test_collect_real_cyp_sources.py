from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from scripts.collect_data.add_som_labels import add_som_labels
from scripts.collect_data.merge_all_sources import get_validated_drugs
from scripts.collect_data.scrape_supercyp import parse_autocomplete_options, parse_interaction_table


def test_parse_supercyp_autocomplete_options():
    html = '<tr><td><select><option value="Ibuprofen">Ibuprofen<option value="Naproxen">Naproxen</select></td></tr>'
    assert parse_autocomplete_options(html) == ["Ibuprofen", "Naproxen"]


def test_parse_supercyp_interaction_table():
    html = '''
    <table>
      <tr><td>Name</td><td>1A2 Cytochrome 1A2</td><td>2C9 Cytochrome 2C9</td><td>2D6 Cytochrome 2D6</td><td>3A4 Cytochrome 3A4</td></tr>
      <tr><td>Ibuprofen Ibuprofen</td><td></td><td>S 12345 Inh 67890</td><td></td><td>Ind 555</td></tr>
    </table>
    '''
    parsed = parse_interaction_table(html, 'Ibuprofen')
    assert parsed is not None
    assert parsed['cyp_substrate'] == ['CYP2C9']
    assert parsed['cyp_inhibitor'] == ['CYP2C9']
    assert parsed['cyp_inducer'] == ['CYP3A4']


def test_get_validated_drugs_shape():
    drugs = get_validated_drugs()
    assert len(drugs) >= 14
    assert all('smiles' in drug for drug in drugs)
    assert all(drug['confidence'] == 'validated' for drug in drugs)


def test_add_som_labels_enriches_missing_sites(tmp_path: Path):
    input_path = tmp_path / 'input.json'
    output_path = tmp_path / 'output.json'
    payload = {
        'metadata': {'total_drugs': 2},
        'drugs': [
            {
                'name': 'Ibuprofen',
                'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
                'primary_cyp': 'CYP2C9',
                'confidence': 'validated',
            },
            {
                'name': 'Codeine',
                'smiles': 'COC1=CC2=C(C=C1)C3C4C=CC(C2C3O)N(C)CC4',
                'primary_cyp': 'CYP2D6',
                'site_atoms': [0],
                'som_label_source': 'validated',
                'confidence': 'validated',
            },
        ],
    }
    input_path.write_text(json.dumps(payload))
    output = add_som_labels(str(input_path), str(output_path))
    assert output['metadata']['som_labeled'] == 1
    assert output['metadata']['som_validated'] == 1
    first = output['drugs'][0]
    assert first['site_atoms']
    assert first['som_label_source'] == 'bde_predicted'
    second = output['drugs'][1]
    assert second['site_atoms'] == [0]
    assert second['som_label_source'] == 'validated'
