from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rdkit import Chem

from enzyme_software.liquid_nn_v2.data.curated_50_drugs import CURATED_50_DRUGS, CURATED_50_COUNTS


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else smiles


def merge_datasets(
    existing_path: str = 'data/training_dataset_530.json',
    output_path: str = 'data/training_dataset_580.json',
):
    existing_payload = json.loads(Path(existing_path).read_text())
    drugs = list(existing_payload.get('drugs', existing_payload))
    existing_smiles = {_canonical_smiles(str(d['smiles'])) for d in drugs}

    print(f'Existing drugs: {len(drugs)}')
    print(f'Curated candidates: {len(CURATED_50_DRUGS)}')

    added = 0
    for drug in CURATED_50_DRUGS:
        canon = _canonical_smiles(str(drug['smiles']))
        if canon in existing_smiles:
            continue
        drugs.append(dict(drug))
        existing_smiles.add(canon)
        added += 1

    print(f'Added: {added} new drugs')
    print(f'Total: {len(drugs)}')

    by_cyp = {}
    by_source = {}
    by_confidence = {}
    for drug in drugs:
        cyp = str(drug.get('primary_cyp') or drug.get('cyp') or 'UNKNOWN')
        by_cyp[cyp] = by_cyp.get(cyp, 0) + 1
        source = str(drug.get('source', 'unknown'))
        by_source[source] = by_source.get(source, 0) + 1
        conf = str(drug.get('confidence', 'unknown'))
        by_confidence[conf] = by_confidence.get(conf, 0) + 1

    output = {
        'metadata': {
            'total_drugs': len(drugs),
            'curated_50_expected_distribution': CURATED_50_COUNTS,
            'added_curated_drugs': added,
            'cyp_distribution': by_cyp,
            'source_distribution': by_source,
            'confidence_distribution': by_confidence,
        },
        'drugs': drugs,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))

    print(f'\nSaved to {out}')
    print('\nCYP Distribution:')
    for cyp, count in sorted(by_cyp.items()):
        print(f'  {cyp}: {count}')
    print('\nSource Distribution:')
    for source, count in sorted(by_source.items()):
        print(f'  {source}: {count}')
    return output


if __name__ == '__main__':
    merge_datasets()
