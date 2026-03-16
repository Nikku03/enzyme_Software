from __future__ import annotations

import json
from pathlib import Path

from scripts.build_dataset.build_final_dataset import build_dataset
from scripts.build_dataset.identify_som import identify_som_from_reaction_type
from scripts.build_dataset.parse_drugbank import extract_cyp_enzymes, extract_reaction_types, parse_drugbank_xml
from scripts.build_dataset.validate_dataset import validate_dataset

from enzyme_software.liquid_nn_v2.data.drug_database import load_training_dataset


def _write_sample_drugbank_xml(path: Path) -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca">
  <drug type="small molecule">
    <drugbank-id primary="true">DBTEST1</drugbank-id>
    <name>Testazole</name>
    <calculated-properties>
      <property>
        <kind>SMILES</kind>
        <value>COc1ccccc1</value>
      </property>
    </calculated-properties>
    <metabolism>CYP2D6 mediates O-demethylation and CYP3A4 contributes minor hydroxylation.</metabolism>
  </drug>
</drugbank>
"""
    path.write_text(xml)


def test_parse_drugbank_helpers():
    enzymes = extract_cyp_enzymes("Metabolized primarily by CYP2D6 and CYP3A4.")
    assert enzymes == ["CYP2D6", "CYP3A4"]
    reactions = extract_reaction_types("Undergoes O-demethylation followed by hydroxylation.")
    assert "o_dealkylation" in reactions
    assert "hydroxylation" in reactions


def test_identify_som_from_reaction_type():
    hits = identify_som_from_reaction_type("COc1ccccc1", "o_dealkylation", "CYP2D6")
    assert hits
    atom_indices = [idx for idx, _bond_class in hits]
    assert 0 in atom_indices


def test_build_and_validate_dataset(tmp_path: Path):
    xml_path = tmp_path / "drugbank.xml"
    out_path = tmp_path / "dataset.json"
    _write_sample_drugbank_xml(xml_path)

    parsed = parse_drugbank_xml(xml_path)
    assert len(parsed) == 1
    assert parsed[0]["enzymes"] == ["CYP2D6", "CYP3A4"]

    dataset = build_dataset(str(xml_path), output_path=str(out_path))
    assert out_path.exists()
    assert dataset["metadata"]["total_drugs"] >= 14
    assert any(drug["name"] == "Testazole" for drug in dataset["drugs"])

    loaded = load_training_dataset(str(out_path))
    assert len(loaded) == dataset["metadata"]["total_drugs"]

    with out_path.open() as handle:
        payload = json.load(handle)
    assert payload["metadata"]["cyp_distribution"]["CYP2D6"] >= 1

    assert validate_dataset(str(out_path)) is True
