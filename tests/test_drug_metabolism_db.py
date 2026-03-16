from __future__ import annotations

from enzyme_software.calibration.drug_metabolism_db import (
    DRUG_DATABASE,
    get_drug,
    get_validation_set,
    list_by_cyp,
    list_drugs,
)


def test_get_drug_by_key_and_name_case_insensitive():
    by_key = get_drug("ibuprofen")
    by_name = get_drug("Ibuprofen")
    assert by_key is not None
    assert by_name is not None
    assert by_key["primary_cyp"] == "CYP2C9"
    assert by_name["drugbank_id"] == "DB01050"


def test_list_by_cyp_returns_expected_entries():
    cyp2d6 = list_by_cyp("cyp2d6")
    names = {item["name"] for item in cyp2d6}
    assert "Codeine" in names
    assert "Metoprolol" in names
    assert len(cyp2d6) >= 3


def test_validation_set_shape_and_size():
    rows = get_validation_set()
    assert len(rows) == len(DRUG_DATABASE)
    required = {
        "drug_key",
        "name",
        "smiles",
        "expected_cyp",
        "expected_site_type",
        "expected_bde_class",
        "primary_isoform",
        "ground_truth",
        "expected",
        "has_acidic_group",
        "has_basic_nitrogen",
        "is_large_hydrophobic",
        "is_planar_aromatic",
    }
    for row in rows:
        assert required.issubset(row.keys())
        assert isinstance(row["ground_truth"], dict)
        assert isinstance(row["expected"], dict)


def test_list_drugs_contains_known_drugs():
    names = set(list_drugs())
    assert "Warfarin (S-enantiomer)" in names
    assert "Midazolam" in names
