from __future__ import annotations

import pytest


def test_predictor_smoke_and_stability():
    pytest.importorskip("rdkit")
    from enzyme_software.modules.moduleB2_drug_metabolism_predictor import predict_drug_metabolism

    smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"
    out1 = predict_drug_metabolism(smiles=smiles, isoform_hint="CYP2C9", topk=5)
    out2 = predict_drug_metabolism(smiles=smiles, isoform_hint="CYP2C9", topk=5)

    ranked1 = out1.get("ranked_sites") or []
    ranked2 = out2.get("ranked_sites") or []
    assert ranked1
    assert len(ranked1) <= 5
    assert (ranked1[0] or {}).get("site_id") == (ranked2[0] or {}).get("site_id")
    row = ranked1[0]
    assert "reaction_class" in row
    assert "predicted_route" in row
    assert "predicted_kcat_s_inv" in row
