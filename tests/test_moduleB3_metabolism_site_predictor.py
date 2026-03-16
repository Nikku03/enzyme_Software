from __future__ import annotations

import pytest


def test_predict_metabolism_sites_ibuprofen_smoke():
    pytest.importorskip("rdkit")
    from enzyme_software.moduleB.metabolism_site_predictor import predict_metabolism_sites

    out = predict_metabolism_sites("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", topk=5, use_xtb=False)
    ranked = out.get("ranked_sites") or []
    assert ranked
    assert len(ranked) <= 5
    top = ranked[0]
    assert "bond_class" in top
    assert "bde" in top
    assert "reaction_class" in top
    assert top["reaction_class"] in {
        "benzylic_hydroxylation",
        "alkyl_hydroxylation",
        "o_or_n_demethylation",
    }


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("COc1ccc2C[C@H]3N(C)CC[C@@]45[C@@H](Oc1c24)[C@H](O)C=C[C@@H]35", {"o_demethylation", "o_or_n_demethylation", "n_demethylation"}),
        ("COc1ccc2C[C@H]3N(C)CC[C@@]45[C@@H](Oc1c24)[C@@H](C=C[C@@H]35)O", {"o_demethylation", "o_or_n_demethylation", "n_demethylation"}),
        ("Clc1ccc2c(c1)C(=NC1=CN(C)C=N1)c1cc(F)ccc1-2", {"allylic_hydroxylation", "benzylic_hydroxylation"}),
    ],
)
def test_predict_metabolism_sites_sanity_reaction_classes(smiles: str, expected: set[str]):
    pytest.importorskip("rdkit")
    from enzyme_software.moduleB.metabolism_site_predictor import predict_metabolism_sites

    out = predict_metabolism_sites(smiles, topk=8, use_xtb=False)
    classes = {str(row.get("reaction_class")) for row in (out.get("ranked_sites") or [])}
    assert classes & expected


def test_predict_drug_metabolism_wrapper_shape():
    pytest.importorskip("rdkit")
    from enzyme_software.moduleB.metabolism_site_predictor import predict_drug_metabolism

    out = predict_drug_metabolism("Cc1ccccc1", topk=3, use_xtb=False, isoform_hint="CYP2C9")
    assert out.get("predicted_cyp") == "CYP2C9"
    assert isinstance(out.get("ranked_sites"), list)
    assert out.get("top_metabolism_site") is not None
    assert "reaction_class" in out

