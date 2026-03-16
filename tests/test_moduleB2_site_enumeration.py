from __future__ import annotations

import pytest


def test_enumerate_sites_for_ibuprofen_has_candidates():
    Chem = pytest.importorskip("rdkit.Chem")
    from enzyme_software.modules.moduleB2_site_enumeration import enumerate_metabolism_targets

    mol = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")
    sites = enumerate_metabolism_targets(mol)
    assert len(sites) >= 5
    assert any("ch" in str(site.get("site_class")) for site in sites)


def test_enumerate_sites_detects_dealkylation_for_codeine_like_scaffold():
    Chem = pytest.importorskip("rdkit.Chem")
    from enzyme_software.modules.moduleB2_site_enumeration import enumerate_metabolism_targets

    mol = Chem.MolFromSmiles("COc1ccccc1")
    sites = enumerate_metabolism_targets(mol)
    classes = {str(site.get("site_class")) for site in sites}
    assert "o_demethyl" in classes or "o_dealkyl" in classes
