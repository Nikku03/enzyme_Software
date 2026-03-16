from __future__ import annotations

import json

import pytest


def test_prepare_mol_logs_provenance_for_repaired_smiles(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol
    from enzyme_software.liquid_nn_v2.utils.mol_provenance import mol_provenance_context

    log_path = tmp_path / "provenance.jsonl"
    monkeypatch.setenv("LNN_MOL_PROVENANCE_LOG", str(log_path))

    with mol_provenance_context(
        caller_module="unit_test",
        module_triggered="unit_test",
        source_category="graph builder",
        original_smiles="cC",
        drug_name="repairable",
        drug_id="drug-1",
    ):
        result = prepare_mol("cC")

    assert result.mol is not None
    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert len(lines) >= 1
    assert any(line["source_category"] == "graph builder" for line in lines)
    assert any(line["drug_id"] == "drug-1" for line in lines)


def test_manual_engine_internal_provenance_log(tmp_path, monkeypatch):
    from enzyme_software.liquid_nn_v2.features import manual_engine_features as mef

    log_path = tmp_path / "provenance_manual.jsonl"
    monkeypatch.setenv("LNN_MOL_PROVENANCE_LOG", str(log_path))

    class DummyCtx:
        data = {}

    def _fake_run_pipeline(*args, **kwargs):
        import sys

        print("[RDKit] ERROR: Can't kekulize mol.", file=sys.stderr)
        return DummyCtx()

    monkeypatch.setattr(mef, "run_pipeline", _fake_run_pipeline)
    mef._run_manual_engine("CCO", "C-H")

    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert any(line["source_category"] == "manual-engine internal module" for line in lines)
    assert any("kekulize" in (line.get("rdkit_message") or "").lower() for line in lines)
