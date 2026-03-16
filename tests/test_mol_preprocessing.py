from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_prepare_mol_valid_smiles_status_ok():
    pytest.importorskip("rdkit")
    from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol

    result = prepare_mol("CCO")
    assert result.mol is not None
    assert result.status == "ok"
    assert result.canonical_smiles is not None
    assert result.repaired is False


def test_prepare_mol_recoverable_problematic_smiles():
    pytest.importorskip("rdkit")
    from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol

    result = prepare_mol("cC")
    assert result.mol is not None
    assert result.status in {"repaired_full_sanitize", "repaired_partial_sanitize"}
    assert result.canonical_smiles is not None
    assert result.repaired is True


def test_prepare_mol_invalid_smiles_fails_cleanly():
    pytest.importorskip("rdkit")
    from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol

    result = prepare_mol("not_a_smiles")
    assert result.mol is None
    assert result.status == "failed"
    assert result.error


def test_prepare_mol_no_aggressive_repair_by_default():
    pytest.importorskip("rdkit")
    from enzyme_software.liquid_nn_v2.utils.mol_preprocessing import prepare_mol

    result = prepare_mol("cC")
    assert result.aggressive_repair is False


def test_audit_script_writes_report(tmp_path):
    pytest.importorskip("rdkit")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "drugs": [
                    {"name": "valid", "smiles": "CCO"},
                    {"name": "repairable", "smiles": "cC"},
                    {"name": "bad", "smiles": "not_a_smiles"},
                ]
            }
        )
    )
    output_json = tmp_path / "audit.json"
    output_failures = tmp_path / "failures.csv"
    cmd = [
        sys.executable,
        "scripts/audit_smiles_preprocessing.py",
        "--dataset",
        str(dataset_path),
        "--output-json",
        str(output_json),
        "--output-failures",
        str(output_failures),
    ]
    completed = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_json.read_text())
    summary = payload["summary"]
    assert summary["total_molecules"] == 3
    assert summary["failed"] >= 1
    assert output_failures.exists()
