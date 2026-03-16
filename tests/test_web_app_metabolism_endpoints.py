from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from typing import Any, Dict

from enzyme_software.web_app import _RequestHandler


@contextmanager
def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RequestHandler)
    host, port = server.server_address[0], int(server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _post(base_url: str, path: str, payload: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(
        url=f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return int(resp.status), body
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        return int(exc.code), body


def test_api_metabolism_sites_ok():
    with _serve() as base:
        status, data = _post(
            base,
            "/api/metabolism/sites",
            {
                "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                "topk": 5,
                "use_xtb": False,
            },
        )
    assert status == 200
    assert "ranked_sites" in data
    assert "all_ranked_sites" in data
    assert "top_prediction" in data
    assert isinstance(data["ranked_sites"], list)


def test_api_metabolism_predict_ok():
    with _serve() as base:
        status, data = _post(
            base,
            "/api/metabolism/predict",
            {
                "smiles": "CN1CCC23C4C1CC=C2C3(CC5=CC(=C(C=C45)OC)O)O",
                "topk": 5,
                "use_xtb": False,
            },
        )
    assert status == 200
    for key in ["drug", "predicted_cyp", "top_metabolism_site", "reaction_class", "ranked_sites"]:
        assert key in data


def test_api_invalid_smiles_400():
    with _serve() as base:
        status, data = _post(
            base,
            "/api/metabolism/sites",
            {"smiles": "NOT_A_SMILES", "topk": 5, "use_xtb": False},
        )
    assert status == 400
    assert "error" in data


def test_api_pgx_query_ok():
    with _serve() as base:
        status, data = _post(
            base,
            "/api/pgx/query",
            {
                "drug_name": "warfarin",
                "genotype": {
                    "CYP2C9": "*1/*3",
                    "VKORC1": "GA",
                },
            },
        )
    assert status == 200
    assert data["drug"] == "Warfarin (S-enantiomer)"
    assert data["primary_cyp"] == "CYP2C9"
    assert data["combined_warfarin_dose_mg"] == 2.2


def test_api_pgx_query_invalid_drug_400():
    with _serve() as base:
        status, data = _post(
            base,
            "/api/pgx/query",
            {
                "drug_name": "tramadol",
                "genotype": {"CYP2D6": "*1/*1"},
            },
        )
    assert status == 400
    assert "error" in data


def test_api_metabolism_validate_ok():
    with _serve() as base:
        status, data = _post(
            base,
            "/api/metabolism/validate",
            {"topk_list": [1, 3]},
        )
    assert status == 200
    assert "validation" in data
    assert "report" in data
    assert "metrics" in data["validation"]
