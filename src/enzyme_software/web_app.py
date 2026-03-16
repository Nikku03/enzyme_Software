from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import errno
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from enzyme_software.context import OperationalConstraints
from enzyme_software.evidence_store import load_datapoints
from enzyme_software.input_parsing import ParsedEntry, parse_input_payload
from enzyme_software.pipeline import run_pipeline
from enzyme_software.calibration.layer4_validation import (
    VALIDATION_CASES,
    run_validation,
    run_validation_with_pipeline,
)
from enzyme_software.calibration.drug_metabolism_db import (
    DRUG_DATABASE,
    get_drug,
)
from enzyme_software.moduleB.metabolism_site_predictor import (
    predict_drug_metabolism,
    predict_metabolism_sites,
)
from enzyme_software.moduleC import patient_query
from enzyme_software.modules.moduleB2_validation import (
    run_drug_metabolism_validation,
)
from enzyme_software.modules.moduleB2_reporting import (
    build_metabolism_validation_report,
)
from enzyme_software.reporting import (
    render_debug,
    render_debug_report,
    render_demo,
    render_pretty,
    render_scientific_report,
    render_scientist,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT_DIR / "web" / "index.html"


def run_web_app(host: str = "127.0.0.1", port: int = 8000) -> int:
    server = _bind_server_with_fallback(host=host, preferred_port=port)
    bound_port = int(server.server_address[1])
    if bound_port != int(port):
        print(f"Preferred port {port} is busy; using {bound_port} instead.")
    print(f"Web UI running at http://{host}:{bound_port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BondBreak local web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args(argv)
    return run_web_app(host=args.host, port=args.port)


def _bind_server_with_fallback(
    host: str,
    preferred_port: int,
    max_attempts: int = 20,
) -> ThreadingHTTPServer:
    start = int(preferred_port)
    for offset in range(max(1, int(max_attempts))):
        port = start + offset
        try:
            return ThreadingHTTPServer((host, port), _RequestHandler)
        except OSError as exc:
            if exc.errno not in {errno.EADDRINUSE, 48, 98}:
                raise
    raise OSError(
        f"Could not bind web server on {host}:{start}-{start + max(1, int(max_attempts)) - 1}"
    )


class _RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/evidence":
            self._handle_evidence(parsed)
            return
        if parsed.path in {"/", "/index.html"}:
            self._send_index()
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/parse":
            self._handle_parse()
            return
        if parsed.path == "/api/route":
            self._handle_route()
            return
        if parsed.path == "/api/validation":
            self._handle_validation()
            return
        if parsed.path == "/api/metabolism/predict":
            self._handle_metabolism_predict()
            return
        if parsed.path == "/api/metabolism/sites":
            self._handle_metabolism_sites()
            return
        if parsed.path == "/api/metabolism/validate":
            self._handle_metabolism_validate()
            return
        if parsed.path == "/api/pgx/query":
            self._handle_pgx_query()
            return
        self._send_json(404, {"error": "Not found"})

    def _handle_parse(self) -> None:
        payload = self._read_json()
        if payload is None:
            self._send_json(400, {"error": "Invalid JSON payload."})
            return
        file_content = payload.get("file_content")
        file_name = payload.get("file_name")
        if not isinstance(file_content, str) or not file_content.strip():
            self._send_json(400, {"error": "file_content is required."})
            return
        try:
            entries, warnings = parse_input_payload(file_content, filename=file_name)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        serialized = [_serialize_entry(entry) for entry in entries]
        self._send_json(200, {"entries": serialized, "warnings": warnings})

    def _handle_route(self) -> None:
        payload = self._read_json()
        if payload is None:
            self._send_json(400, {"error": "Invalid JSON payload."})
            return
        smiles = payload.get("smiles")
        target_bond = payload.get("target_bond")
        requested_output = payload.get("requested_output")
        trap_target = payload.get("trap_target")
        constraints_payload = payload.get("constraints") or {}

        if not isinstance(smiles, str) or not smiles.strip():
            self._send_json(400, {"error": "smiles is required."})
            return
        if not isinstance(target_bond, str) or not target_bond.strip():
            self._send_json(400, {"error": "target_bond is required."})
            return

        constraints = _constraints_from_payload(constraints_payload)
        ctx = run_pipeline(
            smiles,
            target_bond,
            requested_output=requested_output,
            trap_target=trap_target,
            constraints=constraints,
        )
        payload = ctx.to_dict()
        payload["reports"] = {
            "demo_report": render_demo(payload),
            "pretty_report": render_pretty(payload),
            "scientist_report": render_scientist(payload),
            "scientific_report": render_scientific_report(payload),
            "debug_report": render_debug(payload),
        }
        self._send_json(200, payload)

    def _handle_validation(self) -> None:
        payload = self._read_json() or {}
        template_only = bool(payload.get("template_only"))
        case_ids = payload.get("case_ids")
        selected_cases = None
        if isinstance(case_ids, list) and case_ids:
            wanted = {str(item).strip() for item in case_ids if str(item).strip()}
            selected_cases = [case for case in VALIDATION_CASES if case.get("case_id") in wanted]

        report = (
            run_validation(pipeline_function=None, cases=selected_cases)
            if template_only
            else run_validation_with_pipeline(run_pipeline, cases=selected_cases)
        )
        self._send_json(
            200,
            {
                "validation_report": report,
                "selected_case_count": len(selected_cases) if selected_cases is not None else len(VALIDATION_CASES),
                "template_only": template_only,
            },
        )

    def _handle_metabolism_predict(self) -> None:
        payload = self._read_json() or {}
        drug_name = payload.get("drug_name") or payload.get("drug_key")
        drug = get_drug(str(drug_name)) if drug_name else None
        smiles = payload.get("smiles") or (drug or {}).get("smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            self._send_json(400, {"error": "smiles or known drug_name/drug_key is required."})
            return
        isoform_hint = payload.get("isoform_hint") or (drug or {}).get("primary_isoform") or (drug or {}).get("primary_cyp")
        try:
            topk = int(payload.get("topk", 5))
        except (TypeError, ValueError):
            topk = 5
        topk = max(1, min(topk, 20))
        use_xtb = bool(payload.get("use_xtb", False))

        try:
            prediction = predict_drug_metabolism(
                smiles=smiles,
                isoform_hint=str(isoform_hint) if isoform_hint else None,
                topk=topk,
                use_xtb=use_xtb,
            )
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception:
            self._send_json(500, {"error": "Drug metabolism prediction failed"})
            return
        self._send_json(
            200,
            {
                **prediction,
                "ground_truth": (drug or {}).get("ground_truth"),
                "expected": (drug or {}).get("expected"),
            },
        )

    def _handle_metabolism_sites(self) -> None:
        payload = self._read_json() or {}
        smiles = payload.get("smiles")
        if not isinstance(smiles, str) or not smiles.strip():
            self._send_json(400, {"error": "smiles is required."})
            return
        try:
            topk = int(payload.get("topk", 5))
        except (TypeError, ValueError):
            topk = 5
        topk = max(1, min(topk, 50))
        use_xtb = bool(payload.get("use_xtb", False))
        isoform_hint = payload.get("isoform_hint")

        try:
            result = predict_metabolism_sites(
                smiles=smiles,
                topk=topk,
                use_xtb=use_xtb,
                isoform_hint=str(isoform_hint) if isoform_hint else None,
            )
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except Exception:
            self._send_json(500, {"error": "Metabolism site prediction failed"})
            return
        self._send_json(200, result)

    def _handle_metabolism_validate(self) -> None:
        payload = self._read_json() or {}
        selected = payload.get("drug_keys")
        use_db = DRUG_DATABASE
        if isinstance(selected, list) and selected:
            wanted = {str(item).strip().lower() for item in selected if str(item).strip()}
            use_db = {k: v for k, v in DRUG_DATABASE.items() if k.lower() in wanted}
        topk_list = payload.get("topk_list")
        if not isinstance(topk_list, list) or not topk_list:
            topk_list = [1, 3, 5]
        validation = run_drug_metabolism_validation(drug_db=use_db, topk_list=topk_list)
        report = build_metabolism_validation_report(validation)
        self._send_json(
            200,
            {
                "validation": validation,
                "report": report,
            },
        )

    def _handle_pgx_query(self) -> None:
        payload = self._read_json() or {}
        drug_name = payload.get("drug_name") or payload.get("drug")
        genotype = payload.get("genotype")

        if not isinstance(drug_name, str) or not drug_name.strip():
            self._send_json(400, {"error": "drug_name is required."})
            return
        if genotype is not None and not isinstance(genotype, dict):
            self._send_json(400, {"error": "genotype must be an object when provided."})
            return

        result = patient_query(str(drug_name).strip(), genotype=genotype or None)
        if "error" in result:
            self._send_json(400, result)
            return
        self._send_json(200, result)

    def _send_index(self) -> None:
        if INDEX_PATH.is_file():
            content = INDEX_PATH.read_text(encoding="utf-8")
        else:
            content = _fallback_html()
        encoded = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _handle_evidence(self, parsed: Any) -> None:
        db_path = os.environ.get("EVIDENCE_DB_PATH")
        if not db_path:
            self._send_json(400, {"error": "EVIDENCE_DB_PATH is not set."})
            return
        params = parse_qs(parsed.query or "")
        run_id = (params.get("run_id") or [None])[0]
        if not run_id:
            self._send_json(400, {"error": "run_id is required."})
            return
        limit_raw = (params.get("limit") or [None])[0]
        module_id = (params.get("module_id") or [None])[0]
        item_type = (params.get("item_type") or [None])[0]
        scaffold_id = (params.get("scaffold_id") or [None])[0]
        variant_id = (params.get("variant_id") or [None])[0]
        limit = None
        if limit_raw:
            try:
                limit = int(limit_raw)
            except ValueError:
                limit = None
        datapoints = load_datapoints(db_path, run_id, limit=limit)
        filtered = []
        for point in datapoints:
            if module_id is not None:
                try:
                    if int(module_id) != int(point.get("module_id", -1)):
                        continue
                except ValueError:
                    pass
            if item_type and point.get("item_type") != item_type:
                continue
            if scaffold_id and point.get("scaffold_id") != scaffold_id:
                continue
            if variant_id and point.get("variant_id") != variant_id:
                continue
            filtered.append(point)
        self._send_json(
            200,
            {
                "run_id": run_id,
                "count": len(filtered),
                "datapoints": filtered,
            },
        )

    def _read_json(self) -> Dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length <= 0:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _serialize_entry(entry: ParsedEntry) -> Dict[str, Any]:
    return {
        "label": entry.label,
        "payload": entry.payload,
        "kind": entry.kind,
    }


def _constraints_from_payload(payload: Dict[str, Any]) -> OperationalConstraints:
    return OperationalConstraints(
        ph_min=_coerce_float(payload.get("ph_min")),
        ph_max=_coerce_float(payload.get("ph_max")),
        temperature_c=_coerce_float(payload.get("temperature_c")),
        metals_allowed=_coerce_bool(payload.get("metals_allowed")),
        oxidation_allowed=_coerce_bool(payload.get("oxidation_allowed")),
        host=_coerce_str(payload.get("host")),
        receptor_pdbqt=_coerce_str(payload.get("receptor_pdbqt")),
        receptor_pdb_id=_coerce_str(payload.get("receptor_pdb_id")),
        cyp_isoform=_coerce_str(payload.get("cyp_isoform")),
        enable_vina=_coerce_bool(payload.get("enable_vina")),
        enable_openmm=_coerce_bool(payload.get("enable_openmm")),
    )


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "allow"}:
            return True
        if lowered in {"false", "no", "n", "0", "forbid"}:
            return False
    return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _fallback_html() -> str:
    return """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>BondBreak UI</title>
  </head>
  <body>
    <p>Missing web/index.html. Re-install or run from source.</p>
  </body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
