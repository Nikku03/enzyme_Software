"""Layer 4 literature validation suite.

Reality-check cases used to compare pipeline outputs against published
enzyme engineering campaigns.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional


CALIBRATION_VERSION = "layer4_validation.v1"


VALIDATION_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "val_p450bm3_propane",
        "title": "P450-BM3 propane hydroxylation",
        "reference": "Fasan et al. 2007/2008",
        "substrate": {"smiles": "CCC", "target_bond": "C-H", "name": "propane"},
        "pipeline_should_predict": {
            "route": ["P450", "non_heme_iron"],
            "kcat_order_of_magnitude": [-2, 0],
            "decision": ["LOW_CONF_GO", "NO_GO", "GO"],
        },
    },
    {
        "case_id": "val_p450bm3_toluene",
        "title": "P450-BM3 toluene benzylic hydroxylation",
        "reference": "Li et al. 2008; Whitehouse et al. 2012",
        "substrate": {"smiles": "Cc1ccccc1", "target_bond": "C-H", "name": "toluene"},
        "pipeline_should_predict": {
            "route": ["P450", "non_heme_iron"],
            "kcat_order_of_magnitude": [-1, 2],
        },
    },
    {
        "case_id": "val_p450cam_camphor",
        "title": "P450cam camphor hydroxylation",
        "reference": "Mueller et al. 1995",
        "substrate": {
            "smiles": "CC1(C)C2CCC1(C)C(=O)C2",
            "target_bond": "C-H",
            "name": "camphor",
        },
        "pipeline_should_predict": {
            "route": ["P450"],
            "kcat_order_of_magnitude": [0, 2],
        },
    },
    {
        "case_id": "val_subtilisin_ester_amide",
        "title": "Subtilisin amide hydrolysis",
        "reference": "Chen & Arnold 1993",
        "substrate": {"smiles": "CC(=O)NC", "target_bond": "amide", "name": "N-methylacetamide"},
        "pipeline_should_predict": {
            "route": ["serine_hydrolase", "metallo_esterase"],
            "kcat_order_of_magnitude": [-1, 1],
        },
    },
    {
        "case_id": "val_cutinase_ester",
        "title": "Cutinase ester hydrolysis",
        "reference": "Martinez et al. 1994",
        "substrate": {"smiles": "CC(=O)OCC", "target_bond": "ester", "name": "ethyl acetate"},
        "pipeline_should_predict": {
            "route": ["serine_hydrolase"],
            "kcat_order_of_magnitude": [0, 2],
        },
    },
    {
        "case_id": "val_calb_resolution",
        "title": "CalB kinetic resolution substrate",
        "reference": "Rotticci et al. 2001",
        "substrate": {
            "smiles": "CC(=O)OC(C)c1ccccc1",
            "target_bond": "ester",
            "name": "1-phenylethyl acetate",
        },
        "pipeline_should_predict": {
            "route": ["serine_hydrolase"],
            "kcat_order_of_magnitude": [0, 2],
        },
    },
    {
        "case_id": "val_dhaa_dce",
        "title": "DhaA 1,2-dichloroethane",
        "reference": "Pavlova et al. 2009",
        "substrate": {"smiles": "ClCCCl", "target_bond": "C-Cl", "name": "1,2-dichloroethane"},
        "pipeline_should_predict": {
            "route": ["haloalkane_dehalogenase"],
            "kcat_order_of_magnitude": [-2, 1],
        },
    },
    {
        "case_id": "val_linb_chlorobutane",
        "title": "LinB 1-chlorobutane",
        "reference": "Damborsky et al. 2001",
        "substrate": {"smiles": "CCCCCl", "target_bond": "C-Cl", "name": "1-chlorobutane"},
        "pipeline_should_predict": {
            "route": ["haloalkane_dehalogenase"],
            "kcat_order_of_magnitude": [-1, 1],
        },
    },
    {
        "case_id": "val_taud_taurine",
        "title": "TauD taurine hydroxylation",
        "reference": "Hausinger 2004",
        "substrate": {"smiles": "NCCS(=O)(=O)O", "target_bond": "C-H", "name": "taurine"},
        "pipeline_should_predict": {
            "route": ["non_heme_iron", "P450"],
            "kcat_order_of_magnitude": [0, 1],
        },
    },
    {
        "case_id": "val_cf3h_impossible",
        "title": "CF3-H activation is very difficult",
        "reference": "negative control",
        "substrate": {"smiles": "FC(F)F", "target_bond": "C-H", "name": "fluoroform"},
        "pipeline_should_predict": {
            "decision": ["NO_GO", "LOW_CONF_GO"],
            "kcat_order_of_magnitude": [-4, -2],
        },
    },
    {
        "case_id": "val_perfluorooctane_impossible",
        "title": "Perfluorooctane has no C-H",
        "reference": "negative control",
        "substrate": {
            "smiles": "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
            "target_bond": "C-H",
            "name": "perfluorooctane",
        },
        "pipeline_should_predict": {"decision": ["NO_GO", "HALT_NEED_SELECTION"]},
    },
    {
        "case_id": "val_aspirin_ester",
        "title": "Aspirin ester hydrolysis",
        "reference": "benchmark case",
        "substrate": {
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "target_bond": "ester",
            "name": "aspirin",
        },
        "pipeline_should_predict": {
            "route": ["serine_hydrolase"],
            "kcat_order_of_magnitude": [0, 2],
        },
    },
]


def _extract_observed(output: Dict[str, Any]) -> Dict[str, Any]:
    summary = output.get("pipeline_summary") or {}
    results = summary.get("results") or {}
    job_card = output.get("job_card") or {}
    energy = job_card.get("energy_ledger") or {}
    return {
        "route": results.get("route") or job_card.get("route"),
        "decision": results.get("decision") or job_card.get("decision"),
        "predicted_kcat_s_inv": energy.get("k_eff_s_inv"),
    }


def _check_case(expected: Dict[str, Any], observed: Dict[str, Any]) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    if "route" in expected:
        allowed = expected["route"] if isinstance(expected["route"], list) else [expected["route"]]
        got = observed.get("route")
        checks.append({"check": "route", "expected": allowed, "got": got, "pass": got in allowed})
    if "decision" in expected:
        allowed = expected["decision"] if isinstance(expected["decision"], list) else [expected["decision"]]
        got = observed.get("decision")
        checks.append({"check": "decision", "expected": allowed, "got": got, "pass": got in allowed})
    if "kcat_order_of_magnitude" in expected:
        lo, hi = expected["kcat_order_of_magnitude"]
        kcat = observed.get("predicted_kcat_s_inv")
        if isinstance(kcat, (int, float)) and kcat > 0:
            log_k = math.log10(float(kcat))
            ok = float(lo) <= float(log_k) <= float(hi)
            checks.append(
                {
                    "check": "kcat_magnitude",
                    "expected_log10_range": [lo, hi],
                    "got_log10": round(log_k, 2),
                    "got_kcat": float(kcat),
                    "pass": ok,
                }
            )
        else:
            checks.append(
                {
                    "check": "kcat_magnitude",
                    "expected_log10_range": [lo, hi],
                    "got_kcat": kcat,
                    "pass": False,
                }
            )
    return checks


def run_validation(
    pipeline_function: Optional[Callable[[str, str], Dict[str, Any]]] = None,
    *,
    cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    use_cases = list(cases or VALIDATION_CASES)

    if pipeline_function is None:
        for case in use_cases:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "status": "template",
                    "what_to_check": case.get("pipeline_should_predict") or {},
                }
            )
        return {
            "version": CALIBRATION_VERSION,
            "total": len(rows),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "templates": len(rows),
            "cases": rows,
        }

    for case in use_cases:
        smiles = (case.get("substrate") or {}).get("smiles")
        token = (case.get("substrate") or {}).get("target_bond")
        row: Dict[str, Any] = {
            "case_id": case.get("case_id"),
            "title": case.get("title"),
            "status": "error",
            "checks": [],
        }
        try:
            output = pipeline_function(str(smiles or ""), str(token or ""))
            observed = _extract_observed(output)
            checks = _check_case(case.get("pipeline_should_predict") or {}, observed)
            row.update(
                {
                    "status": "pass" if checks and all(c.get("pass") for c in checks) else "fail",
                    "checks": checks,
                    "observed": observed,
                }
            )
        except Exception as exc:
            row.update({"status": "error", "error": str(exc)})
        rows.append(row)

    return {
        "version": CALIBRATION_VERSION,
        "total": len(rows),
        "passed": sum(1 for r in rows if r.get("status") == "pass"),
        "failed": sum(1 for r in rows if r.get("status") == "fail"),
        "errors": sum(1 for r in rows if r.get("status") == "error"),
        "templates": sum(1 for r in rows if r.get("status") == "template"),
        "cases": rows,
    }


def run_validation_with_pipeline(
    run_pipeline_fn: Callable[..., Any],
    *,
    cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Adapter for enzyme_software.pipeline.run_pipeline signature."""

    def _adapter(smiles: str, target_bond: str) -> Dict[str, Any]:
        ctx = run_pipeline_fn(smiles=smiles, target_bond=target_bond)
        return {
            "pipeline_summary": ctx.data.get("pipeline_summary") or {},
            "job_card": ctx.data.get("job_card") or {},
        }

    return run_validation(_adapter, cases=cases)
