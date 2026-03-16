from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from enzyme_software.moduleB.module_b_validation import (
        CYP_ISOFORMS,
        _mock_cyp_predictions,
        _mock_site_predictions,
        export_confusion_matrix_data,
        export_site_accuracy_data,
        validate_cyp_predictions,
        validate_site_predictions,
    )
except Exception:  # pragma: no cover
    # Script-mode fallback
    from module_b_validation import (  # type: ignore
        CYP_ISOFORMS,
        _mock_cyp_predictions,
        _mock_site_predictions,
        export_confusion_matrix_data,
        export_site_accuracy_data,
        validate_cyp_predictions,
        validate_site_predictions,
    )


def _safe(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _build_confusion_matrix_html(cm_matrix: List[List[int]], labels: List[str]) -> str:
    header = "".join(f'<div class="cm-cell cm-header">{_safe(label)}</div>' for label in labels)
    rows: List[str] = []
    max_val = max((v for row in cm_matrix for v in row), default=1) or 1
    for i, pred in enumerate(labels):
        row_cells = [f'<div class="cm-cell cm-label">{_safe(pred)}</div>']
        for j, _actual in enumerate(labels):
            val = int(cm_matrix[i][j]) if i < len(cm_matrix) and j < len(cm_matrix[i]) else 0
            is_diag = i == j
            if val <= 0:
                bg = "#f8f9fa"
                fg = "#bdc3c7"
            elif is_diag:
                bg = "#27ae60"
                fg = "#ffffff"
            else:
                ratio = min(1.0, max(0.0, float(val) / float(max_val)))
                intensity = int(240 - 70 * ratio)
                bg = f"rgb(255,{intensity},{intensity})"
                fg = "#922b21"
            row_cells.append(f'<div class="cm-cell" style="background:{bg};color:{fg}">{val}</div>')
        rows.append("".join(row_cells))
    return f'<div class="cm-grid"><div class="cm-cell"></div>{header}{"".join(rows)}</div>'


def _build_site_accuracy_bars(drugs: List[str], correct_flags: List[int]) -> str:
    bars: List[str] = []
    for drug, flag in zip(drugs, correct_flags):
        value = 1 if int(flag) else 0
        width = 100 if value else 8
        color = "#27ae60" if value else "#e74c3c"
        label = "Hit" if value else "Miss"
        bars.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{_safe(drug)}</div>
              <div class="bar-track">
                <div class="bar-fill" style="width:{width}%;background:{color}">{label}</div>
              </div>
            </div>
            """
        )
    return "".join(bars)


def _build_bde_walkthrough_html() -> str:
    ibuprofen_bonds: List[Tuple[str, float, str, str]] = [
        ("Benzylic C-H", 375.5, "#e74c3c", "most vulnerable"),
        ("Isobutyl 2° C-H", 412.5, "#f39c12", "known metabolic region"),
        ("Isobutyl 1° C-H", 423.0, "#f1c40f", "known metabolic region"),
        ("Aryl C-H", 472.2, "#27ae60", "resistant"),
    ]
    max_bde = 500.0
    rows: List[str] = []
    for label, bde, color, note in ibuprofen_bonds:
        width = max(4.0, min(100.0, (float(bde) / max_bde) * 100.0))
        rows.append(
            f"""
            <div class="bde-row">
              <div class="bde-title">{_safe(label)}</div>
              <div class="bde-note">{_safe(note)}</div>
              <div class="bde-track">
                <div class="bde-fill" style="width:{width:.1f}%;background:{color}">
                  {bde:.1f} kJ/mol
                </div>
              </div>
            </div>
            """
        )
    return "".join(rows)


def _build_pgx_case_html() -> str:
    rows = [
        ("Normal Metabolizer (NM)", "60-70%", "Normal codeine -> morphine conversion", "Standard dosing", "#27ae60"),
        ("Intermediate Metabolizer (IM)", "20-25%", "Reduced morphine formation", "Consider alternative", "#f39c12"),
        ("Poor Metabolizer (PM)", "5-10%", "Low efficacy risk", "Alternative analgesic", "#e74c3c"),
        ("Ultrarapid Metabolizer (UM)", "1-10%", "Toxicity risk from excess morphine", "Avoid codeine", "#8e44ad"),
    ]
    tr = []
    for phenotype, freq, effect, action, color in rows:
        tr.append(
            f"""
            <tr>
              <td style="border-left:4px solid {color};padding-left:10px">{_safe(phenotype)}</td>
              <td style="text-align:center">{_safe(freq)}</td>
              <td>{_safe(effect)}</td>
              <td><span class="pill" style="background:{color}">{_safe(action)}</span></td>
            </tr>
            """
        )
    return "".join(tr)


def generate_dashboard_html(cyp_validation: Dict[str, Any], site_validation: Dict[str, Any]) -> str:
    cm = export_confusion_matrix_data(cyp_validation)
    sa = export_site_accuracy_data(site_validation)
    cm_html = _build_confusion_matrix_html(cm["matrix"], cm["labels"])
    bars_html = _build_site_accuracy_bars(sa["drugs"], sa["correct"])
    bde_html = _build_bde_walkthrough_html()
    pgx_html = _build_pgx_case_html()

    cyp_score = f"{cyp_validation.get('correct', 0)}/{cyp_validation.get('total', 0)}"
    site_score = f"{site_validation.get('correct', 0)}/{site_validation.get('total', 0)}"
    combined_correct = int(cyp_validation.get("correct", 0)) + int(site_validation.get("correct", 0))
    combined_total = int(cyp_validation.get("total", 0)) + int(site_validation.get("total", 0))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Module B Validation Dashboard</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:#f5f7fb; color:#1f2d3d; }}
    .header {{ background: linear-gradient(135deg,#1a5276,#2980b9); color:#fff; padding:28px 34px; }}
    .header h1 {{ margin:0; font-size:26px; }}
    .header p {{ margin:6px 0 0; opacity:.9; font-size:13px; }}
    .container {{ max-width: 1120px; margin: 0 auto; padding: 20px; }}
    .metrics {{ display:grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap:14px; margin-bottom:18px; }}
    .metric {{ background:#fff; border-radius:8px; padding:16px; text-align:center; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    .metric .v {{ font-size:30px; font-weight:700; color:#1f78b4; }}
    .metric .l {{ font-size:12px; color:#6c7a89; text-transform:uppercase; letter-spacing:.5px; }}
    .panel {{ background:#fff; border-radius:8px; padding:18px; margin-bottom:16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
    .panel h2 {{ margin:0 0 12px; font-size:18px; color:#1a5276; border-bottom:2px solid #ecf0f1; padding-bottom:8px; }}
    .grid-2 {{ display:grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap:16px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th {{ background:#1a5276; color:#fff; text-align:left; padding:8px 10px; }}
    td {{ border-bottom:1px solid #edf1f5; padding:7px 10px; }}
    .cm-grid {{ display:grid; grid-template-columns: 130px repeat(5,minmax(0,1fr)); gap:2px; font-size:12px; }}
    .cm-cell {{ min-height:34px; display:flex; align-items:center; justify-content:center; border-radius:3px; font-weight:600; }}
    .cm-header {{ background:#1a5276; color:#fff; }}
    .cm-label {{ background:#2980b9; color:#fff; }}
    .bar-row {{ display:grid; grid-template-columns: 180px 1fr; gap:10px; align-items:center; margin:7px 0; }}
    .bar-label {{ font-size:12px; color:#34495e; }}
    .bar-track {{ background:#ecf0f1; border-radius:4px; overflow:hidden; }}
    .bar-fill {{ min-height:24px; display:flex; align-items:center; padding:0 8px; color:#fff; font-size:11px; font-weight:700; }}
    .bde-row {{ margin:8px 0 12px; }}
    .bde-title {{ font-size:13px; font-weight:600; }}
    .bde-note {{ font-size:11px; color:#6c7a89; margin:2px 0 4px; }}
    .bde-track {{ background:#ecf0f1; border-radius:4px; overflow:hidden; }}
    .bde-fill {{ min-height:24px; display:flex; align-items:center; color:#fff; font-size:11px; font-weight:700; padding-left:8px; }}
    .pill {{ display:inline-block; color:#fff; border-radius:10px; padding:2px 8px; font-size:11px; font-weight:700; }}
    .foot {{ margin-top:8px; font-size:11px; color:#7f8c8d; font-style:italic; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>CYP-Predict Module B Dashboard</h1>
    <p>Validation figures for CYP assignment, site prediction, BDE walkthrough, and pharmacogenomics case study.</p>
  </div>
  <div class="container">
    <div class="metrics">
      <div class="metric"><div class="v">{_safe(cyp_score)}</div><div class="l">CYP Accuracy</div></div>
      <div class="metric"><div class="v">{_safe(site_score)}</div><div class="l">Site Accuracy</div></div>
      <div class="metric"><div class="v">{combined_correct}/{combined_total}</div><div class="l">Combined Score</div></div>
      <div class="metric"><div class="v">{_safe(len(cyp_validation.get("per_drug", [])))}</div><div class="l">FDA Drugs</div></div>
    </div>

    <div class="grid-2">
      <div class="panel">
        <h2>Figure 1: CYP Isoform Confusion Matrix</h2>
        {cm_html}
        <div class="foot">Rows = predicted, columns = actual. Diagonal cells indicate correct assignments.</div>
      </div>
      <div class="panel">
        <h2>Figure 2: Metabolism Site Accuracy (Per Drug)</h2>
        {bars_html}
        <div class="foot">Green bars are top-k hits; red bars are misses for expected site class.</div>
      </div>
    </div>

    <div class="panel">
      <h2>Figure 3: BDE Bond Ranking Walkthrough (Ibuprofen)</h2>
      {bde_html}
      <div class="foot">Lower BDE values correspond to higher oxidative vulnerability in this ranking framework.</div>
    </div>

    <div class="panel">
      <h2>Figure 4: Pharmacogenomics Case Study (Codeine / CYP2D6)</h2>
      <table>
        <tr><th>Phenotype</th><th style="text-align:center">Frequency</th><th>Metabolic Effect</th><th>Clinical Action</th></tr>
        {pgx_html}
      </table>
      <div class="foot">Illustrative clinical stratification; align final labels with current CPIC/FDA references in clinical materials.</div>
    </div>
  </div>
</body>
</html>"""


def generate_dashboard_from_report(report: Dict[str, Any]) -> str:
    return generate_dashboard_html(report["cyp_validation"], report["site_validation"])


def write_dashboard_html(html_text: str, output_path: str) -> str:
    path = Path(output_path).expanduser().resolve()
    path.write_text(html_text, encoding="utf-8")
    return str(path)


def _default_output_path() -> str:
    return str((Path(__file__).resolve().parent / "module_b_dashboard.html").resolve())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Module B validation dashboard HTML.")
    parser.add_argument("--json", dest="json_path", default=None, help="Path to validation report JSON.")
    parser.add_argument("--output", dest="output_path", default=_default_output_path(), help="Output HTML file path.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.json_path:
        report = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
        cyp_val = report["cyp_validation"]
        site_val = report["site_validation"]
    else:
        cyp_preds = _mock_cyp_predictions()
        site_preds = _mock_site_predictions()
        cyp_val = validate_cyp_predictions(cyp_preds)
        site_val = validate_site_predictions(site_preds)

    html_text = generate_dashboard_html(cyp_val, site_val)
    out = write_dashboard_html(html_text, args.output_path)
    print(f"Dashboard generated: {out}")
    print(f"CYP accuracy: {cyp_val.get('correct', 0)}/{cyp_val.get('total', 0)}")
    print(f"Site accuracy: {site_val.get('correct', 0)}/{site_val.get('total', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

