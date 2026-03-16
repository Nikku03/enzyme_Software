from __future__ import annotations

import json
from pathlib import Path

from enzyme_software.moduleB.module_b_figures import (
    generate_dashboard_html,
    main as figures_main,
)
from enzyme_software.moduleB.module_b_validation import (
    _mock_cyp_predictions,
    _mock_site_predictions,
    validate_cyp_predictions,
    validate_site_predictions,
)


def test_generate_dashboard_html_contains_4_figure_sections():
    cyp = validate_cyp_predictions(_mock_cyp_predictions())
    site = validate_site_predictions(_mock_site_predictions())
    html = generate_dashboard_html(cyp, site)
    assert "Figure 1: CYP Isoform Confusion Matrix" in html
    assert "Figure 2: Metabolism Site Accuracy (Per Drug)" in html
    assert "Figure 3: BDE Bond Ranking Walkthrough (Ibuprofen)" in html
    assert "Figure 4: Pharmacogenomics Case Study (Codeine / CYP2D6)" in html
    assert "<html" in html.lower()


def test_figures_main_writes_html_from_mock(tmp_path: Path):
    out = tmp_path / "dash_mock.html"
    rc = figures_main(["--output", str(out)])
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "CYP-Predict Module B Dashboard" in text


def test_figures_main_writes_html_from_json(tmp_path: Path):
    cyp = validate_cyp_predictions(_mock_cyp_predictions())
    site = validate_site_predictions(_mock_site_predictions())
    report = {"cyp_validation": cyp, "site_validation": site}
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(report), encoding="utf-8")
    out = tmp_path / "dash_real.html"

    rc = figures_main(["--json", str(json_path), "--output", str(out)])
    assert rc == 0
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Figure 1: CYP Isoform Confusion Matrix" in text

