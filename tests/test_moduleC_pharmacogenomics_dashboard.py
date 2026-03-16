from __future__ import annotations

from pathlib import Path

from enzyme_software.moduleC.pharmacogenomics_dashboard import (
    check_drug_interactions,
    determine_phenotype,
    end_to_end_query,
    generate_clinical_report,
    generate_module_c_dashboard,
    patient_query,
    population_risk_summary,
)


def test_determine_phenotype_valid_cases():
    pm = determine_phenotype("CYP2D6", "*4/*4")
    assert pm.get("abbreviation") == "PM"
    assert pm.get("activity_score") == 0.0

    im = determine_phenotype("CYP2D6", "*1/*4")
    assert im.get("abbreviation") == "IM"
    assert im.get("activity_score") == 1.0

    nm = determine_phenotype("CYP2C9", "*1/*2")
    assert nm.get("abbreviation") == "NM"
    assert nm.get("activity_score") == 1.5

    vkor = determine_phenotype("VKORC1", "GA")
    assert vkor.get("abbreviation") == "IS"
    assert vkor.get("activity_score") == 0.75


def test_determine_phenotype_invalid_cases():
    bad = determine_phenotype("CYP2D6", "*1")
    assert "error" in bad

    unknown = determine_phenotype("CYP2D6", "*1/*999")
    assert "error" in unknown


def test_patient_query_codeine_um_and_unknown_drug():
    result = patient_query("codeine", genotype={"CYP2D6": "*1/*1xN"})
    assert result.get("drug") == "Codeine"
    assert result.get("risk_level") == "CRITICAL"
    assert "fda_warning" in result
    assert "recommendation" in result
    assert result.get("pharmacokinetics", {}).get("morphine_conversion_pct") == 30

    missing = patient_query("not_a_drug", genotype={"CYP2D6": "*1/*1"})
    assert "error" in missing


def test_patient_query_warfarin_combines_vkorc1_and_cyp2c9():
    out = patient_query("warfarin", genotype={"CYP2C9": "*1/*3", "VKORC1": "GA"})
    assert out.get("drug") == "Warfarin (S-enantiomer)"
    assert out.get("recommended_dose_mg") == 3.75
    assert out.get("combined_warfarin_dose_mg") == 2.2
    assert out.get("vkorc1", {}).get("abbreviation") == "IS"
    assert out.get("pharmacokinetics", {}).get("auc_fold_change") == 1.5


def test_ddi_checker_detects_competition_and_inhibition():
    # Shared CYP2C9 substrate competition
    rep1 = check_drug_interactions(["warfarin", "ibuprofen"])
    assert rep1["n_interactions"] >= 1
    assert rep1["risk_summary"] in {"HIGH", "MODERATE"}
    assert any(i.get("type") == "substrate_competition" for i in rep1["interactions"])

    # Omeprazole as CYP2C19 moderate inhibitor affecting clopidogrel
    rep2 = check_drug_interactions(["omeprazole", "clopidogrel"])
    assert any(i.get("type") == "inhibition" for i in rep2["interactions"])


def test_population_risk_summary_shape():
    out = population_risk_summary("codeine", ethnicity="caucasian")
    assert out.get("drug") == "Codeine"
    assert out.get("gene") == "CYP2D6"
    assert isinstance(out.get("phenotype_distribution"), list)
    assert out.get("total_requiring_action_pct") is not None
    assert out.get("at_risk_pct") is not None


def test_end_to_end_query_uses_module_b_hooks():
    def _mock_cyp(smiles: str):
        return {"predicted_cyp": "CYP2C9", "smiles": smiles}

    def _mock_sites(_smiles: str):
        return [{"bond_class": "ch__benzylic", "bde_kj_mol": 375.5, "rank": 1}]

    out = end_to_end_query(
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        genotype={"CYP2C9": "*1/*3"},
        cyp_predictor=_mock_cyp,
        site_predictor=_mock_sites,
    )
    assert out.get("drug_match") == "ibuprofen"
    assert out.get("pharmacogenomics", {}).get("drug") == "Ibuprofen"
    assert out.get("cyp_prediction", {}).get("predicted_cyp") == "CYP2C9"
    assert out.get("site_prediction", {}).get("ranked_sites")


def test_report_and_dashboard_generation(tmp_path: Path):
    q = patient_query("codeine", genotype={"CYP2D6": "*4/*4"})
    report = generate_clinical_report(q)
    assert "PHARMACOGENOMICS CONSULTATION REPORT" in report
    assert "RISK LEVEL:" in report

    out_path = tmp_path / "module_c_dashboard.html"
    written = generate_module_c_dashboard(
        queries=[q],
        ddi_result=check_drug_interactions(["warfarin", "ibuprofen"]),
        population_results=[population_risk_summary("codeine", "caucasian")],
        output_path=str(out_path),
    )
    assert written == str(out_path)
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert "CYP-Predict: Pharmacogenomics Dashboard" in html
    assert "Patient Pharmacogenomics Queries" in html
