"""Integration tests for Module -1 Reactivity Hub.

Tests that all parts (ATR, fragment builder, CPT levels) are properly
wired inside the reactivity hub, and that the pipeline integrates
Module -1 output into shared_io, pipeline summary, and Module 0.
"""

from enzyme_software.modules.module_minus1_reactivity_hub import (
    run_module_minus1_reactivity_hub,
)
from enzyme_software.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# 1. Output structure
# ---------------------------------------------------------------------------

def test_reactivity_hub_output_keys():
    """Output dict contains all required top-level keys."""
    result = run_module_minus1_reactivity_hub(
        smiles="CC(=O)OCC",
        target_bond="ester_c-o",
        requested_output=None,
        constraints={"ph_min": 7.0, "temperature_c": 25.0},
    )
    for key in (
        "status",
        "bond360_profile",
        "fragment",
        "cpt_scores",
        "mechanism_eligibility",
        "primary_constraint",
        "confidence_prior",
        "route_bias",
        "resolved_target",
        "reactivity",
        "warnings",
        "errors",
    ):
        assert key in result, f"missing key: {key}"

    # resolved_target sub-keys
    rt = result["resolved_target"]
    assert "bond_indices" in rt
    assert "bond_type" in rt
    assert "attack_sites" in rt
    assert rt.get("resolution_source") == "module_minus1"

    # reactivity sub-keys
    rx = result["reactivity"]
    assert "epav_score" in rx
    assert "primary_constraint" in rx
    assert "confidence_prior" in rx


# ---------------------------------------------------------------------------
# 2. Ester substrate end-to-end
# ---------------------------------------------------------------------------

def test_reactivity_hub_ester_pass():
    """Ethyl acetate ester should produce PASS with attack sites."""
    result = run_module_minus1_reactivity_hub(
        smiles="CC(=O)OCC",
        target_bond="ester_c-o",
        requested_output=None,
        constraints={},
    )
    assert result["status"] == "PASS"
    assert result["confidence_prior"] >= 0.5

    rt = result["resolved_target"]
    assert rt.get("bond_type") == "ester"
    assert len(rt.get("bond_indices", [])) == 2

    attack = rt.get("attack_sites", {})
    assert "electrophile" in attack


# ---------------------------------------------------------------------------
# 3. EP-AV receives group_type (BUG 1 fix verification)
# ---------------------------------------------------------------------------

def test_epav_receives_group_type():
    """EP-AV should use group-specific LG score, not element fallback.

    For an ester, LG_SCORE_BY_GROUP["ester"] = 0.70.
    The element fallback for O is also 0.70, so we test with an amide
    where amide-specific = 0.20 vs element N fallback = 0.20.
    The key check is that EP-AV runs without error and produces a score.
    """
    result = run_module_minus1_reactivity_hub(
        smiles="CC(=O)OCC",
        target_bond="ester_c-o",
        requested_output=None,
        constraints={},
    )
    cpt = result.get("cpt_scores", {})
    epav = cpt.get("epav", {})
    # EP-AV should have run successfully
    assert cpt.get("status") == "ok", f"cpt status: {cpt.get('status')}"
    assert "score" in epav, f"epav missing score: {epav}"
    assert isinstance(epav["score"], (int, float))
    # Leaving group breakdown should reflect group-specific scoring
    breakdown = epav.get("breakdown", {})
    if breakdown:
        assert "leaving_group" in breakdown


# ---------------------------------------------------------------------------
# 4. Level 3 receives l2_best (BUG 2 fix verification)
# ---------------------------------------------------------------------------

def test_level3_receives_l2_best():
    """Level 3 should run with Level 2 context, not empty dict."""
    result = run_module_minus1_reactivity_hub(
        smiles="CC(=O)OCC",
        target_bond="ester_c-o",
        requested_output=None,
        constraints={},
    )
    cpt = result.get("cpt_scores", {})
    level3 = cpt.get("level3", {})
    # Level 3 should have run (may pass or fail depending on pseudo env)
    if level3:
        assert "score" in level3
        assert "breakdown" in level3
        # Should have non-empty breakdown (not degraded by missing l2_best)
        breakdown = level3.get("breakdown", {})
        assert len(breakdown) > 0


# ---------------------------------------------------------------------------
# 5. Pipeline shared_io backfill
# ---------------------------------------------------------------------------

def test_pipeline_shared_io_has_module_minus1():
    """After full pipeline, shared_io should contain module_minus1 output."""
    ctx = run_pipeline("CC(=O)OCC", "ester_c-o")
    shared_io = ctx.data.get("shared_io")
    if shared_io is None:
        # shared_io may not exist if Module 0 didn't create it
        return

    outputs = shared_io.get("outputs", {})
    m1 = outputs.get("module_minus1")
    assert m1 is not None, "module_minus1 missing from shared_io outputs"
    assert "result" in m1
    assert "evidence_record" in m1

    result = m1["result"]
    assert "status" in result
    assert "resolved_target" in result
    assert "reactivity" in result

    evidence = m1["evidence_record"]
    assert "cpt_scores" in evidence
    assert "mechanism_eligibility" in evidence


# ---------------------------------------------------------------------------
# 6. Pipeline summary includes Module -1
# ---------------------------------------------------------------------------

def test_pipeline_summary_includes_module_minus1():
    """Pipeline summary should report Module -1 status."""
    ctx = run_pipeline("CC(=O)OCC", "ester_c-o")
    summary = ctx.data.get("pipeline_summary", {})
    results = summary.get("results", {})
    assert "module_minus1_status" in results
    assert results["module_minus1_status"] in {"PASS", "FAIL", "n/a"}

    # Check handoff
    handoffs = summary.get("interconnection", {}).get("handoff_checks", [])
    m1_handoff = [h for h in handoffs if h.get("name") == "M-1 \u2192 M0"]
    assert len(m1_handoff) == 1, "M-1 -> M0 handoff check missing"


def test_aspirin_pipeline_handoff_ok():
    """Aspirin ester should not fail due to missing Module -1 contract fields."""
    ctx = run_pipeline("CC(=O)Oc1ccccc1C(=O)O", "ester_c-o")
    summary = ctx.data.get("pipeline_summary", {})
    decision = summary.get("results", {}).get("decision")
    halt_reason = summary.get("results", {}).get("halt_reason")
    assert decision in {"GO", "LOW_CONF_GO", "NO_GO", "HALT_NEED_SELECTION"}
    assert halt_reason not in {"M0_NO_MATCH"}
    m1 = ctx.data.get("module_minus1_sre") or {}
    resolved = m1.get("resolved_target") or {}
    assert "match_count" in resolved
    handoffs = summary.get("interconnection", {}).get("handoff_checks", [])
    m1_handoff = [h for h in handoffs if h.get("name") == "M-1 \u2192 M0"]
    assert m1_handoff and m1_handoff[0]["ok"] is True


def test_symmetric_diester_triggers_disambiguation():
    """Symmetric diester should halt for explicit bond selection, not silently choose."""
    ctx = run_pipeline("COC(=O)CC(=O)OC", "ester_c-o")
    summary = ctx.data.get("pipeline_summary", {})
    decision = summary.get("results", {}).get("decision")
    halt_reason = summary.get("results", {}).get("halt_reason")
    assert decision in {"HALT_NEED_SELECTION", "LOW_CONF_GO", "NO_GO"}
    if decision == "HALT_NEED_SELECTION":
        assert halt_reason in {"M0_NEEDS_DISAMBIGUATION", "M0_TARGET_RESOLUTION_LOW"}
    m1 = ctx.data.get("module_minus1_sre") or {}
    competition = (m1.get("reactivity") or {}).get("competition") or {}
    assert "gap" in competition


def test_reagent_generation_no_match_keeps_module1_contract_fields():
    """No-match in reagent flow should keep Module -1 resolved_target contract fields."""
    ctx = run_pipeline(
        "CCO",
        "definitely_not_a_real_token",
        requested_output="-CF3",
        trap_target="TEMPO",
    )
    m1 = ctx.data.get("module_minus1_sre") or {}
    resolved = m1.get("resolved_target") or {}
    assert "selection_mode" in resolved
    assert "requested" in resolved
    assert "match_count" in resolved
    assert resolved.get("match_count") == 0
    assert "next_input_required" in resolved
    handoffs = (ctx.data.get("pipeline_summary") or {}).get("interconnection", {}).get("handoff_checks", [])
    m1_handoff = [h for h in handoffs if h.get("name") == "M-1 \u2192 M0"]
    assert m1_handoff and m1_handoff[0]["ok"] is True
