from enzyme_software.pipeline import run_pipeline


def test_run_pipeline_smoke():
    ctx = run_pipeline("CCO", "C-O")
    assert ctx.smiles == "CCO"
    assert ctx.target_bond == "C-O"
    assert "module0_strategy_router" in ctx.data
    job_card = ctx.data.get("job_card") or {}
    assert job_card["decision"] in {"GO", "LOW_CONF_GO", "NO_GO", "HALT_NEED_SELECTION"}
    assert "bond_context" in job_card
    assert "module1_topogate" in ctx.data
    module1 = ctx.data["module1_topogate"]
    assert module1.get("status") in {"PASS", "FAIL"}
    assert "module3_experiment_designer" in ctx.data
    module3 = ctx.data["module3_experiment_designer"]
    protocol_card = module3.get("protocol_card") or {}
    assert protocol_card
    assert len(protocol_card.get("arms", [])) == 5
    controls = protocol_card.get("controls") or {}
    assert controls.get("negative_control_arm_id")
