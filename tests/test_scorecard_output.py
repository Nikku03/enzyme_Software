from enzyme_software.pipeline import run_pipeline


def test_scorecards_present_and_defined():
    ctx = run_pipeline("CCO", "C-O")
    job_card = ctx.data.get("job_card") or {}
    module0 = ctx.data.get("module0_strategy_router") or {}
    module1 = ctx.data.get("module1_topogate") or {}
    module2 = ctx.data.get("module2_active_site_refinement") or {}
    module3 = ctx.data.get("module3_experiment_designer") or {}

    scorecards = [
        job_card.get("scorecard") or module0.get("scorecard"),
        module1.get("scorecard"),
        module2.get("scorecard"),
        module3.get("scorecard"),
    ]
    assert all(scorecards), "Each module should emit a scorecard."

    for card in scorecards:
        metrics = card.get("metrics") or []
        assert metrics, "Scorecard should contain metrics."
        for metric in metrics:
            assert metric.get("definition"), "Metric definition should be non-empty."
            ci90 = metric.get("ci90")
            if ci90 is not None:
                assert len(ci90) == 2
