from __future__ import annotations

from enzyme_software.mathcore.eyring import sample_k_pred


def test_k_pred_ci90_bounds():
    stats = sample_k_pred(18.0, 0.6, 30.0, n=256, seed=11)
    low, high = stats["ci90"]
    assert low <= stats["mean"] <= high


def test_k_pred_ci90_widens_with_std():
    tight = sample_k_pred(18.0, 0.2, 30.0, n=256, seed=11)
    wide = sample_k_pred(18.0, 1.2, 30.0, n=256, seed=11)
    tight_width = tight["ci90"][1] - tight["ci90"][0]
    wide_width = wide["ci90"][1] - wide["ci90"][0]
    assert wide_width >= tight_width


def test_temperature_increases_k_pred():
    low_temp = sample_k_pred(18.0, 0.0, 25.0, n=64, seed=7)
    high_temp = sample_k_pred(18.0, 0.0, 40.0, n=64, seed=7)
    assert high_temp["mean"] > low_temp["mean"]
