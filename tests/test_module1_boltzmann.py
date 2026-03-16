from __future__ import annotations

from enzyme_software.modules.module1_topogate import (
    _boltzmann_weights,
    _ensemble_metrics,
    compute_boltzmann_weight,
)


def test_boltzmann_weights_ordered():
    energies = [0.2, 0.4, 0.6]
    weights = _boltzmann_weights(energies, temperature=0.15)
    assert abs(sum(weights) - 1.0) < 1e-6
    assert weights[0] > weights[1] > weights[2]


def test_ci_band_narrows_with_more_samples():
    weights = {"access": 0.35, "reach": 0.45, "retention": 0.2}
    energy_weights = {"access": 0.35, "reach": 0.45, "retention": 0.2, "topology": 0.25}
    small = _ensemble_metrics(
        "scaffold_test",
        0.72,
        0.61,
        0.52,
        0.78,
        weights,
        sample_count=16,
        energy_weights=energy_weights,
        temperature=0.15,
    )
    large = _ensemble_metrics(
        "scaffold_test",
        0.72,
        0.61,
        0.52,
        0.78,
        weights,
        sample_count=64,
        energy_weights=energy_weights,
        temperature=0.15,
    )
    small_width = small["score_ci90"]["total"][1] - small["score_ci90"]["total"][0]
    large_width = large["score_ci90"]["total"][1] - large["score_ci90"]["total"][0]
    assert large_width <= small_width + 1e-6


def test_compute_boltzmann_weight_monotonic():
    temp_k = 298.15
    low = compute_boltzmann_weight(0.0, temp_k)
    high = compute_boltzmann_weight(10.0, temp_k)
    assert low > high


def test_compute_boltzmann_weight_stable():
    temp_k = 298.15
    weights = [
        compute_boltzmann_weight(delta, temp_k) for delta in (0.0, 5.0, 10.0)
    ]
    assert all(weight > 0.0 for weight in weights)
    assert weights[0] > weights[1] > weights[2]
