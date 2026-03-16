from __future__ import annotations

from enzyme_software.physicscore import (
    eyring_rate_constant,
    henderson_hasselbalch_fraction_deprotonated,
    kj_to_j,
    physics_prior_success_probability,
)


def test_eyring_monotonicity():
    temp_k = 298.15
    k_low_barrier = eyring_rate_constant(kj_to_j(60.0), temp_k)
    k_high_barrier = eyring_rate_constant(kj_to_j(90.0), temp_k)
    assert k_low_barrier > k_high_barrier


def test_eyring_prefactor_order():
    temp_k = 298.15
    k0 = eyring_rate_constant(0.0, temp_k)
    assert 1.0e12 < k0 < 1.0e13


def test_henderson_hasselbalch_midpoint():
    frac = henderson_hasselbalch_fraction_deprotonated(7.0, 7.0)
    assert abs(frac - 0.5) < 1e-6


def test_physics_prior_success_probability_bounds():
    temp_k = 298.15
    p_low = physics_prior_success_probability(kj_to_j(120.0), temp_k)
    p_high = physics_prior_success_probability(kj_to_j(60.0), temp_k)
    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
    assert p_high > p_low
