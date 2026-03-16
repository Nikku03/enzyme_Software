from enzyme_software.physicscore import expected_turnovers, detectability_probability


def test_detectability_prior_low_k():
    turnovers = expected_turnovers(1e-6, 3600.0)
    prior = detectability_probability(turnovers, n_required=3.0)
    assert prior < 0.2


def test_detectability_prior_high_k():
    turnovers = expected_turnovers(1e-2, 3600.0)
    prior = detectability_probability(turnovers, n_required=3.0)
    assert prior > 0.8
