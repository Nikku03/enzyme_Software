from enzyme_software.mathcore.uncertainty import ProbabilityCalibrator, beta_credible_interval


def test_calibrator_identity_for_small_samples():
    calibrator = ProbabilityCalibrator()
    samples = [(0.2, 0), (0.8, 1)] * 5
    calibrator.fit(samples)
    assert abs(calibrator.predict(0.33) - 0.33) < 1e-6


def test_calibrator_outputs_valid_probability():
    calibrator = ProbabilityCalibrator()
    samples = [(0.1, 0), (0.2, 0), (0.8, 1), (0.9, 1)] * 8
    calibrator.fit(samples)
    calibrated = calibrator.predict(0.6)
    assert 0.0 < calibrated < 1.0


def test_credible_interval_brackets_probability():
    low, high = beta_credible_interval(0.62, 12.0)
    assert low <= 0.62 <= high


def test_higher_n_eff_narrows_interval():
    low_small, high_small = beta_credible_interval(0.55, 6.0)
    low_large, high_large = beta_credible_interval(0.55, 40.0)
    assert (high_large - low_large) < (high_small - low_small)
