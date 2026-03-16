"""Mathcore provides shared mathematical engines for Modules 0-2."""

from enzyme_software.mathcore.bayes_dag import predict as bayes_predict
from enzyme_software.mathcore.bayes_dag_router import BayesianDAGRouter
from enzyme_software.mathcore.features import FeatureVector, extract_features
from enzyme_software.mathcore.eyring import (
    delta_g_with_conditions,
    estimate_delta_g,
    rate_constant,
    sample_k_pred,
    search_optimum_conditions,
)
from enzyme_software.mathcore.persistent_homology import (
    compute_signature,
    score_signature,
)
from enzyme_software.mathcore.uncertainty import (
    DistributionEstimate,
    ProbabilityEstimate,
    QCReport,
    ProbabilityCalibrator,
    beta_credible_interval,
    distribution_from_ci,
    calibrate_probability,
    confidence_interval,
    estimate_uncertainty,
    percentile,
    validate_math_contract,
)
from enzyme_software.mathcore.telemetry import record_event

__all__ = [
    "FeatureVector",
    "extract_features",
    "bayes_predict",
    "BayesianDAGRouter",
    "delta_g_with_conditions",
    "estimate_delta_g",
    "rate_constant",
    "sample_k_pred",
    "search_optimum_conditions",
    "compute_signature",
    "score_signature",
    "ProbabilityCalibrator",
    "ProbabilityEstimate",
    "DistributionEstimate",
    "QCReport",
    "beta_credible_interval",
    "distribution_from_ci",
    "calibrate_probability",
    "confidence_interval",
    "estimate_uncertainty",
    "percentile",
    "validate_math_contract",
    "record_event",
]
