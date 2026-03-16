from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Tuple


def estimate_uncertainty(probability: float, missing_count: int) -> float:
    base = 0.12
    penalty = min(0.25, 0.03 * max(0, missing_count))
    uncertainty = min(0.45, base + penalty + (0.5 - abs(0.5 - probability)) * 0.1)
    return round(uncertainty, 3)


def confidence_interval(probability: float, uncertainty: float) -> Tuple[float, float]:
    low = max(0.0, probability - uncertainty)
    high = min(1.0, probability + uncertainty)
    return round(low, 3), round(high, 3)


def calibrate_probability(probability: float, uncertainty: float) -> float:
    calibration = max(0.0, probability - uncertainty * 0.35)
    return round(min(1.0, calibration), 3)


class ProbabilityCalibrator:
    def __init__(
        self,
        l2: float = 1.0,
        max_iter: int = 400,
        lr: float = 0.1,
        min_samples: int = 20,
    ) -> None:
        self.l2 = l2
        self.max_iter = max_iter
        self.lr = lr
        self.min_samples = min_samples
        self.a = 1.0
        self.b = 0.0
        self._fitted = False
        self._identity = True

    def fit(self, samples: List[Tuple[float, int]]) -> None:
        if len(samples) < self.min_samples:
            self._identity = True
            self._fitted = False
            return
        self._identity = False
        a = self.a
        b = self.b
        for _ in range(self.max_iter):
            grad_a = 0.0
            grad_b = 0.0
            for p_raw, y_true in samples:
                p_raw = _clamp(p_raw)
                x = _logit(p_raw)
                pred = _sigmoid(a * x + b)
                err = pred - float(y_true)
                grad_a += err * x
                grad_b += err
            count = float(len(samples))
            grad_a = grad_a / count + self.l2 * a
            grad_b = grad_b / count
            a -= self.lr * grad_a
            b -= self.lr * grad_b
        self.a = a
        self.b = b
        self._fitted = True

    def predict(self, p_raw: float) -> float:
        p_raw = _clamp(p_raw)
        if self._identity or not self._fitted:
            return p_raw
        x = _logit(p_raw)
        return _clamp(_sigmoid(self.a * x + self.b))


def beta_credible_interval(
    probability: float,
    n_eff: float,
    ci: float = 0.9,
) -> Tuple[float, float]:
    probability = _clamp(probability)
    n_eff = max(2.0, float(n_eff))
    alpha = max(1e-6, probability * n_eff)
    beta = max(1e-6, (1.0 - probability) * n_eff)
    tail = (1.0 - ci) / 2.0
    low = beta_inv_cdf(tail, alpha, beta)
    high = beta_inv_cdf(1.0 - tail, alpha, beta)
    return round(low, 3), round(high, 3)


def beta_inv_cdf(target: float, alpha: float, beta: float) -> float:
    target = min(max(target, 0.0), 1.0)
    low = 0.0
    high = 1.0
    for _ in range(64):
        mid = (low + high) / 2.0
        cdf = beta_cdf(mid, alpha, beta)
        if cdf < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def beta_cdf(x: float, alpha: float, beta: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    front = math.exp(
        math.log(x) * alpha + math.log(1 - x) * beta - ln_beta
    )
    if x < (alpha + 1.0) / (alpha + beta + 2.0):
        return front * _betacf(alpha, beta, x) / alpha
    return 1.0 - front * _betacf(beta, alpha, 1.0 - x) / beta


def _betacf(alpha: float, beta: float, x: float) -> float:
    max_iter = 200
    eps = 3e-7
    fpmin = 1e-30
    am = 1.0
    bm = 1.0
    az = 1.0
    qab = alpha + beta
    qap = alpha + 1.0
    qam = alpha - 1.0
    bz = 1.0 - qab * x / qap
    if abs(bz) < fpmin:
        bz = fpmin
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (beta - em) * x / ((qam + tem) * (alpha + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = -(alpha + em) * (qab + em) * x / ((alpha + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        if abs(bpp) < fpmin:
            bpp = fpmin
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if abs(app - az) < eps * abs(az):
            break
    return az


def _sigmoid(value: float) -> float:
    if value < -35:
        return 0.0
    if value > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    probability = _clamp(probability)
    return math.log(probability / (1.0 - probability))


def _clamp(probability: float) -> float:
    return min(max(float(probability), 1e-6), 1.0 - 1e-6)


@dataclass
class ProbabilityEstimate:
    p_raw: float
    p_cal: float
    ci90: Tuple[float, float]
    n_eff: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_raw": float(self.p_raw),
            "p_cal": float(self.p_cal),
            "ci90": [float(self.ci90[0]), float(self.ci90[1])],
            "n_eff": float(self.n_eff),
        }


@dataclass
class DistributionEstimate:
    mean: float
    std: float
    ci90: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "ci90": [float(self.ci90[0]), float(self.ci90[1])],
        }


@dataclass
class QCReport:
    status: str
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
        }


def distribution_from_ci(mean: float, ci90: Tuple[float, float]) -> DistributionEstimate:
    low, high = ci90
    span = max(0.0, float(high) - float(low))
    std = span / (2.0 * 1.645) if span else 0.0
    return DistributionEstimate(mean=float(mean), std=float(std), ci90=(float(low), float(high)))


def validate_math_contract(module_output: Dict[str, Any]) -> List[str]:
    violations: List[str] = []
    contract = module_output.get("math_contract")
    if not isinstance(contract, dict):
        violations.append("missing math_contract")
        return violations
    confidence = contract.get("confidence")
    if not isinstance(confidence, dict):
        violations.append("math_contract.confidence missing")
    else:
        for key in ("p_raw", "p_cal", "ci90", "n_eff"):
            if key not in confidence:
                violations.append(f"confidence missing {key}")
        if isinstance(confidence.get("ci90"), list) and len(confidence["ci90"]) != 2:
            violations.append("confidence.ci90 length != 2")
    predictions = contract.get("predictions")
    if not isinstance(predictions, dict):
        violations.append("math_contract.predictions missing")
    else:
        for name, prediction in predictions.items():
            if not isinstance(prediction, dict):
                violations.append(f"prediction {name} invalid")
                continue
            for key in ("mean", "std", "ci90"):
                if key not in prediction:
                    violations.append(f"prediction {name} missing {key}")
            if isinstance(prediction.get("ci90"), list) and len(prediction["ci90"]) != 2:
                violations.append(f"prediction {name} ci90 length != 2")
    qc = contract.get("qc")
    if not isinstance(qc, dict):
        violations.append("math_contract.qc missing")
    else:
        if "status" not in qc:
            violations.append("qc missing status")
        if "reasons" not in qc:
            violations.append("qc missing reasons")
        if "metrics" not in qc:
            violations.append("qc missing metrics")
    return violations


def weighted_mean(values: List[float], weights: List[float]) -> float:
    if not values or not weights:
        return 0.0
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return sum(values) / len(values)
    return sum(val * weight for val, weight in zip(values, weights)) / total_weight


def weighted_std(values: List[float], weights: List[float], mean: float | None = None) -> float:
    if not values or not weights:
        return 0.0
    if mean is None:
        mean = weighted_mean(values, weights)
    total_weight = sum(weights)
    if total_weight <= 0.0:
        variance = sum((val - mean) ** 2 for val in values) / len(values)
        return math.sqrt(variance)
    variance = sum(weight * (val - mean) ** 2 for val, weight in zip(values, weights)) / total_weight
    return math.sqrt(variance)


def weighted_quantile(values: List[float], weights: List[float], quantile: float) -> float:
    if not values or not weights:
        return 0.0
    quantile = max(0.0, min(1.0, quantile))
    paired = sorted(zip(values, weights), key=lambda item: item[0])
    total_weight = sum(weight for _, weight in paired)
    if total_weight <= 0.0:
        index = int(round(quantile * (len(values) - 1)))
        return paired[index][0]
    cumulative = 0.0
    for value, weight in paired:
        cumulative += weight
        if cumulative / total_weight >= quantile:
            return value
    return paired[-1][0]


def percentile(values: List[float], quantile: float) -> float:
    if not values:
        return 0.0
    quantile = max(0.0, min(1.0, quantile))
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = quantile * (len(ordered) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def sigmoid(value: float) -> float:
    if value < -35:
        return 0.0
    if value > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-value))


def bernoulli_entropy(probability: float) -> float:
    probability = max(1e-6, min(1.0 - 1e-6, float(probability)))
    return -(probability * math.log(probability) + (1.0 - probability) * math.log(1.0 - probability))


def beta_entropy(alpha: float, beta: float) -> float:
    alpha = max(1e-6, float(alpha))
    beta = max(1e-6, float(beta))
    ln_beta = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return ln_beta - (alpha - 1.0) * _digamma(alpha) - (beta - 1.0) * _digamma(beta) + (
        alpha + beta - 2.0
    ) * _digamma(alpha + beta)


def _digamma(value: float) -> float:
    x = float(value)
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += (
        math.log(x)
        - 0.5 * inv
        - inv2 / 12.0
        + (inv2 * inv2) / 120.0
        - (inv2 * inv2 * inv2) / 252.0
    )
    return result
