from enzyme_software.mathcore.uncertainty import (
    DistributionEstimate,
    ProbabilityEstimate,
    QCReport,
    validate_math_contract,
)


def test_validate_math_contract_ok():
    output = {
        "math_contract": {
            "confidence": ProbabilityEstimate(
                p_raw=0.6, p_cal=0.62, ci90=(0.4, 0.8), n_eff=12.0
            ).to_dict(),
            "predictions": {
                "score": DistributionEstimate(
                    mean=0.6, std=0.1, ci90=(0.45, 0.75)
                ).to_dict()
            },
            "qc": QCReport(status="N/A", reasons=[], metrics={}).to_dict(),
        }
    }
    assert validate_math_contract(output) == []


def test_validate_math_contract_missing():
    output = {}
    violations = validate_math_contract(output)
    assert "missing math_contract" in violations
