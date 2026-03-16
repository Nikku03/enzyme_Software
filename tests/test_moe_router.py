from __future__ import annotations

from enzyme_software.mechanism_registry import resolve_mechanism
from enzyme_software.moe_router import MoERouter


def test_moe_router_does_not_flip_route_without_fork() -> None:
    mechanism = resolve_mechanism("serine_hydrolase")
    shared_state = {
        "physics": {
            "priors": {
                "serine_hydrolase": {"route_prior_any_activity": 0.6},
                "metallo_esterase": {"route_prior_any_activity": 0.8},
            }
        },
        "scoring": {"ledger": {"module0": {"data_support": 0.0}}},
        "mechanism": {"mismatch": {"penalty": 0.0}},
    }
    features = {"route_family_serine_hydrolase": 1.0, "route_family_metallo_esterase": 0.0}
    router = MoERouter()
    result = router.evaluate(shared_state, mechanism, features, top_k=2, fork_hypotheses=False)
    decision = result.get("decision") or {}
    assert decision.get("route_id") == "serine_hydrolase"
