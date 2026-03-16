from __future__ import annotations

from pathlib import Path
import tempfile

from enzyme_software.domain import ConditionProfile, ExperimentRecord, ReactionTask
from enzyme_software.mathcore.bayes_dag_router import BayesianDAGRouter


def test_bayesian_router_prefers_supported_route():
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "router_state.json"
        router = BayesianDAGRouter(state_path=state_path, prior_alpha=2.0, prior_beta=2.0)

        records = []
        for _ in range(24):
            records.append(
                ExperimentRecord(
                    reaction_task_fingerprint="amide_task",
                    condition_profile=ConditionProfile(pH=7.0, temperature_K=298.0),
                    candidate_fingerprint="cand_a",
                    observed_success=1.0,
                    route="ROUTE_A",
                    substrate_bin="amide",
                    catalyst_family="unknown",
                )
            )
        for _ in range(6):
            records.append(
                ExperimentRecord(
                    reaction_task_fingerprint="amide_task",
                    condition_profile=ConditionProfile(pH=7.0, temperature_K=298.0),
                    candidate_fingerprint="cand_a",
                    observed_success=0.0,
                    route="ROUTE_A",
                    substrate_bin="amide",
                    catalyst_family="unknown",
                )
            )
        for _ in range(10):
            records.append(
                ExperimentRecord(
                    reaction_task_fingerprint="amide_task",
                    condition_profile=ConditionProfile(pH=7.0, temperature_K=298.0),
                    candidate_fingerprint="cand_b",
                    observed_success=1.0,
                    route="ROUTE_B",
                    substrate_bin="amide",
                    catalyst_family="unknown",
                )
            )
        for _ in range(20):
            records.append(
                ExperimentRecord(
                    reaction_task_fingerprint="amide_task",
                    condition_profile=ConditionProfile(pH=7.0, temperature_K=298.0),
                    candidate_fingerprint="cand_b",
                    observed_success=0.0,
                    route="ROUTE_B",
                    substrate_bin="amide",
                    catalyst_family="unknown",
                )
            )

        router.update_from_records(records)

        task = ReactionTask(
            bond_to_break_or_form="amide_C-N",
            substrates=["CC(=O)NC"],
        )
        prediction = router.predict(
            task=task,
            candidates=[],
            conditions=ConditionProfile(pH=7.0, temperature_K=298.0),
            routes=["ROUTE_A", "ROUTE_B"],
        )

        assert prediction["chosen_route"] == "ROUTE_A"
        posteriors = {item["route"]: item for item in prediction["route_posteriors"]}
        assert posteriors["ROUTE_A"]["posterior"] > posteriors["ROUTE_B"]["posterior"]
        assert prediction["confidence_label"] == "High"
