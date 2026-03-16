from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from lab import ROOT, get_artifacts_root, load_config
from lab.backtest.evaluator import evaluate_strategy, ensure_evaluator_seal, write_metrics_json
from lab.data.vault import DataVault
from lab.pivot_rules import PivotPlan, apply_pivot
from lab.reporting.leaderboard import update_leaderboard
from lab.reporting.render import write_research_note
from lab.research.extractors import filter_dates, parse_stooq_csv, resample_timeframe
from lab.research.source_registry import REGISTRY
from lab.research.web_client import build_search_urls, fetch_url
from lab.strategies.generator import build_strategy


class LoopController:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or load_config()
        self.root = get_artifacts_root()
        self.state_path = self.root / "leaderboards" / "loop_state.json"
        self.vault = DataVault(
            root=ROOT / self.config["data_vault"]["root"],
            processed_format=self.config["data_vault"]["processed_format"],
        )
        self.rng = random.Random(self.config["lab"].get("random_seed", 42))

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {
                "history": [],
                "plan": None,
                "dataset_id": None,
            }
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, state: Dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _default_plan(self) -> PivotPlan:
        return PivotPlan(
            strategy_family=self.config["strategies"]["families"][0],
            timeframe=self.config.get("timeframes", ["daily"])[0],
            universe=self.config["lab"]["default_symbols"],
            risk_sizing=self.config.get("risk", {}).get("sizing", "fixed"),
            entry_exit=self.config.get("entry_exit", {}).get("styles", ["standard"])[0],
        )

    def _ensure_dataset(self, state: Dict[str, Any]) -> Optional[str]:
        if state.get("dataset_id"):
            return state["dataset_id"]
        source_name = self.config["lab"]["default_source"]
        source = REGISTRY.get(source_name)
        if not source:
            return None
        dataset_ids = []
        for symbol in self.config["lab"]["default_symbols"]:
            url = source.build_download_url(symbol)
            content = fetch_url(url)

            def parser(payload: bytes, start=self.config["lab"]["default_start"]):
                return filter_dates(parse_stooq_csv(payload), start=start)

            metadata = self.vault.ingest(
                source_url=url,
                content=content,
                parser=parser,
                symbol=symbol,
                file_ext="csv",
            )
            dataset_ids.append(metadata.dataset_id)
        state["dataset_id"] = dataset_ids[0] if dataset_ids else None
        return state.get("dataset_id")

    def _should_pivot(self, history: List[Dict[str, Any]]) -> bool:
        limit = int(self.config.get("loop", {}).get("stagnation_limit", 2))
        if len(history) < limit:
            return False
        recent = history[-limit:]
        no_artifacts = all(entry.get("artifact_count", 0) == 0 for entry in recent)
        no_improvement = all(not entry.get("improved", False) for entry in recent)
        return no_artifacts and no_improvement

    def run(self, iterations: int) -> None:
        ensure_evaluator_seal(
            artifacts_root=self.root,
            maintenance_mode=self.config["lab"].get("maintenance_mode", False),
        )
        state = self._load_state()
        plan = (
            PivotPlan(**state["plan"]) if state.get("plan") else self._default_plan()
        )
        dataset_id = self._ensure_dataset(state)

        for iteration in range(1, iterations + 1):
            artifact_paths: List[str] = []
            source_url = None
            search_urls = build_search_urls(
                ["stooq", "csv", ",".join(plan.universe), plan.timeframe]
            )

            try:
                if dataset_id is None:
                    tasks = ["Fetch dataset from configured source."]
                    note_path = (
                        self.root
                        / "research_notes"
                        / f"{datetime.utcnow().date()}_iter{iteration}.md"
                    )
                    write_research_note(
                        note_path,
                        iteration=iteration,
                        dataset_id=None,
                        source_url=None,
                        search_urls=search_urls,
                        evaluator_json_path=None,
                        notes=["Dataset missing. Blocking backtest."],
                        tasks=tasks,
                    )
                    artifact_paths.append(str(note_path))
                    state["history"].append(
                        {
                            "iteration": iteration,
                            "artifact_count": len(artifact_paths),
                            "improved": False,
                            "plan": asdict(plan),
                            "score": None,
                        }
                    )
                    self._save_state(state)
                    continue

                meta = self.vault.get_metadata(dataset_id)
                source_url = meta.get("source_url")
                df = self.vault.load(dataset_id)
                df = resample_timeframe(df, plan.timeframe)
                strategy = build_strategy(plan.strategy_family, self.config)
                payload = evaluate_strategy(
                    df=df,
                    strategy=strategy,
                    config=self.config,
                    dataset_id=dataset_id,
                    walkforward=True,
                )

                ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
                metrics_path = (
                    self.root
                    / "backtests"
                    / plan.strategy_family
                    / f"run_{ts}.json"
                )
                write_metrics_json(metrics_path, payload)
                artifact_paths.append(str(metrics_path))

                leaderboard_path = self.root / "leaderboards" / "leaderboard.json"
                entries = update_leaderboard(
                    leaderboard_path,
                    payload,
                    metrics_path,
                    self.config["scoring"],
                )

                best_score = entries[0]["score"] if entries else 0.0
                previous_best = state.get("best_score")
                improved = previous_best is None or best_score > previous_best
                state["best_score"] = best_score

                note_path = (
                    self.root
                    / "research_notes"
                    / f"{datetime.utcnow().date()}_iter{iteration}.md"
                )
                write_research_note(
                    note_path,
                    iteration=iteration,
                    dataset_id=dataset_id,
                    source_url=source_url,
                    search_urls=search_urls,
                    evaluator_json_path=metrics_path,
                    notes=[
                        "Backtest complete. Metrics stored in evaluator JSON.",
                        f"Failure tags: {', '.join(payload.get('failure_tags', [])) or 'NONE'}",
                    ],
                    tasks=None,
                )
                artifact_paths.append(str(note_path))

                state["history"].append(
                    {
                        "iteration": iteration,
                        "artifact_count": len(artifact_paths),
                        "improved": improved,
                        "plan": asdict(plan),
                        "score": best_score,
                    }
                )
            except Exception as exc:
                note_path = (
                    self.root
                    / "research_notes"
                    / f"{datetime.utcnow().date()}_iter{iteration}.md"
                )
                write_research_note(
                    note_path,
                    iteration=iteration,
                    dataset_id=dataset_id,
                    source_url=source_url,
                    search_urls=search_urls,
                    evaluator_json_path=None,
                    notes=[f"Post-mortem: {exc}"],
                    tasks=["Investigate failure and re-run."],
                )
                artifact_paths.append(str(note_path))
                state["history"].append(
                    {
                        "iteration": iteration,
                        "artifact_count": len(artifact_paths),
                        "improved": False,
                        "plan": asdict(plan),
                        "score": None,
                    }
                )

            if self._should_pivot(state["history"]):
                plan, changed = apply_pivot(plan, self.config, self.rng)
                state["plan"] = asdict(plan)
                state["pivoted_on"] = {
                    "iteration": iteration,
                    "changed": changed,
                }
            else:
                state["plan"] = asdict(plan)

            self._save_state(state)
