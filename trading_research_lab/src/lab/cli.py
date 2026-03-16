from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import List

from lab import ROOT, get_artifacts_root, load_config
from lab.backtest.evaluator import evaluate_strategy, ensure_evaluator_seal, write_metrics_json
from lab.data.vault import DataVault
from lab.loop_controller import LoopController
from lab.reporting.leaderboard import load_leaderboard, top_entries, update_leaderboard
from lab.research.extractors import filter_dates, parse_stooq_csv, resample_timeframe
from lab.research.source_registry import REGISTRY
from lab.research.web_client import fetch_url
from lab.strategies.generator import build_strategy


def _vault_from_config(config) -> DataVault:
    root = ROOT / config["data_vault"]["root"]
    return DataVault(root=root, processed_format=config["data_vault"]["processed_format"])


def cmd_fetch(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    vault = _vault_from_config(config)
    source = REGISTRY.get(args.source)
    if not source:
        raise SystemExit(f"Unknown source: {args.source}")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    dataset_ids: List[str] = []
    for symbol in symbols:
        url = source.build_download_url(symbol)
        content = fetch_url(url)

        def parser(payload: bytes, start=args.start, end=args.end):
            return filter_dates(parse_stooq_csv(payload), start=start, end=end)

        metadata = vault.ingest(
            source_url=url,
            content=content,
            parser=parser,
            symbol=symbol,
            file_ext="csv",
        )
        dataset_ids.append(metadata.dataset_id)
        print(f"{symbol}: {metadata.dataset_id}")

    if not dataset_ids:
        print("No datasets fetched.")


def cmd_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    vault = _vault_from_config(config)
    result = vault.validate(args.dataset)
    print(json.dumps(result, indent=2))


def cmd_backtest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    ensure_evaluator_seal(
        artifacts_root=get_artifacts_root(),
        maintenance_mode=config["lab"].get("maintenance_mode", False),
    )
    vault = _vault_from_config(config)
    df = vault.load(args.dataset)
    timeframe = args.timeframe or config.get("timeframes", ["daily"])[0]
    df = resample_timeframe(df, timeframe)

    strategy = build_strategy(args.strategy, config)
    payload = evaluate_strategy(
        df=df,
        strategy=strategy,
        config=config,
        dataset_id=args.dataset,
        walkforward=args.wf,
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    metrics_path = (
        get_artifacts_root()
        / "backtests"
        / args.strategy
        / f"run_{ts}.json"
    )
    write_metrics_json(metrics_path, payload)

    leaderboard_path = get_artifacts_root() / "leaderboards" / "leaderboard.json"
    update_leaderboard(leaderboard_path, payload, metrics_path, config["scoring"])

    print(f"Backtest written to {metrics_path}")


def cmd_run_loop(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    controller = LoopController(config=config)
    controller.run(args.iterations)
    print(f"Loop complete for {args.iterations} iterations.")


def cmd_leaderboard(args: argparse.Namespace) -> None:
    leaderboard_path = get_artifacts_root() / "leaderboards" / "leaderboard.json"
    entries = load_leaderboard(leaderboard_path)
    top = top_entries(entries, args.top)
    print(json.dumps(top, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading Research Lab CLI")
    parser.add_argument("--config", default=None, help="Path to lab_config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch")
    fetch_parser.add_argument("--source", required=True)
    fetch_parser.add_argument("--symbols", required=True)
    fetch_parser.add_argument("--start", default=None)
    fetch_parser.add_argument("--end", default=None)
    fetch_parser.set_defaults(func=cmd_fetch)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset", required=True)
    validate_parser.set_defaults(func=cmd_validate)

    backtest_parser = subparsers.add_parser("backtest")
    backtest_parser.add_argument("--strategy", required=True)
    backtest_parser.add_argument("--dataset", required=True)
    backtest_parser.add_argument("--wf", action="store_true", help="Enable walk-forward")
    backtest_parser.add_argument("--timeframe", default=None)
    backtest_parser.set_defaults(func=cmd_backtest)

    loop_parser = subparsers.add_parser("run-loop")
    loop_parser.add_argument("--iterations", type=int, default=10)
    loop_parser.set_defaults(func=cmd_run_loop)

    leaderboard_parser = subparsers.add_parser("leaderboard")
    leaderboard_parser.add_argument("--top", type=int, default=10)
    leaderboard_parser.set_defaults(func=cmd_leaderboard)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
