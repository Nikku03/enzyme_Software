from __future__ import annotations

import argparse
import json
from pathlib import Path

from enzyme_software.manual_engine_eval import (
    build_console_report,
    build_markdown_report,
    evaluate_cases,
    load_benchmark_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the manual mechanistic metabolism engine")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = load_benchmark_cases(args.benchmark, max_cases=args.max_cases)
    report = evaluate_cases(cases, repeat=args.repeat)
    print(build_console_report(report))
    if args.verbose:
        print(json.dumps(report, indent=2))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_markdown_report(report))


if __name__ == "__main__":
    main()
