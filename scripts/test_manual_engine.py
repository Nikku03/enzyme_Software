from __future__ import annotations

from enzyme_software.manual_engine_eval import build_console_report, demo_benchmark_cases, evaluate_cases


def main() -> None:
    cases = demo_benchmark_cases()[:3]
    report = evaluate_cases(cases, repeat=1)
    print(build_console_report(report))


if __name__ == "__main__":
    main()
