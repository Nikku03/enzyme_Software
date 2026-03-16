from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from enzyme_software.context import OperationalConstraints
from enzyme_software.input_parsing import load_smiles_from_file, parse_bond_indices
from enzyme_software.pipeline import run_pipeline
from enzyme_software.reporting import (
    render_debug,
    render_debug_report,
    render_demo,
    render_pretty,
    render_scientific_report,
    render_scientist,
)
from enzyme_software.tui import run_terminal_ui
from enzyme_software.web_app import run_web_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the enzyme design pipeline scaffold",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--ui", action="store_true", help="Launch the terminal UI")
    mode_group.add_argument("--web", action="store_true", help="Launch the local web UI")
    parser.add_argument("--web-host", default="127.0.0.1", help="Web UI host")
    parser.add_argument("--web-port", type=int, default=8000, help="Web UI port")
    parser.add_argument("--smiles", help="Input SMILES string")
    parser.add_argument("--smiles-file", help="Path to SMILES/SDF/MOL file")
    parser.add_argument("--target-bond", help="Target bond identifier")
    parser.add_argument(
        "--target-bond-idx",
        help="Target bond atom indices (e.g., \"1,3\" or \"[1,3]\")",
    )
    request_group = parser.add_argument_group("request")
    request_group.add_argument(
        "--requested-output",
        help="Desired output (e.g., -CF3, CF3 radical, carbene)",
    )
    request_group.add_argument(
        "--trap-target",
        help="Trap/acceptor definition for reagent generation",
    )
    constraint_group = parser.add_argument_group("constraints")
    constraint_group.add_argument("--ph-min", type=float, help="Minimum pH")
    constraint_group.add_argument("--ph-max", type=float, help="Maximum pH")
    constraint_group.add_argument(
        "--temperature-c",
        type=float,
        help="Operating temperature in Celsius",
    )
    metals_group = constraint_group.add_mutually_exclusive_group()
    metals_group.add_argument(
        "--allow-metals",
        action="store_true",
        help="Allow metal cofactors",
    )
    metals_group.add_argument(
        "--forbid-metals",
        action="store_true",
        help="Disallow metal cofactors",
    )
    oxidation_group = constraint_group.add_mutually_exclusive_group()
    oxidation_group.add_argument(
        "--allow-oxidation",
        action="store_true",
        help="Allow oxidative mechanisms",
    )
    oxidation_group.add_argument(
        "--forbid-oxidation",
        action="store_true",
        help="Disallow oxidative mechanisms",
    )
    constraint_group.add_argument("--host", help="Host organism")
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Render a compact pretty report (no raw JSON).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of the scientific report",
    )
    parser.add_argument(
        "--json-pretty",
        action="store_true",
        help="Pretty-print JSON output (used with --json).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Append the debug report after the scientific report",
    )
    parser.add_argument(
        "--view",
        choices=("demo", "pretty", "scientist", "debug"),
        default="scientist",
        help="Select report view (demo, pretty, scientist, debug).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.web:
        return run_web_app(host=args.web_host, port=args.web_port)
    if args.ui:
        return run_terminal_ui()

    smiles = args.smiles
    if args.smiles_file:
        if smiles:
            parser.error("Use either --smiles or --smiles-file, not both.")
        try:
            smiles, warnings = load_smiles_from_file(args.smiles_file)
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)

    if not smiles:
        parser.error("Missing --smiles or --smiles-file.")
    target_bond = args.target_bond
    if args.target_bond_idx:
        indices = parse_bond_indices(args.target_bond_idx)
        if indices is None:
            parser.error("Invalid --target-bond-idx format. Use \"1,3\" or \"[1,3]\".")
        target_bond = f"[{indices[0]},{indices[1]}]"

    if not target_bond:
        parser.error("Missing --target-bond or --target-bond-idx.")

    metals_allowed = None
    if args.allow_metals:
        metals_allowed = True
    elif args.forbid_metals:
        metals_allowed = False

    oxidation_allowed = None
    if args.allow_oxidation:
        oxidation_allowed = True
    elif args.forbid_oxidation:
        oxidation_allowed = False

    constraints = OperationalConstraints(
        ph_min=args.ph_min,
        ph_max=args.ph_max,
        temperature_c=args.temperature_c,
        metals_allowed=metals_allowed,
        oxidation_allowed=oxidation_allowed,
        host=args.host,
    )

    ctx = run_pipeline(
        smiles,
        target_bond,
        requested_output=args.requested_output,
        trap_target=args.trap_target,
        constraints=constraints,
    )
    ctx_dict = ctx.to_dict()
    if args.json:
        indent = 2 if args.json_pretty else None
        output = json.dumps(ctx_dict, indent=indent)
        print(output)
        return 0

    view = args.view
    if args.pretty:
        view = "pretty"
    if args.debug:
        view = "debug"
    if view == "demo":
        report = render_demo(ctx_dict)
    elif view == "pretty":
        report = render_pretty(ctx_dict)
    elif view == "debug":
        report = render_debug(ctx_dict)
    else:
        report = render_scientist(ctx_dict)
    if args.debug and view != "debug":
        report = f"{report}\n\n{render_debug_report(ctx_dict)}"
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
