# Enzyme Software Pipeline

This project scaffolds a modular pipeline for enzyme design and validation.

Pipeline overview:

INPUT
  -> SMILES + target bond
     -> Module 0 - Strategy Router (RDKit)
     -> Module 1 - TopoGate (Tunnel + Reachability)
     -> Module 2 - Reactivity Hub (QM-lite)
     -> Module 3 - Constraint-Gated Design
     -> Module 4 - Sequence Optimization
     -> Module 5 - Validation (Docking + MD)
     -> Module 6 - DNA Output + Report

## Quick start

1) Create a virtual environment and install the package:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

2) Run the CLI (target bond can be element pair like `C-O` or atom indices like `3-7`):

```bash
enzyme --smiles "CCO" --target-bond "C-O"
```

Interactive terminal UI:

```bash
enzyme --ui
```

Local web UI (loads in your browser at http://127.0.0.1:8000):

```bash
enzyme --web
```

Desktop app (Electron wrapper for the local web UI):

```bash
cd desktop
npm install
npm run start
```

Optional constraints:

```bash
enzyme --smiles "CCO" --target-bond "3-7" --ph-min 6.5 --ph-max 8.0 --forbid-metals
```

## Notes

- Module 0 includes a rule-based router with optional RDKit parsing if installed.
- External tools such as docking engines and MD packages are not wired yet.
- Modules 1-6 still return stub payloads in the pipeline context.
