# Trading Research Lab

A minimal, robust lab for legal data acquisition, versioned datasets, sealed backtests, and anti-looped strategy iteration.

## Quick Start

```bash
cd trading_research_lab
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### Fetch datasets (Stooq)

```bash
python -m lab.cli fetch --source stooq --symbols SPY,QQQ --start 2010-01-01
```

### Validate a dataset

```bash
python -m lab.cli validate --dataset <dataset_id>
```

### Run a backtest

```bash
python -m lab.cli backtest --strategy sma_crossover --dataset <dataset_id> --wf
```

### Run the research loop

```bash
python -m lab.cli run-loop --iterations 10
```

### View leaderboard

```bash
python -m lab.cli leaderboard --top 10
```

## Truth + Sanity Constitution

- No fabrication: unknown data is marked as `UNKNOWN` and logged for fetch/verify.
- Metrics are only printed from evaluator JSON outputs.
- Two stagnant iterations without new artifacts force a pivot.
- Evaluator is sealed; version changes require maintenance mode.

## Artifacts

Artifacts are written under `artifacts/`:

- `artifacts/research_notes/YYYY-MM-DD_iterN.md`
- `artifacts/backtests/<strategy>/run_YYYYMMDD_HHMM.json`
- `artifacts/datasets/meta/<dataset_id>.json`

Example artifact templates are included with `UNKNOWN` fields and are replaced on actual runs.

## Extending

- Add new sources in `src/lab/research/source_registry.py`.
- Add strategies in `src/lab/strategies/library/` and register them in `src/lab/strategies/generator.py`.
- Adjust scoring and costs in `config/lab_config.yaml`.
