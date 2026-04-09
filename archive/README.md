# Legacy Archive

This repo now has one active mainline:

- `scripts/build_master_dataset.py`
- `scripts/build_regime_subsets.py`
- `scripts/build_mainline_splits.py`
- `scripts/train_mainline_som.py`
- `scripts/evaluate_mainline_benchmarks.py`
- `src/enzyme_software/mainline/`

Everything else remains available for reproducibility, but is legacy unless it is explicitly referenced by the mainline wrappers above.

The active design principles are:

1. Raw-source ingest and merged master dataset first.
2. Regime-separated downstream subsets.
3. Explicit split artifacts.
4. Shortlist-first training.
5. Small local winner.
6. Separate strict exact, tier-aware, and high-confidence benchmarks.
