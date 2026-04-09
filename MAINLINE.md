# Mainline Migration Summary

## Active

The repo now has one clean active path:

1. `scripts/build_master_dataset.py`
2. `scripts/build_regime_subsets.py`
3. `scripts/build_mainline_splits.py`
4. `scripts/train_mainline_som.py`
5. `scripts/evaluate_mainline_benchmarks.py`

This path keeps the good multi-regime dataset work and makes shortlist-first training the default modeling direction.

## Archived In Practice

Older branch-heavy trainers, routing variants, hard-source-specialized paths, and one-off benchmark scripts remain in the repo for reproducibility, but are legacy.

See [archive/legacy_manifest.md](/Users/deepika/Desktop/books/enzyme_software/archive/legacy_manifest.md).

## Benchmark Semantics

The active mainline does not use one blended headline metric. It reports:

- `strict_exact_benchmark`
- `tier_aware_benchmark`
- `high_confidence_benchmark`

## Mainline Training Policy

- Supervised training rows:
  - `strict_exact_clean`
  - `tiered_multisite_eval`
- Not used in first-line supervised training:
  - `broad_region_aux`
  - `conflict_audit`

## Model Policy

- shortlist carries the main optimization burden
- winner is small and local
- no active dual-winner routing
- no active hard-source-only branch
