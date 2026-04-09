# Legacy Manifest

## Active Mainline

- `src/enzyme_software/mainline/config.py`
- `src/enzyme_software/mainline/data/master_builder.py`
- `src/enzyme_software/mainline/data/regime_builder.py`
- `src/enzyme_software/mainline/data/split_builder.py`
- `src/enzyme_software/mainline/models/winner.py`
- `src/enzyme_software/mainline/training/shortlist_first.py`
- `src/enzyme_software/mainline/eval/benchmarks.py`
- `scripts/build_master_dataset.py`
- `scripts/build_regime_subsets.py`
- `scripts/build_mainline_splits.py`
- `scripts/train_mainline_som.py`
- `scripts/evaluate_mainline_benchmarks.py`

## Active Lower-Level Data Tooling

- `scripts/build_all_family_merged_dataset.py`
- `scripts/build_family_downstream_subsets.py`
- `scripts/build_cyp3a4_phase12_splits.py`

These remain the underlying dataset builders used by the new wrappers.

## Legacy Trainers

These remain in-place for reproducibility, but are no longer the default active path:

- `src/enzyme_software/liquid_nn_v2/training/two_head_shortlist_winner_v2_1_trainer.py`
- `src/enzyme_software/liquid_nn_v2/training/two_head_shortlist_winner_v2_2_trainer.py`
- `src/enzyme_software/liquid_nn_v2/training/two_head_shortlist_winner_v2_3_trainer.py`
- `src/enzyme_software/liquid_nn_v2/training/two_head_shortlist_winner_v2_rebuild_boundary_reranker_trainer.py`
- `src/enzyme_software/liquid_nn_v2/training/two_head_shortlist_winner_v2_rebuild_context_features_trainer.py`
- `src/enzyme_software/liquid_nn_v2/training/two_head_shortlist_winner_v2_rebuild_dual_winner_routing_trainer.py`
- `src/enzyme_software/liquid_nn_v2/training/two_head_shortlist_winner_v2_rebuild_hard_source_finetune_trainer.py`
- `src/enzyme_software/liquid_nn_v2/training/pairwise_distilled_proposer_trainer.py`
- `src/enzyme_software/liquid_nn_v2/training/pairwise_site_tournament_trainer.py`
- `src/enzyme_software/liquid_nn_v2/training/candidate_winner_trainer.py`

## Legacy Scripts

These scripts are retained, but are no longer the recommended starting point:

- `scripts/train_hybrid_full_xtb.py`
- `scripts/colab_train_hybrid_lnn.py`
- phase-specific train/eval scripts under `scripts/train_phase*` and `scripts/evaluate_*`

## Rationale

The legacy stack grew around many winner-heavy and source-specialized experiments. The new mainline keeps the useful data pipeline, but narrows the active training path to:

1. shortlist-first learning
2. one small local winner
3. explicit benchmark separation
