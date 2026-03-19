#!/usr/bin/env bash
# train_phase_2.sh — trains full_xtb, micropattern, then meta learner
# Run after train_phase_1.sh finishes: bash train_phase_2.sh 2>&1 | tee logs/train_phase2_$(date +%Y%m%d).log
set -euo pipefail

PYTHON="/Users/deepika/anaconda3/envs/bondbreak/bin/python"
export PYTHONPATH="$(pwd)/src"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

DATASET="data/prepared_training/main5_all_models_conservative.json"
mkdir -p logs checkpoints/hybrid_full_xtb checkpoints/micropattern_xtb \
         checkpoints/meta_learner_multihead artifacts/hybrid_full_xtb \
         artifacts/micropattern_xtb artifacts/meta_learner_multihead \
         cache/meta_learner cache/meta_learner_multihead

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Step 2: hybrid_full_xtb ──────────────────────────────────────────────────
log "STEP 2/4: hybrid_full_xtb (~5 hrs)"
"$PYTHON" scripts/train_hybrid_full_xtb.py \
  --dataset                 "$DATASET" \
  --checkpoint              checkpoints/hybrid_lnn_best.pt \
  --xtb-cache-dir           cache/full_xtb \
  --train-ratio             0.70 \
  --val-ratio               0.15 \
  --epochs                  100 \
  --early-stopping-patience 100 \
  --learning-rate           5e-5 \
  --weight-decay            1e-4 \
  --batch-size              16 \
  --seed                    42 \
  --output-dir              checkpoints/hybrid_full_xtb \
  --artifact-dir            artifacts/hybrid_full_xtb \
  --compute-xtb-if-missing
log "STEP 2 done."

# ── Step 3: micropattern_xtb ─────────────────────────────────────────────────
log "STEP 3/4: micropattern_xtb (~4 hrs)"
"$PYTHON" scripts/train_micropattern_xtb.py \
  --dataset                 "$DATASET" \
  --checkpoint              checkpoints/hybrid_lnn_best.pt \
  --xtb-cache-dir           cache/micropattern_xtb \
  --train-ratio             0.70 \
  --val-ratio               0.15 \
  --epochs                  100 \
  --learning-rate           5e-5 \
  --weight-decay            1e-4 \
  --batch-size              16 \
  --seed                    42 \
  --site-labeled-only \
  --compute-xtb-if-missing \
  --output-dir              checkpoints/micropattern_xtb \
  --artifact-dir            artifacts/micropattern_xtb
log "STEP 3 done."

# ── Step 4: extract base predictions ────────────────────────────────────────
log "STEP 4/5: extract base predictions (~45 min)"
"$PYTHON" scripts/extract_base_predictions.py \
  --dataset     "$DATASET" \
  --output      cache/meta_learner/base_predictions.pt \
  --model-names hybrid_lnn hybrid_full_xtb micropattern_xtb \
  --seed        42
log "STEP 4 done."

# ── Step 5: multi-head meta learner ─────────────────────────────────────────
log "STEP 5/5: meta learner (~1 hr)"
"$PYTHON" scripts/train_multihead_meta_learner.py \
  --predictions   cache/meta_learner/base_predictions.pt \
  --dataset       "$DATASET" \
  --train-ratio   0.70 \
  --val-ratio     0.20 \
  --epochs        100 \
  --patience      100 \
  --learning-rate 1e-3 \
  --weight-decay  1e-4 \
  --hidden-dim    32 \
  --mirank-weight 1.0 \
  --bce-weight    0.3 \
  --seed          42 \
  --output-dir    checkpoints/meta_learner_multihead \
  --artifact-dir  artifacts/meta_learner_multihead \
  --cache-dir     cache/meta_learner_multihead
log "ALL DONE."
