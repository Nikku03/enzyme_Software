#!/usr/bin/env bash
# train_all.sh — full training pipeline, sequential, overnight
# Run from project root:
#   bash train_all.sh 2>&1 | tee logs/train_all_$(date +%Y%m%d).log
set -euo pipefail

# ── Python: auto-detect or override with PYTHON env var ──────────────────────
# Local Mac (bondbreak env):  PYTHON=/Users/deepika/anaconda3/envs/bondbreak/bin/python bash train_all.sh
# Google Colab / Linux:       bash train_all.sh   (uses system python)
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$(pwd)/src"
export KMP_DUPLICATE_LIB_OK=TRUE             # prevents OpenMP conflict on some systems
# export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0  # Mac MPS only — harmless no-op elsewhere

# ── Config ───────────────────────────────────────────────────────────────────
DATASET="data/prepared_training/main5_all_models_conservative.json"
EPOCHS=100
SEED=42
BATCH=16  # 16 — runs full 100 epochs, ~2x faster than batch 8

# ── Dirs ─────────────────────────────────────────────────────────────────────
mkdir -p logs checkpoints \
         checkpoints/hybrid_full_xtb \
         checkpoints/micropattern_xtb \
         checkpoints/meta_learner_multihead \
         checkpoints/cahml \
         artifacts/hybrid_full_xtb \
         artifacts/micropattern_xtb \
         artifacts/meta_learner_multihead \
         artifacts/cahml \
         cache/meta_learner \
         cache/meta_learner_multihead \
         cache/cahml \
         cache/manual_engine_full

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Python:  $PYTHON"
log "Dataset: $DATASET"
log "Epochs:  $EPOCHS  |  Batch: $BATCH  |  Seed: $SEED"
log "RAM:     8.6 GB total (MPS shares with CPU)"

# ── Step 1: hybrid_lnn — resumes from latest checkpoint if it exists ────────
log "=========================================="
log "STEP 1/5: hybrid_lnn"
log "=========================================="
"$PYTHON" scripts/train_hybrid_lnn.py \
  --dataset                 "$DATASET" \
  --train-ratio             0.70 \
  --val-ratio               0.20 \
  --epochs                  $EPOCHS \
  --early-stopping-patience $EPOCHS \
  --learning-rate           2e-4 \
  --weight-decay            1e-4 \
  --batch-size              $BATCH \
  --seed                    $SEED \
  --output-dir              checkpoints \
  --manual-feature-cache-dir cache/manual_engine_full \
  --auto-resume-latest
log "STEP 1 done."

# ── Step 2: hybrid_full_xtb (fine-tune from step 1) ─────────────────────────
log "=========================================="
log "STEP 2/5: hybrid_full_xtb"
log "=========================================="
"$PYTHON" scripts/train_hybrid_full_xtb.py \
  --dataset                 "$DATASET" \
  --checkpoint              checkpoints/hybrid_lnn_best.pt \
  --xtb-cache-dir           cache/full_xtb \
  --train-ratio             0.70 \
  --val-ratio               0.15 \
  --epochs                  $EPOCHS \
  --early-stopping-patience $EPOCHS \
  --learning-rate           5e-5 \
  --weight-decay            1e-4 \
  --batch-size              $BATCH \
  --seed                    $SEED \
  --output-dir              checkpoints/hybrid_full_xtb \
  --artifact-dir            artifacts/hybrid_full_xtb \
  --compute-xtb-if-missing
log "STEP 2 done."

# ── Step 3: micropattern_xtb (fine-tune from step 1) ────────────────────────
log "=========================================="
log "STEP 3/5: micropattern_xtb"
log "=========================================="
"$PYTHON" scripts/train_micropattern_xtb.py \
  --dataset                 "$DATASET" \
  --checkpoint              checkpoints/hybrid_lnn_best.pt \
  --xtb-cache-dir           cache/micropattern_xtb \
  --train-ratio             0.70 \
  --val-ratio               0.15 \
  --epochs                  $EPOCHS \
  --learning-rate           5e-5 \
  --weight-decay            1e-4 \
  --batch-size              $BATCH \
  --seed                    $SEED \
  --site-labeled-only \
  --compute-xtb-if-missing \
  --output-dir              checkpoints/micropattern_xtb \
  --artifact-dir            artifacts/micropattern_xtb
log "STEP 3 done."

# ── Step 4: extract stacked predictions ─────────────────────────────────────
log "=========================================="
log "STEP 4/7: extract base predictions"
log "=========================================="
"$PYTHON" scripts/extract_base_predictions.py \
  --dataset     "$DATASET" \
  --output      cache/meta_learner/base_predictions.pt \
  --model-names hybrid_lnn hybrid_full_xtb micropattern_xtb \
  --seed        $SEED
log "STEP 4 done."

# ── Step 5: CAHML OOF — train CAHML 5-fold, produce leak-free predictions ────
log "=========================================="
log "STEP 5/7: CAHML OOF predictions (5-fold, leak-free)"
log "=========================================="
"$PYTHON" scripts/generate_oof_cahml_predictions.py \
  --predictions  cache/meta_learner/base_predictions.pt \
  --dataset      "$DATASET" \
  --output       cache/meta_learner/base_predictions_cahml_oof.pt \
  --n-folds      5 \
  --epochs       $EPOCHS \
  --patience     15 \
  --learning-rate 1e-3 \
  --weight-decay  1e-4 \
  --hidden-dim   64 \
  --mirank-weight 1.0 \
  --bce-weight   0.3 \
  --listmle-weight 0.5 \
  --focal-weight  0.2 \
  --seed         $SEED \
  --work-dir     cache/cahml_oof_folds
log "STEP 5 done."

# ── Step 6: multi-head meta learner (trained on 4-expert OOF predictions) ────
log "=========================================="
log "STEP 6/7: multi-head meta learner (4 experts)"
log "=========================================="
"$PYTHON" scripts/train_multihead_meta_learner.py \
  --predictions  cache/meta_learner/base_predictions_cahml_oof.pt \
  --dataset      "$DATASET" \
  --train-ratio  0.70 \
  --val-ratio    0.20 \
  --epochs       $EPOCHS \
  --patience     $EPOCHS \
  --learning-rate 1e-3 \
  --weight-decay  1e-4 \
  --hidden-dim   64 \
  --mirank-weight 1.0 \
  --bce-weight   0.3 \
  --seed         $SEED \
  --output-dir   checkpoints/meta_learner_multihead \
  --artifact-dir artifacts/meta_learner_multihead \
  --cache-dir    cache/meta_learner_multihead
log "STEP 6 done."

# ── Step 7: CAHML final — trained on full data (for standalone inference) ────
log "=========================================="
log "STEP 7/7: CAHML final (full-data, for inference)"
log "=========================================="
"$PYTHON" scripts/train_cahml.py \
  --predictions   cache/meta_learner/base_predictions.pt \
  --dataset       "$DATASET" \
  --train-ratio   0.70 \
  --val-ratio     0.15 \
  --epochs        $EPOCHS \
  --patience      $EPOCHS \
  --learning-rate 1e-3 \
  --weight-decay  1e-4 \
  --hidden-dim    64 \
  --mirank-weight 1.0 \
  --bce-weight    0.3 \
  --listmle-weight 0.5 \
  --focal-weight  0.2 \
  --seed          $SEED \
  --output-dir    checkpoints/cahml \
  --artifact-dir  artifacts/cahml \
  --cache-dir     cache/cahml
log "STEP 7 done."

log "=========================================="
log "ALL DONE. Check checkpoints/ for saved models."
log "=========================================="
