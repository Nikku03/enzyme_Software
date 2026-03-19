#!/usr/bin/env bash
# train_phase_1.sh — trains only hybrid_lnn (base model), ~6-8 hours overnight
# Run: bash train_phase_1.sh 2>&1 | tee logs/train_phase1_$(date +%Y%m%d).log
set -euo pipefail

PYTHON="/Users/deepika/anaconda3/envs/bondbreak/bin/python"
export PYTHONPATH="$(pwd)/src"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

DATASET="data/prepared_training/main5_all_models_conservative.json"
mkdir -p logs checkpoints cache/manual_engine_full

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log "Starting hybrid_lnn training — ~6-8 hours"

"$PYTHON" scripts/train_hybrid_lnn.py \
  --dataset                 "$DATASET" \
  --train-ratio             0.70 \
  --val-ratio               0.20 \
  --epochs                  100 \
  --early-stopping-patience 100 \
  --learning-rate           2e-4 \
  --weight-decay            1e-4 \
  --batch-size              16 \
  --seed                    42 \
  --output-dir              checkpoints \
  --manual-feature-cache-dir cache/manual_engine_full

log "Done. Check checkpoints/hybrid_lnn_best.pt"
log "Run train_phase_2.sh next."
