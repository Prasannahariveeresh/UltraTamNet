#!/usr/bin/env bash
# Runs all pending training jobs sequentially with timestamped logging.
# Table 3 → Table 6 → Ablation

set -uo pipefail

REPO="/home/pras/Desktop/Tamil/UltraTamNet-repo"
UTHCD="/home/pras/Desktop/Tamil/tamil-handwritten-character-recognition"
CUSTOM_RAW="/home/pras/Desktop/Tamil/CUSTOM/OP/new"
CUSTOM_AUG="/home/pras/Desktop/Tamil/CUSTOM/OP/augmented"
LOG="$REPO/outputs/pipeline_pending.log"

cd "$REPO"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ── Table 3 ─────────────────────────────────────────────────────────────────
log "===== TABLE 3: Clear old results and retrain all models ====="
rm -f outputs/table3/table3_results.csv

python3 experiments/train_uthcd_benchmark.py \
    --ds_path "$UTHCD" \
    --epochs  35 \
    --output_dir outputs/table3 \
    2>&1 | tee -a "$LOG"

log "===== TABLE 3 COMPLETE ====="

# ── Table 6 ─────────────────────────────────────────────────────────────────
log "===== TABLE 6: Rebuild augmented datasets and retrain all model×aug combos ====="
rm -f outputs/table6/table6_results.csv

# Rebuild augmented directories (clean rebuild for correctness)
for mult in 4 6 8 10; do
    log "  Building augmented_x${mult} ..."
    rm -rf "${CUSTOM_AUG}_x${mult}"
    python3 augmentation/augment_custom_dataset.py \
        --input_dir  "$CUSTOM_RAW" \
        --output_dir "${CUSTOM_AUG}_x${mult}" \
        --multiplier "$mult" \
        2>&1 | tee -a "$LOG"
    log "  Done augmented_x${mult}"
done

python3 experiments/train_augmentation_study.py \
    --raw_dir  "$CUSTOM_RAW" \
    --aug_dir  "$CUSTOM_AUG" \
    --skip_augmentation \
    --epochs   20 \
    --output_dir outputs/table6 \
    2>&1 | tee -a "$LOG"

log "===== TABLE 6 COMPLETE ====="

# ── Ablation ────────────────────────────────────────────────────────────────
log "===== ABLATION: 5 variants × 3 seeds ====="

python3 experiments/ablation_study.py \
    --ds_path    "$UTHCD" \
    --num_runs   3 \
    --epochs     35 \
    --output_dir outputs/ablation \
    2>&1 | tee -a "$LOG"

log "===== ABLATION COMPLETE ====="
log "===== ALL PENDING TRAINING DONE ====="
