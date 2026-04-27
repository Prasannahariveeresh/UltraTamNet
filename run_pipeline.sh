#!/usr/bin/env bash
# Full pipeline: Table 3 → Table 6 → Ablation → All figures
# Logs to outputs/pipeline.log with timestamps

set -uo pipefail

REPO="/home/pras/Desktop/Tamil/UltraTamNet-repo"
UTHCD="/home/pras/Desktop/Tamil/tamil-handwritten-character-recognition"
CUSTOM_RAW="/home/pras/Desktop/Tamil/CUSTOM/OP/new"
CUSTOM_AUG="/home/pras/Desktop/Tamil/CUSTOM/OP/augmented"
LOG="$REPO/outputs/pipeline.log"

cd "$REPO"
mkdir -p outputs/table3 outputs/table6 outputs/ablation outputs/figures

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------------------
log "======== PIPELINE START ========"
log "Table 3 remaining: LeNet-5 → Xception (easy to hard)"
# ---------------------------------------------------------------------------

for model in "LeNet-5" "MobileNetV3Small" "MobileNetV3Large" "MobileNetV2" "NASNetMobile" "EfficientNetB5" "Xception"; do
    log "--- Starting: $model ---"
    python3 experiments/train_uthcd_benchmark.py \
        --ds_path "$UTHCD" \
        --model "$model" \
        --skip_existing \
        --output_dir outputs/table3 || log "WARNING: $model failed, continuing"
    log "--- Done: $model ---"
done

log "======== TABLE 3 COMPLETE ========"

# ---------------------------------------------------------------------------
log "======== TABLE 6: Augmentation study (custom dataset, 12 classes) ========"
# DenseNet169 excluded — causes OOM on RTX 2050 4 GB in this context
# ---------------------------------------------------------------------------

python3 experiments/train_augmentation_study.py \
    --raw_dir  "$CUSTOM_RAW" \
    --aug_dir  "$CUSTOM_AUG" \
    --skip_augmentation \
    --output_dir outputs/table6

log "======== TABLE 6 COMPLETE ========"

# ---------------------------------------------------------------------------
log "======== ABLATION STUDY (3 seeds × 5 variants, early-stopped) ========"
# ---------------------------------------------------------------------------

python3 experiments/ablation_study.py \
    --ds_path    "$UTHCD" \
    --num_runs   3 \
    --output_dir outputs/ablation

log "======== ABLATION COMPLETE ========"

# ---------------------------------------------------------------------------
log "======== FIGURES ========"
# ---------------------------------------------------------------------------

log "Fig 7"
python3 figures/plot_dataset_samples.py \
    --ds_path "$CUSTOM_RAW" \
    --save_path outputs/figures/fig7_dataset_samples.png || log "WARNING: Fig 7 failed"

log "Fig 8"
python3 figures/plot_preprocessing_steps.py \
    --ds_path "$CUSTOM_RAW" --class_idx 0 \
    --save_path outputs/figures/fig8_preprocessing.png || log "WARNING: Fig 8 failed"

log "Fig 9"
python3 figures/plot_training_curves.py \
    --history_json outputs/table3/UltraTamNet_history.json \
    --title "UltraTamNet on uTHCD" --fig_num 9 || log "WARNING: Fig 9 failed"

log "Fig 10"
python3 figures/plot_fp_fn_analysis.py \
    --model_path outputs/table3/UltraTamNet.keras \
    --ds_path "$UTHCD" || log "WARNING: Fig 10 failed"

log "Fig 11"
python3 figures/plot_per_class_accuracy.py \
    --model_path outputs/table3/UltraTamNet.keras \
    --ds_path "$UTHCD" || log "WARNING: Fig 11 failed"

log "Fig 12"
python3 figures/plot_accuracy_vs_model_size.py \
    --results_csv outputs/table3/table3_results.csv || log "WARNING: Fig 12 failed"

log "Fig 13"
python3 figures/plot_gradcam.py \
    --model_path outputs/table3/UltraTamNet.keras \
    --ds_path "$UTHCD" || log "WARNING: Fig 13 failed"

log "Fig 14"
python3 figures/plot_training_curves.py \
    --history_json outputs/table6/x10/UltraTamNet_history.json \
    --title "UltraTamNet on Custom Tamil Vowel Dataset" --fig_num 14 || log "WARNING: Fig 14 failed"

log "Fig 15"
python3 figures/plot_confusion_matrix.py \
    --model_path outputs/table6/x10/UltraTamNet.keras \
    --ds_path    "${CUSTOM_AUG}_x10" || log "WARNING: Fig 15 failed"

log "======== FIGURES COMPLETE ========"
log "======== PIPELINE DONE — all outputs in $REPO/outputs/ ========"
