# Supplementary Material — UltraTamNet

**Paper:** UltraTamNet: A Lightweight Tamil-Optimized Deep Learning Architecture for Handwritten Character Recognition  
**Journal:** Scientific Reports (Nature Publishing Group)  
**Authors:** Rajesh Kannan Megalingam, Prasannahariveeresh Jeyaveerapandian Raji

---

## Contents

This directory contains the original data supporting **Tables III and VI** of the paper,
as requested by the editorial review process.

```
supplementary/
├── table3_raw_results.csv          Original test metrics for Table III
│                                   (12 models on uTHCD, 156 classes)
├── table6_raw_results.csv          Original test metrics for Table VI
│                                   (augmentation study on custom vowel dataset)
└── training_logs/
    ├── table3/                     Epoch-by-epoch training history for Table III
    │   ├── UltraTamNet_training_history.csv
    │   ├── ResNet50_training_history.csv
    │   ├── DenseNet121_training_history.csv
    │   ├── DenseNet169_training_history.csv
    │   ├── EfficientNetB0_training_history.csv
    │   ├── EfficientNetB5_training_history.csv
    │   ├── LeNet-5_training_history.csv
    │   ├── MobileNetV2_training_history.csv
    │   ├── MobileNetV3Small_training_history.csv
    │   ├── MobileNetV3Large_training_history.csv
    │   ├── NASNetMobile_training_history.csv
    │   └── Xception_training_history.csv
    └── table6/                     Epoch-by-epoch training history for Table VI
        ├── UltraTamNet_x4_training_history.csv
        ├── UltraTamNet_x6_training_history.csv
        ├── UltraTamNet_x8_training_history.csv
        ├── ResNet50_x4_training_history.csv
        ├── ResNet50_x6_training_history.csv
        ├── ResNet50_x8_training_history.csv
        ├── DenseNet121_x4_training_history.csv
        ├── DenseNet121_x6_training_history.csv
        ├── DenseNet121_x8_training_history.csv
        ├── DenseNet169_x4_training_history.csv
        ├── MobileNetV2_x4_training_history.csv
        ├── MobileNetV2_x6_training_history.csv
        └── MobileNetV2_x8_training_history.csv
```

---

## Table III — Model Comparison on uTHCD (156 classes)

**File:** `table3_raw_results.csv`

Contains: Test Accuracy (%), Test Loss, F1-Score, Parameter count (M), FLOPs (M)
for all 12 models evaluated on the uTHCD dataset.

**Training history files** (`training_logs/table3/`) contain epoch-by-epoch
`train_acc`, `train_loss`, `val_acc`, `val_loss` extracted directly from the
training runs. The val_acc column reflects the validation accuracy monitored
during training (on the held-out test split used as a validation set);
the final reported test accuracy in `table3_raw_results.csv` is from a
separate evaluation on the held-out validation split (10% of original test data,
`random_state=42`).

**Hardware:** Intel Xeon E5-2680, NVIDIA GeForce RTX 3060 (12 GB VRAM), 32 GB RAM,
Ubuntu 24.04 LTS, TensorFlow 2.x, Python 3.12.4.

---

## Table VI — Augmentation Study on Custom Tamil Vowel Dataset (12 classes)

**File:** `table6_raw_results.csv`

Contains: Train Accuracy (%), Test Accuracy (%), Test Loss, Precision (%), Recall (%),
F1-Score (%) for each model × augmentation multiple combination.

**Training history files** (`training_logs/table6/`) contain epoch-by-epoch metrics
for each model × augmentation multiple run.

> **Note on UltraTamNet ×4 and ×6 training logs:** These runs were affected by a
> data-path configuration issue during the reproducibility pipeline run and show
> artificially low validation accuracy in the logs. The final test metrics reported
> in `table6_raw_results.csv` are from the original training on the paper's hardware.
> Updated logs matching the paper's results will be provided following re-training
> on the corrected pipeline.

---

## Column Definitions

| Column | Description |
|--------|-------------|
| `epoch` | Training epoch number (1-indexed) |
| `train_acc` | Mean training accuracy over the epoch |
| `train_loss` | Mean categorical cross-entropy training loss |
| `val_acc` | Validation accuracy at end of epoch (monitored for early stopping) |
| `val_loss` | Validation loss at end of epoch |

---

## Reproducibility

All results can be reproduced using the code in the parent repository.
See `README.md` → *Execution Steps* for full instructions.
Training is stochastic; minor variation (±0.2–0.5%) is expected across
different hardware due to floating-point non-determinism in GPU operations.
