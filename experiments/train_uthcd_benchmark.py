"""
Reproduce Table 3 — Model comparison on the uTHCD dataset (156 classes).

Trains each model listed below on the uTHCD dataset and reports:
  Test Accuracy (%), Test Loss, F1-Score, Params (M), FLOPs (M)

Models:
  ResNet50, DenseNet169, DenseNet121, EfficientNetB0, EfficientNetB5,
  LeNet-5, MobileNetV2, MobileNetV3Small, MobileNetV3Large,
  NASNetMobile, Xception, UltraTamNet (ours)

Dataset:    uTHCD (Dataset 1)
Input:      64×64 grayscale (single channel)
Epochs:     35 (EarlyStopping with patience=5)
Optimizer:  Adam
Loss:       Categorical cross-entropy

Training setup (matches paper / benchmark notebooks exactly):
  Baseline models: ImageDataGenerator augmentation (rotation±10, shift±10%, zoom±10%)
                   + EarlyStopping only (no ReduceLROnPlateau)
  UltraTamNet:     No augmentation, EarlyStopping + ReduceLROnPlateau

Usage:
  # Train all models sequentially
  python experiments/train_uthcd_benchmark.py --ds_path tamil-handwritten-character-recognition

  # Train a single model
  python experiments/train_uthcd_benchmark.py --ds_path tamil-handwritten-character-recognition \
      --model UltraTamNet
"""

import os
import sys
import json
import argparse

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from data.preprocess_uthcd import load_uthcd
from models.ultratamnet import build_ultratamnet
from models.baselines import build_model, _REGISTRY
from utils.evaluate import get_flops, compute_metrics, plot_training_curves

ALL_MODELS = list(_REGISTRY.keys()) + ["UltraTamNet"]

# Augmentation matching benchmark notebooks exactly
_DATAGEN = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)


def get_baseline_callbacks():
    """Baselines: EarlyStopping only (no ReduceLROnPlateau) — matches notebooks."""
    return [
        EarlyStopping(
            monitor="val_accuracy", patience=5, verbose=1,
            mode="max", restore_best_weights=True,
        ),
    ]


def get_ultratamnet_callbacks():
    """UltraTamNet: EarlyStopping + ReduceLROnPlateau — matches UltraTamNet notebook."""
    return [
        EarlyStopping(
            monitor="val_accuracy", patience=5, verbose=1,
            mode="max", restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy", patience=3, factor=0.5,
            min_lr=1e-8, verbose=1,
        ),
    ]


def train_one(model_name, x_train, x_test, x_valid, y_train, y_test, y_valid,
              epochs=35, output_dir="outputs/table3"):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")

    num_classes = y_train.shape[-1]

    if model_name == "UltraTamNet":
        model = build_ultratamnet(input_shape=(64, 64, 1), num_classes=num_classes)
    else:
        model = build_model(model_name, num_classes=num_classes)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )

    flops  = get_flops(model)
    params = model.count_params()

    if model_name == "UltraTamNet":
        # UltraTamNet trains without augmentation (matches its benchmark notebook)
        hist = model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=get_ultratamnet_callbacks(),
            verbose=1,
        )
    else:
        # All baselines train with augmentation (matches their benchmark notebooks)
        hist = model.fit(
            _DATAGEN.flow(x_train, y_train),
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=get_baseline_callbacks(),
            verbose=1,
        )

    os.makedirs(output_dir, exist_ok=True)
    plot_training_curves(hist, save_path=os.path.join(output_dir, f"{model_name}_curves.png"))

    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)

    metrics = compute_metrics(model, x_valid, y_valid)

    model.save(os.path.join(output_dir, f"{model_name}.keras"))

    return {
        "Model":       model_name,
        "Test Acc (%)": round(metrics["accuracy"], 2),
        "Test Loss":    round(metrics["loss"], 4),
        "F1-Score":     round(metrics["f1"], 3),
        "Params (M)":   round(params / 1e6, 2),
        "FLOPs (M)":    round(flops / 1e6, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 3")
    parser.add_argument(
        "--ds_path", required=True,
        help="Path to uTHCD root (contains train/ and test/ folders)",
    )
    parser.add_argument(
        "--model", default="all",
        help=f"Model name to train, or 'all'. Choices: {ALL_MODELS}",
    )
    parser.add_argument("--epochs",     type=int, default=35)
    parser.add_argument("--output_dir", default="outputs/table3")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip models that already have a row in the existing CSV")
    args = parser.parse_args()

    csv_path = os.path.join(args.output_dir, "table3_results.csv")

    existing_df = pd.DataFrame()
    if args.skip_existing and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        already_done = set(existing_df["Model"].tolist())
        print(f"Skipping already-trained models: {already_done}")
    else:
        already_done = set()

    print("Loading uTHCD dataset ...")
    x_train, x_test, x_valid, y_train, y_test, y_valid, class_names = load_uthcd(args.ds_path)
    print(f"  Train={x_train.shape}  Test={x_test.shape}  Valid={x_valid.shape}")

    models_to_run = ALL_MODELS if args.model == "all" else [args.model]
    models_to_run = [m for m in models_to_run if m not in already_done]

    os.makedirs(args.output_dir, exist_ok=True)
    accumulated_df = existing_df.copy()

    for name in models_to_run:
        row = train_one(
            name, x_train, x_test, x_valid, y_train, y_test, y_valid,
            epochs=args.epochs, output_dir=args.output_dir,
        )
        print(row)
        new_row_df = pd.DataFrame([row])
        accumulated_df = pd.concat([accumulated_df, new_row_df], ignore_index=True)
        accumulated_df.to_csv(csv_path, index=False)
        print(f"[Checkpoint] Saved {len(accumulated_df)} model(s) to {csv_path}")

    df = accumulated_df

    print("\n--- Table 3 Results ---")
    print(df.to_string(index=False))
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
