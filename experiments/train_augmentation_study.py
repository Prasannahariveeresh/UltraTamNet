"""
Reproduce Table 6 — Augmentation study on the custom Tamil vowel dataset (12 classes).

For each model × augmentation multiple (×4, ×6, ×8, ×10), this script:
  1. Runs the offline augmentation pipeline to build the dataset at that multiple
  2. Trains the model
  3. Reports Train Acc, Test Acc, Test Loss, Precision, Recall, F1-Score

Models evaluated:
  ResNet50, MobileNetV2, DenseNet121, DenseNet169, UltraTamNet (ours)

Also trains UltraTamNet *without* augmentation as the baseline row.

Dataset:    Custom Tamil Vowels (Dataset 2) — 12 classes
Input:      64×64 grayscale
Epochs:     20 (EarlyStopping with patience=5)
Optimizer:  Adam
Loss:       Categorical cross-entropy

Usage:
  # Full table (all models × all multipliers) — takes a while
  python experiments/train_augmentation_study.py \
      --raw_dir  CUSTOM/OP/new \
      --aug_dir  CUSTOM/OP/augmented

  # Single model × single multiplier
  python experiments/train_augmentation_study.py \
      --raw_dir  CUSTOM/OP/new \
      --aug_dir  CUSTOM/OP/augmented \
      --model UltraTamNet --multiplier 10

  # Skip augmentation (load pre-built dataset directly)
  python experiments/train_augmentation_study.py \
      --aug_dir CUSTOM/OP/augmented \
      --skip_augmentation --model UltraTamNet --multiplier 10
"""

import os
import sys
import json
import argparse
import shutil

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from augmentation.augment_custom_dataset import process_images
from data.preprocess_custom import load_custom_dataset
from models.ultratamnet import build_ultratamnet
from models.baselines import build_model
from utils.evaluate import compute_metrics, plot_training_curves

TABLE6_MODELS      = ["ResNet50", "MobileNetV2", "DenseNet121", "UltraTamNet"]
TABLE6_MULTIPLIERS = [4, 6, 8, 10]
NUM_CLASSES        = 12

# MobileNetV2 needs pretrained ImageNet weights + 3-channel RGB input to converge
# on the small custom dataset (as used in the original notebooks).
def _build_mobilenetv2_imagenet(num_classes=12, freeze_base=False):
    from tensorflow.keras.applications import MobileNetV2 as MNV2
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    base = MNV2(include_top=False, weights="imagenet", input_shape=(64, 64, 3))
    base.trainable = not freeze_base
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base.input, outputs=outputs)


def get_callbacks():
    return [
        EarlyStopping(
            monitor="val_accuracy", patience=5, verbose=1,
            mode="max", restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1,
        ),
    ]


def train_one(model_name, x_train, x_test, y_train, y_test, epochs=20, output_dir="outputs/table6"):
    if model_name == "UltraTamNet":
        model = build_ultratamnet(input_shape=(64, 64, 1), num_classes=NUM_CLASSES)
    elif model_name == "MobileNetV2":
        # Convert grayscale (N,64,64,1) → RGB (N,64,64,3) for ImageNet-pretrained model
        x_train = np.repeat(x_train, 3, axis=-1)
        x_test  = np.repeat(x_test,  3, axis=-1)
        model = _build_mobilenetv2_imagenet(num_classes=NUM_CLASSES, freeze_base=False)
    else:
        model = build_model(model_name, num_classes=NUM_CLASSES)

    from tensorflow.keras.optimizers import Adam
    if model_name == "UltraTamNet":
        optimizer = Adam(learning_rate=0.0005)
    else:
        optimizer = "adam"
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    hist = model.fit(
        datagen.flow(x_train, y_train),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=get_callbacks(),
        verbose=1,
    )

    train_acc = max(hist.history["accuracy"]) * 100
    metrics   = compute_metrics(model, x_test, y_test)

    os.makedirs(output_dir, exist_ok=True)
    plot_training_curves(
        hist,
        save_path=os.path.join(output_dir, f"{model_name}_curves.png"),
    )

    history_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)

    model.save(os.path.join(output_dir, f"{model_name}.keras"))

    return train_acc, metrics


def run_no_augmentation(raw_dir, output_dir, epochs):
    """Train UltraTamNet without augmentation (first row of Table 6 for UltraTamNet)."""
    print("\n--- UltraTamNet (no augmentation) ---")
    x_train, x_test, y_train, y_test = load_custom_dataset(raw_dir, add_noise=False)
    train_acc, metrics = train_one("UltraTamNet", x_train, x_test, y_train, y_test,
                                   epochs=epochs, output_dir=output_dir)
    return {
        "Model":          "UltraTamNet",
        "Augmentation":   "None",
        "Train Acc (%)":  round(train_acc, 2),
        "Test Acc (%)":   round(metrics["accuracy"], 2),
        "Test Loss":      round(metrics["loss"], 4),
        "Precision (%)":  round(metrics["precision"], 2),
        "Recall (%)":     round(metrics["recall"], 2),
        "F1-Score (%)":   round(metrics["f1"] * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 6")
    parser.add_argument("--raw_dir",  default="CUSTOM/OP/new",
                        help="Raw (unaug.) custom dataset directory")
    parser.add_argument("--aug_dir",  default="CUSTOM/OP/augmented",
                        help="Output dir for augmented dataset (reused across multipliers)")
    parser.add_argument("--model",   default="all",
                        help=f"Model name or 'all'. Choices: {TABLE6_MODELS}")
    parser.add_argument("--multiplier", type=int, default=0,
                        help="Augmentation multiplier (4/6/8/10), or 0 for all")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--output_dir", default="outputs/table6")
    parser.add_argument("--skip_augmentation", action="store_true",
                        help="Skip offline augmentation (use existing aug_dir)")
    args = parser.parse_args()

    models_to_run = TABLE6_MODELS if args.model == "all" else [args.model]
    multipliers   = TABLE6_MULTIPLIERS if args.multiplier == 0 else [args.multiplier]

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "table6_results.csv")

    # Load existing results so we can skip already-done rows
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        done_keys = set(zip(existing_df["Model"], existing_df["Augmentation"]))
    else:
        existing_df = pd.DataFrame()
        done_keys = set()

    def _checkpoint(row):
        nonlocal existing_df
        new_df = pd.DataFrame([row])
        existing_df = pd.concat([existing_df, new_df], ignore_index=True)
        existing_df.to_csv(csv_path, index=False)
        print(f"[Checkpoint] {len(existing_df)} row(s) saved to {csv_path}")

    # No-augmentation baseline for UltraTamNet
    if "UltraTamNet" in models_to_run and args.multiplier == 0:
        if ("UltraTamNet", "None") not in done_keys:
            row = run_no_augmentation(args.raw_dir, args.output_dir, args.epochs)
            print(row)
            _checkpoint(row)
        else:
            print("Skipping UltraTamNet (no aug) — already in CSV")

    for mult in multipliers:
        aug_dir_mult = f"{args.aug_dir}_x{mult}"

        if not args.skip_augmentation:
            print(f"\n--- Building augmented dataset ×{mult} → {aug_dir_mult} ---")
            if os.path.exists(aug_dir_mult):
                shutil.rmtree(aug_dir_mult)
            process_images(args.raw_dir, aug_dir_mult, multiplier=mult)
        else:
            if not os.path.exists(aug_dir_mult):
                raise FileNotFoundError(
                    f"Augmented dir not found: {aug_dir_mult}\n"
                    f"Run without --skip_augmentation first."
                )

        print(f"\nLoading dataset ×{mult} from {aug_dir_mult} ...")
        x_train, x_test, y_train, y_test = load_custom_dataset(aug_dir_mult, add_noise=False)
        print(f"  Train={x_train.shape}  Test={x_test.shape}")

        for model_name in models_to_run:
            aug_label = f"×{mult}"
            if (model_name, aug_label) in done_keys:
                print(f"Skipping {model_name} ×{mult} — already in CSV")
                continue

            print(f"\n{'='*60}")
            print(f"  {model_name}  ×{mult}")
            print(f"{'='*60}")

            train_acc, metrics = train_one(
                model_name, x_train, x_test, y_train, y_test,
                epochs=args.epochs,
                output_dir=os.path.join(args.output_dir, f"x{mult}"),
            )

            row = {
                "Model":          model_name,
                "Augmentation":   aug_label,
                "Train Acc (%)":  round(train_acc, 2),
                "Test Acc (%)":   round(metrics["accuracy"], 2),
                "Test Loss":      round(metrics["loss"], 4),
                "Precision (%)":  round(metrics["precision"], 2),
                "Recall (%)":     round(metrics["recall"], 2),
                "F1-Score (%)":   round(metrics["f1"] * 100, 2),
            }
            print(row)
            _checkpoint(row)

    print("\n--- Table 6 Results ---")
    print(existing_df.to_string(index=False))
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
