"""
UltraTamNet Ablation Study — systematically evaluates which architectural
components contribute to performance on the uTHCD dataset.

Five variants are tested over multiple random seeds to report mean ± std:

    1. plain CNN baseline
    2. adds residual connections
    3. adds separable convolutions
    4. both, shallow
    5. full UltraTamNet

Reported metrics: Accuracy (%), F1-Score, Precision, Recall, Params (M), Inference (ms/img)

Dataset:    uTHCD (Dataset 1), 156 classes
Epochs:     35 per run
Num runs:   5 (different random seeds per run for statistical reliability)

Usage:
  python experiments/ablation_study.py \
      --ds_path tamil-handwritten-character-recognition
"""

import os
import sys
import time
import random
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.preprocess_uthcd import load_uthcd
from models.ultratamnet import build_ultratamnet, build_ultratamnet_variant

ABLATION_CONFIGS = {
    "A1": {"residual": False, "separable": False, "depth": 2},
    "A2": {"residual": True,  "separable": False, "depth": 2},
    "A3": {"residual": False, "separable": True,  "depth": 2},
    "A4": {"residual": True,  "separable": True,  "depth": 2},
    "A5": {"residual": True,  "separable": True,  "depth": 4},
}


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


def train_variant(cfg, x_train, y_train, x_test, y_test, x_valid, y_valid,
                  input_shape, num_classes, epochs, seed):
    set_seed(seed)

    # A5 (residual=True, separable=True, depth=4) is the full UltraTamNet — build it exactly
    if cfg.get("residual") and cfg.get("separable") and cfg.get("depth") == 4:
        model = build_ultratamnet(input_shape, num_classes)
    else:
        model = build_ultratamnet_variant(input_shape, num_classes, **cfg)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1,
        height_shift_range=0.1, zoom_range=0.1,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=0,
    )
    model.fit(
        datagen.flow(x_train, y_train),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[early_stop],
        verbose=0,
    )

    start = time.time()
    _ = model.predict(x_valid, verbose=0)
    inf_ms = (time.time() - start) / len(x_valid) * 1000

    y_probs = model.predict(x_valid, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)
    y_true  = np.argmax(y_valid, axis=1)

    acc       = np.mean(y_pred == y_true) * 100
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)

    return acc, f1, precision, recall, inf_ms, model.count_params()


def main():
    parser = argparse.ArgumentParser(description="UltraTamNet ablation study")
    parser.add_argument("--ds_path",    required=True, help="Path to uTHCD dataset root")
    parser.add_argument("--num_runs",   type=int, default=5,  help="Random seeds per variant")
    parser.add_argument("--epochs",     type=int, default=35)
    parser.add_argument("--output_dir", default="outputs/ablation")
    parser.add_argument("--variant",    default="all", help="Run only this variant (e.g. A5) or 'all'")
    args = parser.parse_args()

    print("Loading uTHCD ...")
    x_train, x_test, x_valid, y_train, y_test, y_valid, _ = load_uthcd(args.ds_path)
    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[-1]

    print(f"Train={x_train.shape}  Test={x_test.shape}  Valid={x_valid.shape}")

    # Load existing results to support skip/append
    csv_path = os.path.join(args.output_dir, "ablation_results.csv")
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        done_variants = set(existing_df["Variant"].tolist())
    else:
        existing_df = pd.DataFrame()
        done_variants = set()

    rows = []
    configs_to_run = {k: v for k, v in ABLATION_CONFIGS.items()
                      if (args.variant == "all" or k == args.variant) and k not in done_variants}

    for key, cfg in configs_to_run.items():
        print(f"\n=== Variant {key}  {cfg} ===")
        acc_list, f1_list, prec_list, rec_list, inf_list = [], [], [], [], []

        for run in range(args.num_runs):
            seed = 42 + run
            print(f"  Run {run+1}/{args.num_runs}  seed={seed}")
            acc, f1, prec, rec, inf_ms, params = train_variant(
                cfg, x_train, y_train, x_test, y_test, x_valid, y_valid,
                input_shape, num_classes, args.epochs, seed,
            )
            acc_list.append(acc)
            f1_list.append(f1)
            prec_list.append(prec)
            rec_list.append(rec)
            inf_list.append(inf_ms)

        row = {
            "Variant":          key,
            "Residuals":        "✓" if cfg["residual"]  else "✗",
            "Separable Convs":  "✓" if cfg["separable"] else "✗",
            "Feature Depth":    cfg["depth"],
            "Acc (%)":          f"{np.mean(acc_list):.2f} ± {np.std(acc_list):.2f}",
            "F1-Score":         f"{np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}",
            "Precision":        f"{np.mean(prec_list):.3f} ± {np.std(prec_list):.3f}",
            "Recall":           f"{np.mean(rec_list):.3f} ± {np.std(rec_list):.3f}",
            "Params (M)":       round(params / 1e6, 2),
            "Inference (ms/img)": round(np.mean(inf_list), 2),
        }
        rows.append(row)
        print(f"  → Acc: {row['Acc (%)']}  F1: {row['F1-Score']}")

    os.makedirs(args.output_dir, exist_ok=True)
    new_df = pd.DataFrame(rows)
    df = pd.concat([existing_df, new_df], ignore_index=True)
    df.to_csv(csv_path, index=False)

    print("\n--- Ablation Study Results ---")
    print(df.to_string(index=False))
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
