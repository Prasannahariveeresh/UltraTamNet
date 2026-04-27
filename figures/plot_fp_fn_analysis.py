"""
Fig. 10 — Per-class False Positives and False Negatives for UltraTamNet on uTHCD.

Loads a trained model and the uTHCD test set, runs inference, builds a confusion
matrix, and plots the FP/FN bar chart per class.

Requires:
    A trained UltraTamNet .keras model (from experiments/train_uthcd_benchmark.py)
    The uTHCD dataset directory

Usage:
    python figures/plot_fp_fn_analysis.py \
        --model_path outputs/table3/UltraTamNet.keras \
        --ds_path /path/to/uTHCD
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.preprocess_uthcd import load_uthcd


def compute_fp_fn(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    return fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to trained UltraTamNet .keras model")
    parser.add_argument("--ds_path", required=True,
                        help="Path to uTHCD dataset directory")
    parser.add_argument("--top_n", type=int, default=30,
                        help="Show only the top-N classes by total FP+FN (default 30)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found: {args.model_path}\n"
            "Run:  python experiments/train_uthcd_benchmark.py --ds_path <uTHCD_path>"
        )

    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)

    print(f"Loading uTHCD dataset from {args.ds_path} ...")
    _, x_test, _, _, y_test_oh, _, class_names = load_uthcd(args.ds_path)

    y_true = np.argmax(y_test_oh, axis=1)
    print(f"Running inference on {len(x_test)} test samples ...")
    y_pred = np.argmax(model.predict(x_test, batch_size=64, verbose=1), axis=1)

    n_classes = len(class_names)
    fp, fn = compute_fp_fn(y_true, y_pred, n_classes)

    total = fp + fn
    if args.top_n and args.top_n < n_classes:
        top_idx = np.argsort(total)[::-1][: args.top_n]
        top_idx = np.sort(top_idx)
    else:
        top_idx = np.arange(n_classes)

    x_labels = [class_names[i] for i in top_idx]
    fp_vals  = fp[top_idx]
    fn_vals  = fn[top_idx]

    x = np.arange(len(top_idx))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(12, len(top_idx) * 0.5), 5))
    ax.bar(x - width / 2, fp_vals, width, label="False Positives", color="tab:orange", alpha=0.85)
    ax.bar(x + width / 2, fn_vals, width, label="False Negatives", color="tab:purple", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(f"Fig. 10 — Per-class False Positives and False Negatives\n"
                 f"(UltraTamNet on uTHCD, top {len(top_idx)} classes by FP+FN)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/fig10_fp_fn_per_class.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
