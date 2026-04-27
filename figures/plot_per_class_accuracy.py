"""
Fig. 11 — Per-class Accuracy for UltraTamNet on uTHCD.

Loads a trained model and the uTHCD test set, runs inference, and plots
per-class accuracy as a bar chart sorted by class index.

Requires:
    A trained UltraTamNet .keras model (from experiments/train_uthcd_benchmark.py)
    The uTHCD dataset directory

Usage:
    python figures/plot_per_class_accuracy.py \
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


def per_class_accuracy(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.where(cm.sum(axis=1) > 0,
                       np.diag(cm) / cm.sum(axis=1),
                       0.0)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to trained UltraTamNet .keras model")
    parser.add_argument("--ds_path", required=True,
                        help="Path to uTHCD dataset directory")
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
    acc = per_class_accuracy(y_true, y_pred, n_classes)

    overall_acc = np.mean(y_true == y_pred) * 100
    print(f"Overall test accuracy: {overall_acc:.2f}%")
    print(f"Mean per-class accuracy: {acc.mean() * 100:.2f}%")
    print(f"Min per-class accuracy: {acc.min() * 100:.2f}% (class {acc.argmin()}: {class_names[acc.argmin()]})")

    x = np.arange(n_classes)
    colors = ["tab:red" if a < 0.80 else "tab:orange" if a < 0.90 else "tab:blue"
              for a in acc]

    fig, ax = plt.subplots(figsize=(max(14, n_classes * 0.12), 5))
    ax.bar(x, acc * 100, color=colors, alpha=0.85)
    ax.axhline(overall_acc, color="black", linestyle="--", linewidth=1.2,
               label=f"Overall acc: {overall_acc:.2f}%")

    ax.set_xticks(x[::max(1, n_classes // 30)])
    ax.set_xticklabels([class_names[i] for i in x[::max(1, n_classes // 30)]],
                       rotation=90, fontsize=7)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Fig. 11 — Per-class Accuracy (UltraTamNet on uTHCD)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_extras = [
        Patch(color="tab:blue",   label="≥ 90%"),
        Patch(color="tab:orange", label="80–90%"),
        Patch(color="tab:red",    label="< 80%"),
    ]
    ax.legend(handles=[ax.get_legend_handles_labels()[0][0]] + legend_extras,
              labels=[f"Overall acc: {overall_acc:.2f}%", "≥ 90%", "80–90%", "< 80%"])

    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/fig11_per_class_accuracy.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
