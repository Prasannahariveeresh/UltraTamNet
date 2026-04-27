"""
Fig. 15 — Confusion matrix for UltraTamNet on the custom Tamil vowel dataset.

Loads a trained model and the custom dataset test split, runs inference,
and plots the 12×12 confusion matrix as a heatmap.

Requires:
    A trained UltraTamNet .keras model (from experiments/train_augmentation_study.py)
    The custom Tamil vowel dataset directory (raw or augmented)

Usage:
    python figures/plot_confusion_matrix.py \
        --model_path outputs/table6/x10/UltraTamNet.keras \
        --ds_path CUSTOM/OP/new
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.preprocess_custom import load_custom_dataset


TAM_VOWELS = ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to trained UltraTamNet .keras model for custom dataset")
    parser.add_argument("--ds_path", required=True,
                        help="Path to custom Tamil vowel dataset directory")
    parser.add_argument("--use_tamil_labels", action="store_true", default=True,
                        help="Use Tamil vowel Unicode labels on axes (default: True)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found: {args.model_path}\n"
            "Run:  python experiments/train_augmentation_study.py "
            "--raw_dir CUSTOM/OP/new --aug_dir CUSTOM/OP/augmented"
        )

    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)

    print(f"Loading custom dataset from {args.ds_path} ...")
    _, x_test, _, y_test_oh = load_custom_dataset(args.ds_path)

    n_classes = y_test_oh.shape[1]
    if args.use_tamil_labels and n_classes <= len(TAM_VOWELS):
        class_labels = TAM_VOWELS[:n_classes]
    else:
        class_labels = [str(i) for i in range(n_classes)]

    y_true = np.argmax(y_test_oh, axis=1)
    print(f"Running inference on {len(x_test)} test samples ...")
    y_pred = np.argmax(model.predict(x_test, batch_size=32, verbose=1), axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    acc = np.mean(y_true == y_pred) * 100
    print(f"\nOverall test accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm_norm / row_sums, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Normalised (row %)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Confusion Matrix (row-normalised)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    fig.suptitle(
        f"Fig. 15 — Confusion Matrix: UltraTamNet on Custom Tamil Vowel Dataset\n"
        f"(Test accuracy: {acc:.2f}%)",
        fontsize=12,
    )
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/fig15_confusion_matrix.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
