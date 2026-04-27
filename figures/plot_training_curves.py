"""
Fig. 9  — Training/Validation Accuracy and Loss curves for UltraTamNet on uTHCD.
Fig. 14 — Training/Validation Accuracy and Loss curves for UltraTamNet on custom dataset.

Loads the training history JSON saved by the experiment scripts.

Requires:
    outputs/table3/UltraTamNet_history.json   (Fig. 9  — uTHCD)
    outputs/table6/x10/UltraTamNet_history.json  (Fig. 14 — custom dataset, ×10 aug)

Usage:
    # Fig. 9 (uTHCD)
    python figures/plot_training_curves.py \
        --history_json outputs/table3/UltraTamNet_history.json \
        --title "UltraTamNet on uTHCD" \
        --fig_num 9

    # Fig. 14 (custom dataset)
    python figures/plot_training_curves.py \
        --history_json outputs/table6/x10/UltraTamNet_history.json \
        --title "UltraTamNet on Custom Tamil Vowel Dataset" \
        --fig_num 14
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt


def plot_curves(history: dict, title: str, fig_num: int, save_path: str):
    has_lr = "lr" in history

    n_plots = 3 if has_lr else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # Accuracy
    axes[0].plot(history["accuracy"],     label="Train", color="tab:blue")
    axes[0].plot(history["val_accuracy"], label="Validation", color="tab:orange")
    axes[0].set_title("Training accuracy vs. Validation accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history["loss"],     label="Train", color="tab:blue")
    axes[1].plot(history["val_loss"], label="Validation", color="tab:orange")
    axes[1].set_title("Training loss vs. Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    # Learning rate (if present)
    if has_lr:
        axes[2].plot(history["lr"], color="tab:green")
        axes[2].set_title("Learning Rate Schedule")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].grid(alpha=0.3)

    fig.suptitle(f"Fig. {fig_num} — {title}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history_json", required=True,
                        help="Path to *_history.json saved by the experiment script")
    parser.add_argument("--title", default="UltraTamNet Training Curves",
                        help="Plot title describing the dataset/experiment")
    parser.add_argument("--fig_num", type=int, default=9,
                        help="Figure number (9 for uTHCD, 14 for custom dataset)")
    args = parser.parse_args()

    if not os.path.exists(args.history_json):
        raise FileNotFoundError(
            f"History file not found: {args.history_json}\n"
            "Run the relevant experiment script first to generate this file."
        )

    with open(args.history_json) as f:
        history = json.load(f)

    print(f"Loaded history from {args.history_json}")
    print(f"  Keys: {list(history.keys())}")
    print(f"  Epochs trained: {len(history['loss'])}")
    print(f"  Final train acc : {history['accuracy'][-1]:.4f}")
    print(f"  Final val   acc : {history['val_accuracy'][-1]:.4f}")
    print(f"  Final train loss: {history['loss'][-1]:.4f}")
    print(f"  Final val   loss: {history['val_loss'][-1]:.4f}")

    os.makedirs("outputs/figures", exist_ok=True)
    save_path = f"outputs/figures/fig{args.fig_num}_training_curves.png"
    plot_curves(history, args.title, args.fig_num, save_path)


if __name__ == "__main__":
    main()
