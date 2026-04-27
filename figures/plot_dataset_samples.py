"""
Fig. 7 — Sample images from the custom Tamil vowel dataset collected from volunteers
         displayed in a 6×10 grid (as in the paper).

Loads actual images from the raw dataset directory.

Usage:
    python figures/plot_dataset_samples.py --ds_path CUSTOM/OP/new
    python figures/plot_dataset_samples.py --ds_path CUSTOM/OP/augmented --samples_per_class 10
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

TAM_VOWELS = ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", required=True,
                        help="Path to custom dataset directory (raw or augmented)")
    parser.add_argument("--samples_per_class", type=int, default=5,
                        help="Number of sample images to show per class")
    args = parser.parse_args()

    if not os.path.isdir(args.ds_path):
        raise FileNotFoundError(f"Dataset directory not found: {args.ds_path}")

    classes = sorted(os.listdir(args.ds_path))
    classes = [c for c in classes if os.path.isdir(os.path.join(args.ds_path, c))]
    n_classes = len(classes)
    n_cols    = args.samples_per_class
    n_rows    = n_classes

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8))

    for row_idx, cls in enumerate(classes):
        cls_dir = os.path.join(args.ds_path, cls)
        imgs    = sorted(os.listdir(cls_dir))[:n_cols]

        label = TAM_VOWELS[int(cls)] if cls.isdigit() and int(cls) < len(TAM_VOWELS) else cls

        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            if col_idx < len(imgs):
                img_path = os.path.join(cls_dir, imgs[col_idx])
                img = Image.open(img_path).convert("L")
                img = ImageOps.fit(img, (128, 128), Image.LANCZOS)
                ax.imshow(np.array(img), cmap="gray")
            else:
                ax.axis("off")
                continue

            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=12, rotation=0, labelpad=30, va="center")

    plt.suptitle("Fig. 7 — Custom Tamil Vowel Dataset Samples (collected from volunteers)",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/fig7_dataset_samples.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
