"""
Fig. 8 — Step-by-step visualization of image preprocessing applied to the custom dataset.

Steps shown (matching the paper):
  1. Gray Image
  2. Blurred Image    (Gaussian blur 3×3)
  3. Binary Image     (Adaptive thresholding)
  4. Morphological Opening (2×2 kernel)
  5. CLAHE Applied

Loads a real image from the custom dataset.

Usage:
    python figures/plot_preprocessing_steps.py --image_path CUSTOM/OP/new/0/sample.png
    python figures/plot_preprocessing_steps.py --ds_path CUSTOM/OP/new --class_idx 0
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def preprocess_with_steps(rgb_image: np.ndarray):
    """Apply preprocessing pipeline and return intermediate images at each step."""
    steps = {}

    # Step 1: Grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    steps["Gray Image"] = gray

    # Step 2: Resize + Gaussian blur
    gray_resized = cv2.resize(gray, (256, 256))
    blurred = cv2.GaussianBlur(gray_resized, (3, 3), 0)
    steps["Blurred Image"] = blurred

    # Step 3: Adaptive thresholding → binary
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=2,
    )
    steps["Binary Image"] = binary

    # Step 4: Morphological opening (removes noise while keeping strokes)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    steps["Morphological Opening"] = opened

    # Step 5: CLAHE on inverted image
    inverted = cv2.bitwise_not(opened)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result   = clahe.apply(inverted)
    steps["CLAHE Applied"] = result

    return steps


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.fit(img, (128, 128), Image.LANCZOS)
    return np.array(img)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image_path", help="Direct path to a single image file")
    group.add_argument("--ds_path",    help="Dataset root; picks first image from --class_idx")
    parser.add_argument("--class_idx", type=int, default=0,
                        help="Class index to pick sample from (used with --ds_path)")
    args = parser.parse_args()

    if args.image_path:
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"Image not found: {args.image_path}")
        img_rgb = load_image(args.image_path)
        img_label = os.path.basename(args.image_path)
    else:
        cls_dir = os.path.join(args.ds_path, str(args.class_idx))
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Class directory not found: {cls_dir}")
        first_img = sorted(os.listdir(cls_dir))[0]
        img_path  = os.path.join(cls_dir, first_img)
        img_rgb   = load_image(img_path)
        img_label = f"Class {args.class_idx} — {first_img}"

    steps = preprocess_with_steps(img_rgb)

    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 3.5))

    for ax, (title, img) in zip(axes, steps.items()):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Fig. 8 — Preprocessing Pipeline  ({img_label})", fontsize=12)
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/fig8_preprocessing_steps.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
