"""
Offline augmentation pipeline for the custom Tamil vowel dataset.

Input:  raw handwritten images in <input_dir>/<class_label>/*.png
Output: preprocessed + augmented images in <output_dir>/<class_label>/aug_N.png

Preprocessing steps applied to every image before augmentation:
  1. Binarize (Otsu threshold)
  2. Remove small noise components (connected-component area filter)
  3. Correct skew (brute-force angle search)

Augmentation (pure OpenCV + NumPy — no imgaug dependency):
  - Random rotation ±10°
  - Random scale 0.8–1.2×
  - Random translation ±10%
  - Additive Gaussian noise
  - Contrast normalization
  - Random crop up to 10%

Usage:
  python augment_custom_dataset.py \
      --input_dir  CUSTOM/OP/new \
      --output_dir CUSTOM/OP/augmented \
      --multiplier 10

  --multiplier controls the augmentation multiple used in Table 6.
  (×4 → multiplier=4, ×6 → multiplier=6, ×8 → multiplier=8, ×10 → multiplier=10)
  target_samples = original_count * multiplier
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image, ImageOps
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def remove_noise(image: np.ndarray) -> np.ndarray:
    filtered = cv2.medianBlur(image, 3)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filtered, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] <= 10:
            filtered[labels == i] = 0
    return filtered


def correct_skew(image: np.ndarray) -> np.ndarray:
    best_angle, max_white = 0, 0
    for angle in np.arange(-5, 6, 2):
        rotated = Image.fromarray(image).rotate(angle, expand=True, fillcolor=255)
        white = np.sum(np.array(rotated) == 255)
        if white > max_white:
            max_white, best_angle = white, angle

    step = 1.0
    while step >= 0.1:
        rp = Image.fromarray(image).rotate(best_angle + step, expand=True, fillcolor=255)
        rn = Image.fromarray(image).rotate(best_angle - step, expand=True, fillcolor=255)
        wp, wn = np.sum(np.array(rp) == 255), np.sum(np.array(rn) == 255)
        if wp > max_white:
            max_white, best_angle = wp, best_angle + step
        elif wn > max_white:
            max_white, best_angle = wn, best_angle - step
        step /= 2

    return np.array(Image.fromarray(image).rotate(best_angle, expand=True, fillcolor=255))


# ---------------------------------------------------------------------------
# Augmentation (pure OpenCV / NumPy)
# ---------------------------------------------------------------------------

def augment_image(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random augmentation to a grayscale image. Returns same-type uint8 array."""
    h, w = image.shape[:2]
    img = image.astype(np.float32)

    # Random rotation ±10°
    angle = rng.uniform(-10, 10)
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M_rot, (w, h), borderValue=255.0)

    # Random scale 0.8–1.2×
    scale = rng.uniform(0.8, 1.2)
    new_h, new_w = int(h * scale), int(w * scale)
    img_scaled = cv2.resize(img, (new_w, new_h))
    if scale >= 1.0:
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        img = img_scaled[start_y:start_y + h, start_x:start_x + w]
    else:
        pad_top  = (h - new_h) // 2
        pad_left = (w - new_w) // 2
        img = np.full((h, w), 255.0, dtype=np.float32)
        img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_scaled

    # Random translation ±10%
    tx = rng.uniform(-0.1, 0.1) * w
    ty = rng.uniform(-0.1, 0.1) * h
    M_trans = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    img = cv2.warpAffine(img, M_trans, (w, h), borderValue=255.0)

    # Additive Gaussian noise (σ up to 5% of pixel range)
    noise = rng.normal(0, rng.uniform(0, 0.05 * 255), img.shape).astype(np.float32)
    img = img + noise

    # Contrast normalization (multiply contrast by 0.75–1.5)
    mean_val = img.mean()
    contrast_factor = rng.uniform(0.75, 1.5)
    img = (img - mean_val) * contrast_factor + mean_val

    # Random crop up to 10%
    crop_frac = rng.uniform(0, 0.10)
    if crop_frac > 0:
        cy = int(h * crop_frac / 2)
        cx = int(w * crop_frac / 2)
        if cy > 0 and cx > 0:
            cropped = img[cy:h - cy, cx:w - cx]
            img = cv2.resize(cropped, (w, h))

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_images(input_dir: str, output_dir: str, multiplier: int = 10, seed: int = 42):
    """
    Preprocess all raw images and augment to reach target_samples per class.
    target_samples = original_count_per_class * multiplier
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    all_images, labels = [], []

    for class_label in tqdm(
        sorted(os.listdir(input_dir), key=lambda x: int(x) if x.isdigit() else x),
        desc="Preprocessing",
    ):
        class_in  = os.path.join(input_dir, class_label)
        class_out = os.path.join(output_dir, class_label)
        if not os.path.isdir(class_in):
            continue
        os.makedirs(class_out, exist_ok=True)

        for fname in os.listdir(class_in):
            fpath = os.path.join(class_in, fname)
            img = Image.open(fpath).convert("RGB")
            img = ImageOps.fit(img, (128, 128), Image.LANCZOS)
            img_np = np.array(img)
            img_np = binarize(img_np)
            img_np = remove_noise(img_np)
            img_np = correct_skew(img_np)
            all_images.append(img_np)
            labels.append(class_label)

    original_count = len(all_images)
    target_samples  = original_count * multiplier

    print(f"Original samples: {original_count}  |  Target (×{multiplier}): {target_samples}")

    aug_images, aug_labels = [], []
    needed = target_samples - original_count
    idx = 0
    while len(aug_images) < needed:
        aug_images.append(augment_image(all_images[idx % original_count], rng))
        aug_labels.append(labels[idx % original_count])
        idx += 1

    all_out_images = all_images + aug_images
    all_out_labels = labels + aug_labels

    for i, (img, lbl) in enumerate(zip(all_out_images, all_out_labels)):
        out_path = os.path.join(output_dir, lbl, f"aug_{i}.png")
        Image.fromarray(img).save(out_path)

    print(f"Saved {len(all_out_images)} images to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment custom Tamil vowel dataset")
    parser.add_argument("--input_dir",  default="CUSTOM/OP/new",       help="Raw image directory")
    parser.add_argument("--output_dir", default="CUSTOM/OP/augmented", help="Output directory")
    parser.add_argument("--multiplier", type=int, default=10,
                        help="Augmentation multiple (4, 6, 8, or 10 — matches Table 6)")
    args = parser.parse_args()
    process_images(args.input_dir, args.output_dir, multiplier=args.multiplier)
