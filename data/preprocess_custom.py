"""
Dataset 2 — Custom Tamil Vowels Dataset
12-class handwritten Tamil vowel dataset collected by the authors.

Classes (Tamil vowels):
  அ  ஆ  இ  ஈ  உ  ஊ  எ  ஏ  ஐ  ஒ  ஓ  ஔ
  (label indices 0–11, folder names are 0–11)

This script loads the *already-augmented* version of the dataset
(produced by augmentation/augment_custom_dataset.py).

Expected folder structure:
  <ds_path>/
      0/   <- அ
          aug_0.png
          ...
      1/   <- ஆ
          ...
      ...
      11/  <- ஔ

Images are resized to 64x64 and converted to grayscale.
Salt-and-pepper noise is optionally injected to simulate real-world degradation.

Used in: Table 6 (augmentation study), custom dataset evaluation.
"""

import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

TAM_VOWELS = ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ"]
NUM_CLASSES = 12


def _add_salt_pepper_noise(img):
    """Randomly flip pixels to white or black — simulates real handwriting noise."""
    row, col, _ = img.shape
    for _ in range(random.randint(150, 500)):
        img[random.randint(0, row - 1)][random.randint(0, col - 1)] = 255
    for _ in range(random.randint(150, 500)):
        img[random.randint(0, row - 1)][random.randint(0, col - 1)] = 0
    return img


def load_custom_dataset(ds_path: str, add_noise: bool = True, target_size=(64, 64)):
    """
    Load the augmented custom Tamil vowel dataset.

    Args:
        ds_path:     Path to the augmented dataset root (e.g., CUSTOM/OP/augmented).
        add_noise:   Whether to inject salt-and-pepper noise (used during training).
        target_size: Output image size (H, W). Default 64x64.

    Returns:
        x_train, x_test  — float32 arrays, shape (N, H, W, 1)
        y_train, y_test  — one-hot arrays, shape (N, 12)
    """
    X, y = [], []
    for label in tqdm(sorted(os.listdir(ds_path), key=lambda x: int(x) if x.isdigit() else x), desc="Loading custom dataset"):
        class_dir = os.path.join(ds_path, label)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img = cv2.imread(os.path.join(class_dir, img_name))
            if img is None:
                continue
            if add_noise:
                img = _add_salt_pepper_noise(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, target_size)
            X.append(gray)
            y.append(label)

    X = np.array(X, dtype="float32") / 255.0   # normalise to [0, 1]
    y = np.array(y)

    y_int = y.astype(int)
    y_cat = to_categorical(y_int, NUM_CLASSES)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_int
    )

    x_train = np.expand_dims(x_train, axis=-1)
    x_test  = np.expand_dims(x_test,  axis=-1)

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify custom dataset loading")
    parser.add_argument(
        "--ds_path",
        default="CUSTOM/OP/augmented",
        help="Path to the augmented custom dataset directory",
    )
    parser.add_argument("--no_noise", action="store_true", help="Disable noise injection")
    args = parser.parse_args()

    x_train, x_test, y_train, y_test = load_custom_dataset(
        args.ds_path, add_noise=not args.no_noise
    )
    print(f"Train : {x_train.shape}  labels: {y_train.shape}")
    print(f"Test  : {x_test.shape}   labels: {y_test.shape}")
    print(f"Classes: {NUM_CLASSES}  ({TAM_VOWELS})")
