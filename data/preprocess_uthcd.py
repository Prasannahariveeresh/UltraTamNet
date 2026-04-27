"""
Dataset 1 — uTHCD (University of Tehran Handwritten Character Dataset)
156-class Tamil handwritten character recognition dataset.

Kaggle source:
  https://www.kaggle.com/datasets/sudalairajkumar/tamil-handwritten-character-recognition

Expected folder structure on disk:
  <ds_path>/
      train/
          <class_label>/
              image1.png
              ...
      test/
          <class_label>/
              image1.png
              ...

Images are expected to be 64x64 grayscale (the dataset ships at this size).
If your copy differs, pass resize=(H, W) to load_uthcd().

Used in: Table 3 (model comparison) and ablation study.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def load_uthcd(ds_path: str, resize=None):
    """
    Returns:
        x_train, x_test, x_valid  — float32 arrays, shape (N, H, W, 1)
        y_train, y_test, y_valid  — one-hot arrays, shape (N, 156)
        class_names               — sorted list of class label strings
    """
    def _load_split(split_path):
        X, y = [], []
        labels = sorted(os.listdir(split_path))
        for label in tqdm(labels, desc=f"Loading {os.path.basename(split_path)}"):
            class_dir = os.path.join(split_path, label)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img = cv2.imread(os.path.join(class_dir, img_name))
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if resize is not None:
                    gray = cv2.resize(gray, resize)
                X.append(gray)
                y.append(label)
        return X, y

    class_names = sorted(
        os.listdir(os.path.join(ds_path, "train")),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    x_train_raw, y_train_raw = _load_split(os.path.join(ds_path, "train"))
    x_test_raw,  y_test_raw  = _load_split(os.path.join(ds_path, "test"))

    x_train   = np.expand_dims(np.array(x_train_raw, dtype="float32"), axis=-1)
    x_test_all = np.expand_dims(np.array(x_test_raw,  dtype="float32"), axis=-1)

    y_train   = to_categorical(y_train_raw, len(class_names))
    y_test_all = to_categorical(y_test_raw,  len(class_names))

    # Hold out 10% of the test split as a validation set (same split used in paper)
    x_test, x_valid, y_test, y_valid = train_test_split(
        x_test_all, y_test_all,
        test_size=0.1,
        random_state=42,
        stratify=np.argmax(y_test_all, axis=1),
    )

    return x_train, x_test, x_valid, y_train, y_test, y_valid, class_names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify uTHCD loading")
    parser.add_argument(
        "--ds_path",
        default="tamil-handwritten-character-recognition",
        help="Path to the uTHCD dataset root directory",
    )
    args = parser.parse_args()

    x_train, x_test, x_valid, y_train, y_test, y_valid, class_names = load_uthcd(args.ds_path)
    print(f"Train : {x_train.shape}  labels: {y_train.shape}")
    print(f"Test  : {x_test.shape}   labels: {y_test.shape}")
    print(f"Valid : {x_valid.shape}  labels: {y_valid.shape}")
    print(f"Classes ({len(class_names)}): {class_names[:5]} ...")
