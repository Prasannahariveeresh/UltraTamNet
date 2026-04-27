"""
Table V — Top-N Most Confused Class Pairs in Testing of UltraTamNet.

Loads a trained UltraTamNet model and the uTHCD test split,
runs inference, builds the confusion matrix, and reports the
top confused pairs ranked by confusion percentage.

Requires:
    Run experiments/train_uthcd_benchmark.py first (saves UltraTamNet.keras).

Generates:
    outputs/tables/table5_confused_pairs.csv
    outputs/tables/table5_confused_pairs.png

Usage:
    python tables/generate_confusion_pairs.py \
        --model_path outputs/table3/UltraTamNet.keras \
        --ds_path    tamil-handwritten-character-recognition \
        --top_n      10
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.preprocess_uthcd import load_uthcd


def top_confused_pairs(y_true, y_pred, class_names, top_n=10):
    cm = confusion_matrix(y_true, y_pred)
    row_totals = cm.sum(axis=1)

    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    flat_indices = np.argsort(cm_no_diag.ravel())[::-1]
    pairs = np.array(np.unravel_index(flat_indices, cm_no_diag.shape)).T

    rows = []
    for true_cls, pred_cls in pairs:
        count = cm_no_diag[true_cls, pred_cls]
        if count == 0:
            break
        pct = (count / row_totals[true_cls]) * 100
        rows.append({
            "True Class (index)":      true_cls,
            "True Class (label)":      class_names[true_cls] if class_names else str(true_cls),
            "Predicted Class (index)": pred_cls,
            "Predicted Class (label)": class_names[pred_cls] if class_names else str(pred_cls),
            "Confusion (%)":           round(pct, 2),
            "Count":                   int(count),
            "Total Samples":           int(row_totals[true_cls]),
        })
        if len(rows) >= top_n:
            break

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to saved UltraTamNet.keras model")
    parser.add_argument("--ds_path",    required=True,
                        help="Path to uTHCD dataset root")
    parser.add_argument("--top_n",      type=int, default=10,
                        help="Number of top confused pairs to report")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found: {args.model_path}\n"
            "Run:  python experiments/train_uthcd_benchmark.py --ds_path <path> --model UltraTamNet"
        )

    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)

    print(f"Loading uTHCD test data from {args.ds_path} ...")
    _, x_test, x_valid, _, y_test, y_valid, class_names = load_uthcd(args.ds_path)

    # Use the test split (same as used during training evaluation)
    print(f"Running inference on {len(x_test)} test samples ...")
    y_probs = model.predict(x_test, batch_size=32, verbose=1)
    y_pred  = np.argmax(y_probs, axis=1)
    y_true  = np.argmax(y_test, axis=1)

    df = top_confused_pairs(y_true, y_pred, class_names, top_n=args.top_n)

    print(f"\nTable V — Top {args.top_n} Most Confused Class Pairs")
    print("=" * 85)
    print(df.to_string(index=False))

    os.makedirs("outputs/tables", exist_ok=True)
    csv_path = "outputs/tables/table5_confused_pairs.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")

    # Render as table image
    fig, ax = plt.subplots(figsize=(13, len(df) * 0.55 + 1.5))
    ax.axis("off")

    row_colors = [["#F2F3F4" if i % 2 == 0 else "#FFFFFF"] * len(df.columns) for i in range(len(df))]

    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")

    plt.title(f"Table V — Top {args.top_n} Most Confused Class Pairs (UltraTamNet on uTHCD)",
              fontsize=11, pad=12)
    plt.tight_layout()
    png_path = "outputs/tables/table5_confused_pairs.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved PNG  → {png_path}")


if __name__ == "__main__":
    main()
