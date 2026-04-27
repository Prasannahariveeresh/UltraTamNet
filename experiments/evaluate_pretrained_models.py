"""
One-time recovery script: evaluates already-trained .keras models from table3
that are missing from the CSV, then appends them.

Run once before restarting table3_train_uthcd.py with --skip_existing.
"""

import os
import sys
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.preprocess_uthcd import load_uthcd
from utils.evaluate import compute_metrics, get_flops

DS_PATH    = "/home/pras/Desktop/Tamil/tamil-handwritten-character-recognition"
OUTPUT_DIR = "outputs/table3"
CSV_PATH   = os.path.join(OUTPUT_DIR, "table3_results.csv")

COMPLETED = ["UltraTamNet", "ResNet50", "DenseNet121", "DenseNet169", "EfficientNetB0"]

print("Loading uTHCD dataset ...")
x_train, x_test, x_valid, y_train, y_test, y_valid, class_names = load_uthcd(DS_PATH)
print(f"  Valid={x_valid.shape}")

existing_df = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()
already_done = set(existing_df["Model"].tolist()) if not existing_df.empty else set()
print(f"Already in CSV: {already_done}")

rows = []
for name in COMPLETED:
    if name in already_done:
        print(f"  {name}: already in CSV, skipping")
        continue
    keras_path = os.path.join(OUTPUT_DIR, f"{name}.keras")
    if not os.path.exists(keras_path):
        print(f"  {name}: .keras not found, skipping")
        continue

    print(f"\n  Evaluating {name} ...")
    model = tf.keras.models.load_model(keras_path)

    params = model.count_params()
    print(f"    params={params/1e6:.2f} M  — computing FLOPs ...")
    flops = get_flops(model)

    metrics = compute_metrics(model, x_valid, y_valid)
    print(f"    acc={metrics['accuracy']:.2f}%  f1={metrics['f1']:.3f}")

    rows.append({
        "Model":        name,
        "Test Acc (%)": round(metrics["accuracy"], 2),
        "Test Loss":    round(metrics["loss"], 4),
        "F1-Score":     round(metrics["f1"], 3),
        "Params (M)":   round(params / 1e6, 2),
        "FLOPs (M)":    round(flops / 1e6, 1),
    })

    # Save after each model
    combined = pd.concat([existing_df, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"    Saved to CSV ({len(combined)} rows total)")

print("\nDone. Current CSV:")
print(pd.read_csv(CSV_PATH).to_string(index=False))
