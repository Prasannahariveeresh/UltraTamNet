"""
Table III — Evaluation of UltraTamNet with SOTA Models (uTHCD, 156 classes).

Loads the CSV produced by experiments/train_uthcd_benchmark.py and renders it
as a formatted publication-quality table image.

Requires:
    Run experiments/train_uthcd_benchmark.py first.

Generates:
    outputs/tables/table3_sota_comparison.png

Usage:
    python tables/generate_benchmark_table.py --results_csv outputs/table3/table3_results.csv
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CATEGORY_MAP = {
    "ResNet50":         "CNN",
    "DenseNet169":      "CNN",
    "DenseNet121":      "CNN",
    "Xception":         "CNN",
    "EfficientNetB0":   "Lightweight",
    "EfficientNetB5":   "Lightweight",
    "MobileNetV2":      "Lightweight",
    "MobileNetV3Small": "Lightweight",
    "MobileNetV3Large": "Lightweight",
    "NASNetMobile":     "Lightweight",
    "LeNet-5":          "Baseline",
    "UltraTamNet":      "Tamil-Optimized",
}

CATEGORY_COLORS = {
    "CNN":             "#AED6F1",
    "Lightweight":     "#A9DFBF",
    "Baseline":        "#F9E79F",
    "Tamil-Optimized": "#F1948A",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_csv", required=True,
        help="Path to table3_results.csv produced by experiments/train_uthcd_benchmark.py",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        raise FileNotFoundError(
            f"Results file not found: {args.results_csv}\n"
            "Run:  python experiments/train_uthcd_benchmark.py --ds_path <uTHCD_path>"
        )

    df = pd.read_csv(args.results_csv)

    # Add Category column if not present
    if "Category" not in df.columns:
        df["Category"] = df["Model"].map(CATEGORY_MAP).fillna("—")

    # Sort: best accuracy last (UltraTamNet highlighted at bottom)
    df = df.sort_values("Test Acc (%)", ascending=True).reset_index(drop=True)

    print("\nTable III — SOTA Model Comparison on uTHCD")
    print("=" * 80)
    print(df.to_string(index=False))

    os.makedirs("outputs/tables", exist_ok=True)

    # Save formatted table image
    display_cols = ["Model", "Category", "Test Acc (%)", "Test Loss", "F1-Score", "Params (M)", "FLOPs (M)"]
    display_df   = df[[c for c in display_cols if c in df.columns]]

    fig, ax = plt.subplots(figsize=(14, len(display_df) * 0.55 + 1.5))
    ax.axis("off")

    row_colors = []
    for _, row in display_df.iterrows():
        cat   = CATEGORY_MAP.get(row["Model"], "—")
        color = CATEGORY_COLORS.get(cat, "#FFFFFF")
        row_colors.append([color] * len(display_df.columns))

    tbl = ax.table(
        cellText=display_df.values.tolist(),
        colLabels=list(display_df.columns),
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

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, bbox_to_anchor=(1.0, 1.0))

    plt.title("Table III — Evaluation of UltraTamNet with SOTA Models (uTHCD, 156 classes)",
              fontsize=11, pad=12)
    plt.tight_layout()
    png_path = "outputs/tables/table3_sota_comparison.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"\nSaved PNG → {png_path}")


if __name__ == "__main__":
    main()
