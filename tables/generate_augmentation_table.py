"""
Table VI — Comparative Performance Across Augmentation Ratios (Custom Tamil Vowel Dataset).

Loads the CSV produced by experiments/train_augmentation_study.py and renders it
as a publication-quality formatted table image.

Requires:
    Run experiments/train_augmentation_study.py first.

Generates:
    outputs/tables/table6_augmentation.png

Usage:
    python tables/generate_augmentation_table.py --results_csv outputs/table6/table6_results.csv
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

MODEL_COLORS = {
    "ResNet50":        "#AED6F1",
    "MobileNetV2":     "#A9DFBF",
    "DenseNet121":     "#F9E79F",
    "DenseNet169":     "#FAD7A0",
    "UltraTamNet":     "#F1948A",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_csv", required=True,
        help="Path to table6_results.csv produced by experiments/train_augmentation_study.py",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        raise FileNotFoundError(
            f"Results file not found: {args.results_csv}\n"
            "Run:  python experiments/train_augmentation_study.py --raw_dir ... --aug_dir ..."
        )

    df = pd.read_csv(args.results_csv)

    print("\nTable VI — Augmentation Study on Custom Tamil Vowel Dataset")
    print("=" * 100)
    print(df.to_string(index=False))

    os.makedirs("outputs/tables", exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, len(df) * 0.5 + 2))
    ax.axis("off")

    row_colors = []
    for _, row in df.iterrows():
        model = str(row["Model"]) if pd.notna(row["Model"]) else ""
        color = MODEL_COLORS.get(model, "#F2F3F4")
        row_colors.append([color] * len(df.columns))

    tbl = ax.table(
        cellText=df.fillna("").values.tolist(),
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.7)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in MODEL_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, bbox_to_anchor=(1.0, 1.0))

    plt.title("Table VI — Comparative Performance Across Augmentation Ratios\n"
              "(Custom Tamil Vowel Dataset, 12 classes)",
              fontsize=11, pad=12)
    plt.tight_layout()
    png_path = "outputs/tables/table6_augmentation.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"\nSaved PNG → {png_path}")


if __name__ == "__main__":
    main()
