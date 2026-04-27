"""
Table IV — Ablation Study Results of UltraTamNet.

Loads the CSV produced by experiments/ablation_study.py and renders it
as a publication-quality formatted table image.

Requires:
    Run experiments/ablation_study.py first.

Generates:
    outputs/tables/table4_ablation.png

Usage:
    python tables/generate_ablation_table.py --results_csv outputs/ablation/ablation_results.csv
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


ROW_COLORS = ["#FADBD8", "#FDEBD0", "#D5F5E3", "#D6E4F0", "#E8DAEF"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_csv", required=True,
        help="Path to ablation_results.csv produced by experiments/ablation_study.py",
    )
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        raise FileNotFoundError(
            f"Results file not found: {args.results_csv}\n"
            "Run:  python experiments/ablation_study.py --ds_path <uTHCD_path>"
        )

    df = pd.read_csv(args.results_csv)

    print("\nTable IV — Ablation Study Results")
    print("=" * 100)
    print(df.to_string(index=False))

    os.makedirs("outputs/tables", exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, len(df) * 0.7 + 1.5))
    ax.axis("off")

    row_colors = [[ROW_COLORS[i % len(ROW_COLORS)]] * len(df.columns) for i in range(len(df))]

    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.0)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")

    # Highlight the full UltraTamNet row (A5)
    for c in range(len(df.columns)):
        tbl[(len(df), c)].set_facecolor("#F1948A")
        tbl[(len(df), c)].set_text_props(fontweight="bold")

    plt.title("Table IV — Ablation Study Results of UltraTamNet\n"
              "(mean ± std over multiple runs on uTHCD, 156 classes)",
              fontsize=11, pad=12)
    plt.tight_layout()
    png_path = "outputs/tables/table4_ablation.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"\nSaved PNG → {png_path}")


if __name__ == "__main__":
    main()
