"""
Table VII — Benchmarking UltraTamNet with Recent Research.

This table compares UltraTamNet against published THCR methods from the literature.
The values for prior works are as reported in those original papers (cited references).
The UltraTamNet row is loaded from the training results CSV if available,
otherwise it raises an error directing you to run training first.

Requires (for the UltraTamNet row):
    outputs/table3/table3_results.csv  (from experiments/train_uthcd_benchmark.py)
    outputs/table6/table6_results.csv  (from experiments/train_augmentation_study.py)

Generates:
    outputs/tables/table7_benchmarking.csv
    outputs/tables/table7_benchmarking.png

Usage:
    python tables/generate_sota_comparison.py \
        --table3_csv outputs/table3/table3_results.csv \
        --table6_csv outputs/table6/table6_results.csv
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Prior-work rows: values taken from respective cited papers.
# Format: [S.no, Citation, Authors, Dataset, No. of Classes, Method, Accuracy (%)]
PRIOR_WORKS = [
    [1,  "[31]", "R J Kannan et al. (2008)",        "Custom",                      10,  "Octal Graph",                     "82"],
    [2,  "[6]",  "N Shanthi et al. (2010)",          "Custom",                      34,  "SVM",                             "82.04"],
    [3,  "[19]", "RB Lincy et al. (2021)",           "HPLabs",                      156, "Self-Adaptive Lion Algorithm",    "84"],
    [4,  "[32]", "Mukundan V et al. (2023)",         "uTHCD",                       12,  "CatBoost",                        "84.36"],
    [5,  "[9]",  "MAR Raj et al. (2020)",            "HPLabs",                      100, "SVM",                             "90.3"],
    [6,  "[8]",  "MAR Raj et al. (2023)",            "HPLabs",                      125, "SVM",                             "90.31"],
    [7,  "[4]",  "N Shaffi et al. (2021)",           "uTHCD",                       156, "CNN",                             "92.32"],
    [8,  "[35]", "Jayachandran S et al. (2025)",     "Handwritten Tamil Vowel-13",  12,  "EfficientNet-B0+SE+CBAM",         "92.62"],
    [9,  "[42]", "R. Gayathri et al. (2021)",        "HPLabs",                      156, "Inception-v3 Transfer Learning",  "93.1"],
    [10, "[33]", "R Puvanendran et al. (2023)",      "Custom",                      12,  "CNN",                             "93.33"],
    [11, "[18]", "K Shanmugam et al. (2024)",        "HPLabs",                      156, "HBO-DBNN",                        "94"],
    [12, "[16]", "Vijayaraghavan et al. (2014)",     "HPLabs",                      35,  "CNN",                             "94.4"],
    [13, "[34]", "Kaliappan A V et al. (2020)",      "Custom",                      12,  "CNN + MLP",                       "95.58"],
    [14, "[12]", "R Jayakanth et al. (2020)",        "HPLabs",                      156, "CNN",                             "96"],
    [15, "[7]",  "C Sureshkumar et al. (2010)",      "HPLabs",                      156, "CNN",                             "97"],
    [16, "[15]", "BR Kavitha et al. (2022)",         "HPLabs",                      156, "CNN",                             "97.7"],
]

COLUMNS = ["S.no", "Citation", "Authors", "Dataset", "No. of Classes", "Method", "Accuracy (%)"]


def get_ultratamnet_rows(table3_csv, table6_csv):
    """Extract UltraTamNet results from training CSVs."""
    rows = []

    if table3_csv and os.path.exists(table3_csv):
        df3 = pd.read_csv(table3_csv)
        utm = df3[df3["Model"].str.contains("UltraTamNet", case=False, na=False)]
        if not utm.empty:
            acc = utm.iloc[0]["Test Acc (%)"]
            rows.append([17, "Ours", "This work", "uTHCD", 156, "UltraTamNet", str(acc)])
    else:
        raise FileNotFoundError(
            f"Table 3 results not found: {table3_csv}\n"
            "Run:  python experiments/train_uthcd_benchmark.py --ds_path <uTHCD_path>"
        )

    if table6_csv and os.path.exists(table6_csv):
        df6 = pd.read_csv(table6_csv)
        utm6 = df6[
            df6["Model"].str.contains("UltraTamNet", case=False, na=False) &
            df6["Augmentation"].astype(str).str.contains("10", na=False)
        ]
        if not utm6.empty:
            acc = utm6.iloc[0]["Test Acc (%)"]
            rows.append([17, "Ours", "This work", "Custom Tamil Vowels", 12, "UltraTamNet", str(acc)])
    else:
        raise FileNotFoundError(
            f"Table 6 results not found: {table6_csv}\n"
            "Run:  python experiments/train_augmentation_study.py --raw_dir ... --aug_dir ..."
        )

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table3_csv", required=True,
                        help="Path to table3_results.csv from training")
    parser.add_argument("--table6_csv", required=True,
                        help="Path to table6_results.csv from augmentation study")
    args = parser.parse_args()

    our_rows = get_ultratamnet_rows(args.table3_csv, args.table6_csv)
    all_rows = PRIOR_WORKS + our_rows
    df = pd.DataFrame(all_rows, columns=COLUMNS)

    print("\nTable VII — Benchmarking with Recent Research")
    print("=" * 90)
    print(df.to_string(index=False))

    os.makedirs("outputs/tables", exist_ok=True)
    csv_path = "outputs/tables/table7_benchmarking.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")

    fig, ax = plt.subplots(figsize=(15, len(df) * 0.5 + 1.8))
    ax.axis("off")

    row_colors = []
    for _, row in df.iterrows():
        if row["Citation"] == "Ours":
            row_colors.append(["#F1948A"] * len(COLUMNS))
        elif _ % 2 == 0:
            row_colors.append(["#F2F3F4"] * len(COLUMNS))
        else:
            row_colors.append(["#FFFFFF"] * len(COLUMNS))

    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=COLUMNS,
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
        elif r > 0 and df.iloc[r - 1]["Citation"] == "Ours":
            cell.set_text_props(fontweight="bold")

    plt.title("Table VII — Benchmarking UltraTamNet with Recent Research\n"
              "(highlighted row = our results from actual training)",
              fontsize=11, pad=12)
    plt.tight_layout()
    png_path = "outputs/tables/table7_benchmarking.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved PNG  → {png_path}")


if __name__ == "__main__":
    main()
