"""
Fig. 12 — Accuracy vs. Model Size comparison across all models on uTHCD.

Two sub-plots:
  (a) Scatter: Test accuracy vs. number of parameters (size of dot = FLOPs)
  (b) Bar: Test accuracy vs. FLOPs, models sorted by FLOPs

Loads data from the table3_results.csv produced by experiments/train_uthcd_benchmark.py.
UltraTamNet is highlighted in each chart.

Requires:
    outputs/table3/table3_results.csv   (from experiments/train_uthcd_benchmark.py)

Usage:
    python figures/plot_accuracy_vs_model_size.py \
        --results_csv outputs/table3/table3_results.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True,
                        help="Path to table3_results.csv from training")
    args = parser.parse_args()

    if not os.path.exists(args.results_csv):
        raise FileNotFoundError(
            f"Results file not found: {args.results_csv}\n"
            "Run:  python experiments/train_uthcd_benchmark.py --ds_path <uTHCD_path>"
        )

    df = pd.read_csv(args.results_csv)

    required = {"Model", "Test Acc (%)", "Params (M)", "FLOPs (M)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}\n"
                         f"Available columns: {list(df.columns)}")

    df["Test Acc (%)"] = pd.to_numeric(df["Test Acc (%)"], errors="coerce")
    df["Params (M)"]   = pd.to_numeric(df["Params (M)"],   errors="coerce")
    df["FLOPs (M)"]    = pd.to_numeric(df["FLOPs (M)"],    errors="coerce")
    df = df.dropna(subset=["Test Acc (%)", "Params (M)", "FLOPs (M)"])

    is_ours = df["Model"].str.contains("UltraTamNet", case=False, na=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- (a) Scatter: accuracy vs. params, dot size = FLOPs ---
    flops_norm = (df["FLOPs (M)"] / df["FLOPs (M)"].max()) * 600 + 40

    for _, row in df[~is_ours].iterrows():
        ax1.scatter(row["Params (M)"], row["Test Acc (%)"],
                    s=flops_norm[_], color="tab:blue", alpha=0.7, zorder=2)
        ax1.annotate(row["Model"], (row["Params (M)"], row["Test Acc (%)"]),
                     fontsize=7, ha="left", va="bottom",
                     xytext=(3, 3), textcoords="offset points")

    for _, row in df[is_ours].iterrows():
        ax1.scatter(row["Params (M)"], row["Test Acc (%)"],
                    s=flops_norm[_], color="tab:red", alpha=0.9, zorder=3, marker="*")
        ax1.annotate(row["Model"], (row["Params (M)"], row["Test Acc (%)"]),
                     fontsize=8, ha="left", va="bottom", color="tab:red", fontweight="bold",
                     xytext=(3, 3), textcoords="offset points")

    ax1.set_xlabel("Parameters (M)")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("(a) Accuracy vs. Model Size\n(dot area ∝ FLOPs)")
    ax1.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
               markersize=8, label="Baseline"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="tab:red",
               markersize=12, label="UltraTamNet"),
    ]
    ax1.legend(handles=legend_elems, fontsize=8)

    # --- (b) Bar: accuracy vs. FLOPs, sorted by FLOPs ---
    df_sorted = df.sort_values("FLOPs (M)")
    colors = ["tab:red" if is_ours.loc[i] else "tab:blue" for i in df_sorted.index]

    x = np.arange(len(df_sorted))
    ax2.bar(x, df_sorted["Test Acc (%)"], color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{row['Model']}\n({row['FLOPs (M)']:.0f}M)" for _, row in df_sorted.iterrows()],
        rotation=45, ha="right", fontsize=7,
    )
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("(b) Accuracy vs. FLOPs\n(sorted by FLOPs ascending)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 105)

    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(color="tab:blue", label="Baseline"),
        Patch(color="tab:red",  label="UltraTamNet"),
    ], fontsize=8)

    fig.suptitle("Fig. 12 — Accuracy vs. Model Complexity (uTHCD, 156 classes)", fontsize=12)
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/fig12_accuracy_vs_size.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
