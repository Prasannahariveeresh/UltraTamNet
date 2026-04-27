"""
Table II — UltraTamNet Architectural Structure and Parameters.

Derives the table DIRECTLY from the model by inspecting its layers at runtime.
No hardcoded values — everything comes from the actual Keras model object.

Generates:
    outputs/tables/table2_architecture.csv
    outputs/tables/table2_architecture.png

Usage:
    python tables/generate_architecture_table.py
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.ultratamnet import build_ultratamnet


def parse_layer_row(layer):
    """Extract stage, type, kernel, filters, output_size from a Keras layer."""
    cfg  = layer.get_config()
    name = layer.name
    ltype = type(layer).__name__

    kernel = "—"
    filters = "—"
    out_shape = "—"

    try:
        out_shape = "×".join(str(d) for d in layer.output_shape[1:])
    except Exception:
        pass

    if ltype == "Conv2D":
        kernel  = "×".join(str(k) for k in cfg.get("kernel_size", []))
        filters = str(cfg.get("filters", "—"))
    elif ltype == "SeparableConv2D":
        kernel  = "×".join(str(k) for k in cfg.get("kernel_size", []))
        filters = str(cfg.get("filters", "—"))
    elif ltype == "MaxPooling2D":
        kernel  = "×".join(str(k) for k in cfg.get("pool_size", []))
    elif ltype == "Dense":
        filters = str(cfg.get("units", "—"))
    elif ltype == "GlobalAveragePooling2D":
        ltype = "GAP"

    return ltype, kernel, filters, out_shape, name


STAGE_KEYWORDS = {
    "conv2d":                "CB",
    "max_pooling2d":         "CB",
    "separable_conv2d":      "SCB/Deep",
    "global_average_pooling": "Head",
    "dense":                 "Head",
    "add":                   None,
    "batch_normalization":   None,
    "re_lu":                 None,
    "activation":            None,
    "dropout":               None,
    "input":                 None,
}


def assign_stage(layer_name, index, total):
    """Rough stage assignment by layer index position."""
    n = layer_name.lower()
    if "input" in n:
        return "Input"
    if index <= 3:
        return "CB"
    if index <= 9:
        return "SCB"
    if index <= int(total * 0.75):
        return "Deep"
    return "Head"


STAGE_COLORS = {
    "Input": "#FFFFFF",
    "CB":    "#D6E4F0",
    "SCB":   "#D5F5E3",
    "Deep":  "#FDEBD0",
    "Head":  "#E8DAEF",
}


def main():
    model = build_ultratamnet(input_shape=(64, 64, 1), num_classes=156)

    rows = []
    named_layers = [(i, l) for i, l in enumerate(model.layers)
                    if type(l).__name__ not in
                    ("BatchNormalization", "ReLU", "Activation", "Add", "Dropout", "InputLayer")]

    for seq_idx, (orig_idx, layer) in enumerate(named_layers):
        ltype, kernel, filters, out_shape, lname = parse_layer_row(layer)
        stage = assign_stage(lname, seq_idx, len(named_layers))
        rows.append({
            "Stage":       stage,
            "Layer Type":  ltype,
            "Kernel":      kernel,
            "Filters":     filters,
            "Output Size": out_shape,
        })

    df = pd.DataFrame(rows)

    os.makedirs("outputs/tables", exist_ok=True)
    csv_path = "outputs/tables/table2_architecture.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV → {csv_path}")

    # Format as table image
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.38 + 1.5))
    ax.axis("off")

    row_colors = [[STAGE_COLORS.get(r["Stage"], "#FFFFFF")] * len(df.columns) for _, r in df.iterrows()]

    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=list(df.columns),
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")

    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in STAGE_COLORS.items() if k != "Input"]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.9)

    plt.title("Table II — UltraTamNet Architectural Structure and Parameters\n"
              f"(derived from model at runtime  |  total params: {model.count_params():,})",
              fontsize=10, pad=10)
    plt.tight_layout()
    png_path = "outputs/tables/table2_architecture.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved PNG  → {png_path}")

    print(f"\nTotal parameters: {model.count_params():,}  ({model.count_params()/1e6:.2f} M)")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
