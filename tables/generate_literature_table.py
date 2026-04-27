"""
Table I — Summary of Handwritten Character Recognition Methods in Tamil and Other Languages.

This table is a literature survey compiled from published papers cited in the study.
The values (accuracy, dataset, classes) are as reported in those original papers.

Generates:
    outputs/tables/table1_literature_survey.csv
    outputs/tables/table1_literature_survey.png

Usage:
    python tables/generate_literature_table.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Data compiled from cited papers (references match the paper's bibliography)
SURVEY_DATA = [
    ["[6]",  "SVM with RBF kernel",                           "Tamil (Custom)",          "6048 samples", "62.8",  "Unable to distinguish visually similar characters"],
    ["[9]",  "Hierarchical SVM + Strip Tree + PM-Quad Tree",  "Tamil (HPLabs)",          "156",          "90.3",  "Sensitive to handwriting style variations"],
    ["[8]",  "Two-stage feature extraction + SVM",            "Tamil (Thirukural/TDS/HPLabs)", "156",    "90.3/87/89", "Sensitive to noise and lighting conditions"],
    ["[21]", "ELM + FFNN classifier",                         "Tamil postal dataset",    "1500 samples", "96.07", "Limited to 15 town names"],
    ["[14]", "CNN (4 Conv + Pooling layers)",                  "Tamil (HPLabs)",          "156",          "97",    "Limited evaluation on real-world noisy handwriting"],
    ["[16]", "Custom 9-layer CNN",                             "Tamil (IWFHR-10 subset)", "10",           "94.4",  "No comparison with modern deep architectures"],
    ["[17]", "CNN with Grid Search Optimization",              "Tamil (IWFHR-10)",        "10",           "96",    "Dataset lacks natural handwriting noise"],
    ["[10]", "Modified VGG16",                                 "Tamil (HPLabs)",          "156",          "91.8",  "Evaluated mainly on whiteboard digital input"],
    ["[11]", "AlexNet",                                        "Tamil (HPLabs)",          "156",          "92",    "High computational complexity"],
    ["[12]", "Fine-tuned ResNet",                              "Tamil (HPLabs)",          "156",          "96",    "Requires high computational resources"],
    ["[13]", "ResNet, InceptionV3, VGG16, VGG19",              "Tamil (uTHCD)",           "156",          "~90",   "Requires extensive fine-tuning"],
    ["[5]",  "ResNet50, VGG16, LeNet5 comparison",             "Tamil (uTHCD)",           "156",          ">80",   "Struggles with complex cursive variations"],
    ["[18]", "Hummingbird-Optimized DBNN (HBO-DBNN)",          "Tamil (HPLabs)",          "156",          "94",    "Limited dataset generalization"],
    ["[19]", "CNN + Self-Adaptive Lion Algorithm (SALA)",       "Tamil (HPLabs)",          "156",          "84",    "Lower accuracy than existing SOTA"],
    ["[22]", "GAN-based augmentation + CNN",                   "Tamil (Custom)",          "6000 samples", "96.2",  "Synthetic data may introduce bias"],
    ["[23]", "CRNN (CNN + LSTM/GRU)",                          "Tamil & English",         "60000 samples","92",    "Not evaluated on noisy cursive handwriting"],
    ["[39]", "Ensemble ML (BL + SVM + ANN)",                   "Tamil palm leaf",         "—",            "95.53", "Focused on manuscript segmentation"],
    ["[24]", "AlexNet + SVM",                                  "Urdu (UHat dataset)",     "~35000",       "—",     "Limited dataset diversity"],
    ["[25]", "Dual-Input CNN (DICNN)",                         "MNIST digits",            "10",           "98.7",  "Simple digit dataset"],
    ["[26]", "EMACRN (Attention-based CRNN)",                  "Online handwriting",      "~3500 classes","84.3",  "High computational cost (3.2B FLOPs)"],
    ["[37]", "CNN + Dilated Conv + CTC",                       "Cursive handwriting",     "~12000 images","CER 5.8%", "High training complexity"],
    ["[38]", "LU-Net (lightweight U-Net)",                     "Devanagari",              "1000 documents","94.03","FLOPs not evaluated"],
    ["[28]", "CNN (4 Conv + Pool layers)",                     "Sinhala (110k samples)",  "434",          "82.3",  "Performance drops for large class sets"],
    ["[29]", "CNN / MLP",                                      "Gujarati dataset",        "10000 samples","97.21/64.48", "Limited dataset diversity"],
    ["[30]", "VGG19",                                          "Kannada synthetic",       "188",          "73.51", "Synthetic data limits realism"],
    ["[36]", "IDMN + ELBP + LSTM + optimization",              "English / Kannada",       "~70000 images","96.66/96.67", "Complex feature extraction pipeline"],
    ["Ours", "UltraTamNet",                                    "uTHCD + Custom",          "156 / 12",     "98.2 / 99.8", "—"],
]

COLUMNS = ["Ref.", "Method / Architecture", "Script / Dataset", "Classes / Size", "Accuracy (%)", "Key Limitation"]


def main():
    df = pd.DataFrame(SURVEY_DATA, columns=COLUMNS)

    os.makedirs("outputs/tables", exist_ok=True)
    csv_path = "outputs/tables/table1_literature_survey.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV → {csv_path}")

    # Save as formatted PNG table
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values.tolist(),
        colLabels=COLUMNS,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif df.iloc[r - 1]["Ref."] == "Ours":
            cell.set_facecolor("#F1948A")
        elif r % 2 == 0:
            cell.set_facecolor("#F2F3F4")

    plt.title("Table I — Summary of Handwritten Character Recognition Methods in Tamil and Other Languages",
              fontsize=10, pad=12)
    plt.tight_layout()
    png_path = "outputs/tables/table1_literature_survey.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved PNG  → {png_path}")

    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
