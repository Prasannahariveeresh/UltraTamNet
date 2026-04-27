"""
Fig. 16 — Real-time classification results from the Leave-One-Volunteer-Out (LOVO) test.

For each panel (handwriting style), displays test sample images side-by-side with
the model's recognition result (Tamil Unicode character), matching the layout in the paper.

The LOVO test excludes all images from one volunteer from training and evaluates
the trained model exclusively on that volunteer's handwriting.

Requires:
    A trained UltraTamNet .keras model (from experiments/train_augmentation_study.py)
    A directory of test images from the held-out volunteer, organised as:
        <lovo_dir>/
            0/   ← class 0 (அ)
            1/   ← class 1 (ஆ)
            ...
            11/  ← class 11 (ஔ)

Usage:
    python figures/fig16_lovo_test.py \
        --model_path outputs/table6/x10/UltraTamNet.keras \
        --lovo_dir CUSTOM/LOVO/volunteer_holdout \
        --samples_per_class 6
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from PIL import Image, ImageOps

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

TAM_VOWELS = ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ"]


def load_and_preprocess(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert("L")
    img = ImageOps.fit(img, target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[..., np.newaxis]


def collect_samples(lovo_dir, samples_per_class):
    """Return list of (class_idx, image_path) pairs, up to samples_per_class per class."""
    samples = []
    classes = sorted(
        [d for d in os.listdir(lovo_dir) if os.path.isdir(os.path.join(lovo_dir, d))],
        key=lambda x: int(x) if x.isdigit() else x,
    )
    for cls in classes:
        cls_dir = os.path.join(lovo_dir, cls)
        imgs = sorted(os.listdir(cls_dir))[:samples_per_class]
        for img_name in imgs:
            samples.append((int(cls) if cls.isdigit() else cls,
                            os.path.join(cls_dir, img_name)))
    return samples, classes


def run_lovo_inference(model, samples):
    results = []
    for true_cls, img_path in samples:
        arr = load_and_preprocess(img_path)
        pred_probs = model.predict(arr[np.newaxis, ...], verbose=0)
        pred_cls = int(np.argmax(pred_probs))
        results.append({
            "true": true_cls,
            "pred": pred_cls,
            "path": img_path,
            "correct": (pred_cls == true_cls),
        })
    return results


def plot_panel(results, panel_label, ax_images, ax_labels, n_cols, class_labels):
    """Fill one panel row: images on top sub-row, recognised characters below."""
    for col, r in enumerate(results[:n_cols]):
        ax = ax_images[col]
        img = Image.open(r["path"]).convert("RGB")
        ax.imshow(np.array(img))
        title_color = "green" if r["correct"] else "red"
        ax.set_title(
            f"True:{r['true']} Pred:{r['pred']}",
            fontsize=6, color=title_color,
        )
        ax.axis("off")

        ax2 = ax_labels[col]
        ax2.text(0.5, 0.5, class_labels[r["pred"]],
                 ha="center", va="center", fontsize=22,
                 color="green" if r["correct"] else "red",
                 fontproperties=_tamil_font())
        ax2.axis("off")

    for col in range(len(results), n_cols):
        ax_images[col].axis("off")
        ax_labels[col].axis("off")


def _tamil_font():
    """Return a FontProperties object for Tamil Unicode rendering if a font is available."""
    from matplotlib.font_manager import FontProperties
    for candidate in [
        "/usr/share/fonts/truetype/lohit-tamil/Lohit-Tamil.ttf",
        "/usr/share/fonts/truetype/noto/NotoSerifTamil-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf",
    ]:
        if os.path.exists(candidate):
            return FontProperties(fname=candidate)
    return FontProperties()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to trained UltraTamNet .keras model")
    parser.add_argument("--lovo_dir", required=True,
                        help="Directory of held-out volunteer images, sub-dirs 0..11")
    parser.add_argument("--samples_per_class", type=int, default=6,
                        help="Number of test images per class to show (default 6)")
    parser.add_argument("--n_panels", type=int, default=4,
                        help="Number of handwriting style panels to show (default 4)")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found: {args.model_path}\n"
            "Run:  python experiments/train_augmentation_study.py --raw_dir ... --aug_dir ..."
        )
    if not os.path.isdir(args.lovo_dir):
        raise FileNotFoundError(
            f"LOVO directory not found: {args.lovo_dir}\n"
            "Provide the directory of the held-out volunteer's test images."
        )

    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)

    samples, classes = collect_samples(args.lovo_dir, args.samples_per_class)
    n_classes = len(classes)
    class_labels = TAM_VOWELS[:n_classes] if n_classes <= len(TAM_VOWELS) else classes

    print(f"Running inference on {len(samples)} LOVO test samples ...")
    results = run_lovo_inference(model, samples)

    correct = sum(r["correct"] for r in results)
    lovo_acc = correct / len(results) * 100
    print(f"LOVO test accuracy: {lovo_acc:.2f}%  ({correct}/{len(results)})")

    n_cols = args.samples_per_class
    n_panels = min(args.n_panels, (len(results) + n_cols - 1) // n_cols)

    fig = plt.figure(figsize=(n_cols * 2, n_panels * 3.5))
    outer = gridspec.GridSpec(n_panels, 1, figure=fig, hspace=0.55)

    for panel_idx in range(n_panels):
        start = panel_idx * n_cols
        panel_results = results[start: start + n_cols]
        if not panel_results:
            break

        inner = gridspec.GridSpecFromSubplotSpec(
            2, n_cols, subplot_spec=outer[panel_idx], hspace=0.05,
        )
        ax_imgs   = [fig.add_subplot(inner[0, c]) for c in range(n_cols)]
        ax_labels = [fig.add_subplot(inner[1, c]) for c in range(n_cols)]
        plot_panel(panel_results, chr(ord("a") + panel_idx),
                   ax_imgs, ax_labels, n_cols, class_labels)

        ax_imgs[0].set_ylabel(f"({chr(ord('a') + panel_idx)})\nTest Sample",
                              fontsize=8, rotation=0, labelpad=55, va="center")

    fig.suptitle(
        f"Fig. 16 — Real-time LOVO Classification by UltraTamNet\n"
        f"(LOVO test accuracy: {lovo_acc:.2f}% | Green = correct, Red = misclassified)",
        fontsize=11,
    )

    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/fig16_lovo_test.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
