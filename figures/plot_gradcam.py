"""
Fig. 13 — Grad-CAM heatmap grid for UltraTamNet on uTHCD test samples.

For each of N randomly selected test images, overlays the Grad-CAM heatmap
produced from the last SeparableConv2D layer onto the input image.

Requires:
    A trained UltraTamNet .keras model (from experiments/train_uthcd_benchmark.py)
    The uTHCD dataset directory

Usage:
    python figures/plot_gradcam.py \
        --model_path outputs/table3/UltraTamNet.keras \
        --ds_path /path/to/uTHCD \
        --n_samples 20
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.preprocess_uthcd import load_uthcd


def get_last_sepconv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.SeparableConv2D):
            return layer.name
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No SeparableConv2D or Conv2D layer found in the model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Compute Grad-CAM heatmap for img_array (shape: 1, H, W, C)."""
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap, int(pred_index.numpy())


def overlay_heatmap(img_gray, heatmap, alpha=0.5):
    """Overlay heatmap on grayscale image; returns RGB uint8 array."""
    h, w = img_gray.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_rgb = cv2.cvtColor(np.uint8(img_gray * 255) if img_gray.max() <= 1.0
                           else np.uint8(img_gray), cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to trained UltraTamNet .keras model")
    parser.add_argument("--ds_path", required=True,
                        help="Path to uTHCD dataset directory")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of test samples to visualize (default 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model not found: {args.model_path}\n"
            "Run:  python experiments/train_uthcd_benchmark.py --ds_path <uTHCD_path>"
        )

    print(f"Loading model from {args.model_path} ...")
    model = tf.keras.models.load_model(args.model_path)

    last_conv = get_last_sepconv_layer(model)
    print(f"Using Grad-CAM layer: {last_conv}")

    print(f"Loading uTHCD dataset from {args.ds_path} ...")
    _, x_test, _, _, y_test_oh, _, class_names = load_uthcd(args.ds_path)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(x_test), size=min(args.n_samples, len(x_test)), replace=False)

    n = len(indices)
    n_cols = min(5, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 3))
    axes = np.array(axes).reshape(-1)

    for plot_idx, data_idx in enumerate(indices):
        img = x_test[data_idx]
        true_label = int(np.argmax(y_test_oh[data_idx]))

        img_input = img[np.newaxis, ...]
        heatmap, pred_label = make_gradcam_heatmap(img_input, model, last_conv)

        img_2d = img[..., 0] if img.ndim == 3 else img
        overlay = overlay_heatmap(img_2d, heatmap)

        ax = axes[plot_idx]
        ax.imshow(overlay)
        correct = (pred_label == true_label)
        title_color = "green" if correct else "red"
        ax.set_title(
            f"T: {class_names[true_label]}\nP: {class_names[pred_label]}",
            fontsize=7, color=title_color,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Fig. 13 — Grad-CAM Visualizations (UltraTamNet on uTHCD)\n"
        f"Green title = correct, Red title = misclassified",
        fontsize=11,
    )
    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)
    save_path = "outputs/figures/fig13_gradcam.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    main()
