"""
Evaluation utilities shared across all experiments.

Provides:
  - get_flops(model)                 — compute FLOPs via TF v1 profiler
  - compute_metrics(model, x, y)     — accuracy, loss, precision, recall, F1
  - plot_training_curves(hist)       — accuracy/loss/lr learning curves
  - plot_confusion_matrix(y_true, y_pred, class_names)
  - plot_false_pos_neg(y_true, y_pred, n_classes)
  - plot_classwise_accuracy(y_true, y_pred, num_classes)
  - plot_precision_recall_curve(y_true, y_probs, num_classes)
  - make_gradcam_heatmap(img, model, last_conv_layer_name)
  - plot_gradcam_grid(samples, labels, model, last_conv_layer)
"""

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from itertools import cycle
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# FLOPs counter
# ---------------------------------------------------------------------------

def get_flops(model) -> int:
    """Return total floating-point operations for a single forward pass."""
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    concrete = tf.function(lambda x: model(x)).get_concrete_function(
        tf.TensorSpec([1] + list(model.input.shape[1:]), model.input.dtype)
    )
    frozen = convert_variables_to_constants_v2(concrete)
    graph_def = frozen.graph.as_graph_def()

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops_obj = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=run_meta, cmd="op", options=opts
        )
    return flops_obj.total_float_ops


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(model, x, y_onehot, batch_size=32):
    """
    Returns dict with accuracy, loss, precision, recall, f1, inference_ms_per_sample.
    """
    loss, acc = model.evaluate(x, y_onehot, batch_size=batch_size, verbose=0)[:2]

    start = time.time()
    y_probs = model.predict(x, batch_size=batch_size, verbose=0)
    elapsed = (time.time() - start) / len(x) * 1000

    y_pred  = np.argmax(y_probs, axis=1)
    y_true  = np.argmax(y_onehot, axis=1)

    return {
        "accuracy":   acc * 100,
        "loss":       loss,
        "precision":  precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "recall":     recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
        "f1":         f1_score(y_true, y_pred, average="macro", zero_division=0),
        "inference_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_curves(hist, save_path=None):
    """Plot accuracy, loss, and learning rate over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(hist.history["accuracy"],     label="Train")
    axes[0].plot(hist.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(hist.history["loss"],     label="Train")
    axes[1].plot(hist.history["val_loss"], label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    if "lr" in hist.history:
        axes[2].plot(hist.history["lr"])
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fmt = ".2f" if normalize else "d"

    plt.figure(figsize=(16, 14), dpi=300)
    sns.heatmap(
        cm, cmap="Blues",
        xticklabels=class_names if class_names is not None else False,
        yticklabels=class_names if class_names is not None else False,
        annot=(cm.shape[0] <= 20),
        fmt=fmt if cm.shape[0] <= 20 else "",
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_false_pos_neg(y_true_onehot, y_pred_onehot, n_classes, save_path=None):
    cm = confusion_matrix(
        np.argmax(y_true_onehot, axis=-1),
        np.argmax(y_pred_onehot, axis=-1),
    )
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)

    plt.figure(figsize=(12, 5), dpi=300)
    plt.bar(np.arange(n_classes), fp, alpha=0.6, label="False Positives",  color="orange")
    plt.bar(np.arange(n_classes), fn, alpha=0.6, label="False Negatives", color="purple")
    plt.xlabel("Class Label", fontsize=13)
    plt.ylabel("Count",       fontsize=13)
    plt.title("False Positives & False Negatives per Class")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_classwise_accuracy(y_true, y_pred, num_classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    plt.figure(figsize=(20, 5))
    plt.bar(range(num_classes), per_class_acc)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_precision_recall_curve(y_true_onehot, y_probs, num_classes, save_path=None):
    colors = cycle(["blue", "red", "green", "purple", "orange", "brown"])
    plt.figure(figsize=(12, 8))
    for i, color in zip(range(num_classes), colors):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
        plt.plot(recall, precision, color=color, lw=1.5, label=f"Class {i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    if num_classes <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads       = tape.gradient(class_channel, conv_out)
    pooled_grad = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap     = conv_out[0] @ pooled_grad[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def plot_gradcam_grid(samples, labels_onehot, model, last_conv_layer, n=9, save_path=None):
    """Overlay Grad-CAM heatmaps on the first n validation samples."""
    plt.figure(figsize=(12, 12))
    for i in range(min(n, len(samples))):
        img = samples[i : i + 1]
        preds = model.predict(img, verbose=0)
        pred_class = np.argmax(preds[0])
        true_class = np.argmax(labels_onehot[i])

        heatmap = make_gradcam_heatmap(img, model, last_conv_layer)
        h, w = samples[i].shape[:2]
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        original = samples[i]
        if original.max() <= 1.0:
            original = original * 255.0
        original = np.repeat(original.astype(np.uint8), 3, axis=-1)
        superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        plt.subplot(3, 3, i + 1)
        plt.imshow(superimposed)
        plt.title(f"T:{true_class} P:{pred_class}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
