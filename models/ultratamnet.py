"""
UltraTamNet — Tamil-optimized lightweight handwritten character recognition model.

Architecture overview:
  - Initial 5×5 Conv + BN + ReLU  → MaxPool
  - Three SeparableConv + BN + ReLU blocks → MaxPool after each
  - Four residual blocks (pairs at 256 and 512 filters)
  - GlobalAveragePooling → Dense (softmax)

Key design choices:
  - SeparableConv2D throughout (depthwise + pointwise) keeps FLOPs low (167 MFLOPs)
  - Residual skip connections preserve gradient flow for deeper Tamil character features
  - ~1.27 M parameters — smaller than MobileNetV2 (2.45 M)

Used in:
  - Table 3 (uTHCD benchmark, 156 classes)  → build_ultratamnet(input_shape=(64,64,1), num_classes=156)
  - Table 6 (custom dataset, 12 classes)    → build_ultratamnet(input_shape=(64,64,1), num_classes=12)
  - Ablation study                          → build_ultratamnet_variant(...)
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, SeparableConv2D, Add,
    BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D,
)
from tensorflow.keras.models import Model


def _conv_bn(x, filters, kernel_size, strides=1):
    x = Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def _sep_bn(x, filters, kernel_size, strides=1):
    x = SeparableConv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def _residual_block(x, filters):
    shortcut = x
    x = _sep_bn(x, filters, kernel_size=3)
    x = Add()([x, shortcut])
    return x


def build_ultratamnet(input_shape=(64, 64, 1), num_classes=156):
    """
    Full UltraTamNet as described in the paper.

    Args:
        input_shape:  (H, W, C) — default (64, 64, 1) for grayscale 64×64 images
        num_classes:  156 for uTHCD, 12 for custom dataset

    Returns:
        Uncompiled Keras Model
    """
    inputs = Input(shape=input_shape)

    x = _conv_bn(inputs, filters=32, kernel_size=5)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    x = _sep_bn(x, filters=64,  kernel_size=3)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    x = _sep_bn(x, filters=128, kernel_size=3)
    x = MaxPool2D(pool_size=2, strides=2)(x)

    x = _sep_bn(x, 256, kernel_size=3)
    x = _residual_block(x, 256)

    x = _sep_bn(x, 256, kernel_size=3)
    x = _residual_block(x, 256)

    x = _sep_bn(x, 512, kernel_size=3)
    x = _residual_block(x, 512)

    x = _sep_bn(x, 512, kernel_size=3)
    x = _residual_block(x, 512)

    x = GlobalAvgPool2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name="UltraTamNet")


# ---------------------------------------------------------------------------
# Ablation variants (used in ablation_study.py)
# ---------------------------------------------------------------------------

def build_ultratamnet_variant(
    input_shape=(64, 64, 1),
    num_classes=156,
    residual=True,
    separable=True,
    depth=4,
):
    """
    Configurable UltraTamNet variant for the ablation study.

    Ablation configs from the paper:
        A1: residual=False, separable=False, depth=2  — plain CNN baseline
        A2: residual=True,  separable=False, depth=2  — residuals only
        A3: residual=False, separable=True,  depth=2  — separable convs only
        A4: residual=True,  separable=True,  depth=2  — both, shallow
        A5: residual=True,  separable=True,  depth=4  — full UltraTamNet

    Args:
        residual:   Whether to include residual skip connections
        separable:  Whether to use SeparableConv2D (True) or plain Conv2D (False)
        depth:      Number of residual/conv block pairs after the initial stem

    Returns:
        Uncompiled Keras Model
    """
    conv_fn = _sep_bn if separable else _conv_bn

    inputs = Input(shape=input_shape)

    x = _conv_bn(inputs, 32, 5)
    x = MaxPool2D(2)(x)

    x = conv_fn(x, 64, 3)
    x = MaxPool2D(2)(x)

    x = conv_fn(x, 128, 3)
    x = MaxPool2D(2)(x)

    for _ in range(depth):
        x = conv_fn(x, 256, 3)
        if residual:
            x = _residual_block(x, 256)

    x = conv_fn(x, 512, 3)
    if residual:
        x = _residual_block(x, 512)

    x = GlobalAvgPool2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    variant_name = (
        f"UltraTamNet_d{depth}"
        f"_{'res' if residual else 'nores'}"
        f"_{'sep' if separable else 'std'}"
    )
    return Model(inputs, outputs, name=variant_name)


if __name__ == "__main__":
    model = build_ultratamnet(input_shape=(64, 64, 1), num_classes=156)
    model.summary()
    print(f"\nTotal parameters: {model.count_params() / 1e6:.2f} M")
