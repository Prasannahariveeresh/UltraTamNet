"""
Baseline model builders — all models compared against UltraTamNet in Table 3.

All models accept a single-channel (grayscale) input of shape (64, 64, 1) and
are trained from scratch (weights=None) as reported in the paper.

Model          | Category     | Params (M) | FLOPs (M)
---------------|--------------|------------|----------
ResNet50       | CNN          | 23.9       | 620
DenseNet169    | CNN          | 12.0       | 540
DenseNet121    | CNN          |  7.2       | 453
EfficientNetB0 | Lightweight  |  4.24      |  65
EfficientNetB5 | Lightweight  | 28.8       | 402
LeNet-5        | Baseline     |  2.48      |   0.83
MobileNetV2    | Lightweight  |  2.45      |  49
MobileNetV3Small| Lightweight |  1.03      |  10
MobileNetV3Large| Lightweight |  3.14      |  39
NASNetMobile   | Lightweight  |  4.43      |  93
Xception       | CNN          | 21.18      | 1105

Usage:
    from models.baselines import build_model
    model = build_model("ResNet50", num_classes=156)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
"""

from tensorflow.keras.applications import (
    ResNet50,
    DenseNet121,
    DenseNet169,
    EfficientNetB0,
    EfficientNetB5,
    MobileNetV2,
    MobileNetV3Small,
    MobileNetV3Large,
    NASNetMobile,
    Xception,
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, MaxPooling2D, Conv2D, Resizing
from tensorflow.keras.models import Model, Sequential


INPUT_SHAPE = (64, 64, 1)


def _transfer_head(base_model, num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=output)


def build_resnet50(num_classes=156):
    base = ResNet50(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    return _transfer_head(base, num_classes)


def build_densenet121(num_classes=156):
    base = DenseNet121(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    return _transfer_head(base, num_classes)


def build_densenet169(num_classes=156):
    base = DenseNet169(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    return _transfer_head(base, num_classes)


def build_efficientnetb0(num_classes=156):
    base = EfficientNetB0(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    return _transfer_head(base, num_classes)


def build_efficientnetb5(num_classes=156):
    base = EfficientNetB5(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    return _transfer_head(base, num_classes)


def build_mobilenetv2(num_classes=156):
    base = MobileNetV2(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    return _transfer_head(base, num_classes)


def build_mobilenetv3small(num_classes=156):
    base = MobileNetV3Small(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    return _transfer_head(base, num_classes)


def build_mobilenetv3large(num_classes=156):
    base = MobileNetV3Large(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    return _transfer_head(base, num_classes)


def build_nasnetmobile(num_classes=156):
    base = NASNetMobile(include_top=False, weights=None, input_shape=INPUT_SHAPE)
    return _transfer_head(base, num_classes)


def build_xception(num_classes=156):
    # Xception requires ≥71×71; upscale internally so external interface stays 64×64.
    inputs = Input(shape=INPUT_SHAPE)
    x = Resizing(75, 75)(inputs)
    base = Xception(include_top=False, weights=None, input_shape=(75, 75, 1))
    x = base(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


def build_lenet5(num_classes=156):
    """Classic LeNet-5 adapted for 64×64 grayscale input and 156 classes."""
    inputs = Input(shape=INPUT_SHAPE)
    x = Conv2D(32, (5, 5), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(strides=2)(x)
    x = Conv2D(48, (5, 5), padding="valid", activation="relu")(x)
    x = MaxPooling2D(strides=2)(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(84,  activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs, name="LeNet-5")


_REGISTRY = {
    "ResNet50":        build_resnet50,
    "DenseNet121":     build_densenet121,
    "DenseNet169":     build_densenet169,
    "EfficientNetB0":  build_efficientnetb0,
    "EfficientNetB5":  build_efficientnetb5,
    "MobileNetV2":     build_mobilenetv2,
    "MobileNetV3Small": build_mobilenetv3small,
    "MobileNetV3Large": build_mobilenetv3large,
    "NASNetMobile":    build_nasnetmobile,
    "Xception":        build_xception,
    "LeNet-5":         build_lenet5,
}


def build_model(name: str, num_classes: int = 156):
    """
    Build a baseline model by name.

    Args:
        name:        One of the keys in the table above.
        num_classes: 156 for uTHCD, 12 for custom dataset.

    Returns:
        Uncompiled Keras Model.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](num_classes=num_classes)


if __name__ == "__main__":
    for name, builder in _REGISTRY.items():
        m = builder()
        print(f"{name:20s}  params={m.count_params() / 1e6:.2f} M")
