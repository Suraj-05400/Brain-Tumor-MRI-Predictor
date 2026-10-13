"""Model zoo: custom CNN + transfer-learning models (ResNet50V2, MobileNetV2, InceptionV3, EfficientNetB0).

Every model takes RAW 0-255 RGB images and does its own scaling inside the graph, so the saved
.h5 file can be used in the Streamlit app without any extra preprocessing code.
  * custom CNN      -> pixel / 255   (0-1 range, as in the project brief)
  * MobileNetV2, ResNet50V2, InceptionV3 -> [-1, 1] (the range their ImageNet weights expect)
  * EfficientNetB0  -> raw 0-255 (normalises internally)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

PRETRAINED_KINDS = ["mobilenetv2", "resnet50v2", "inceptionv3", "efficientnetb0"]
ALL_KINDS = ["custom_cnn"] + PRETRAINED_KINDS


@keras.utils.register_keras_serializable(package="brain_mri")
class RandomBrightnessShift(layers.Layer):
    """Random brightness jitter (+-factor of the 0-255 range). Active only in training.
    Custom layer because the built-in RandomBrightness cannot be written to legacy .h5 files.
    Import this module (src.models) before loading a saved model that contains it."""

    def __init__(self, factor=0.15, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x, training=None):
        if not training:
            return x
        delta = tf.random.uniform([tf.shape(x)[0], 1, 1, 1], -self.factor * 255.0, self.factor * 255.0)
        return tf.clip_by_value(x + delta, 0.0, 255.0)

    def get_config(self):
        return {**super().get_config(), "factor": self.factor}


def augmentation_block():
    """Rotation, horizontal+vertical flip, zoom, brightness and shifts (active only while training)."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.08, 0.08),
        RandomBrightnessShift(0.15),
    ], name="augment")


def _backbone(kind, img_size, pretrained):
    shape, w = (img_size, img_size, 3), ("imagenet" if pretrained else None)
    if kind == "mobilenetv2":
        return keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights=w), layers.Rescaling(1 / 127.5, offset=-1.0)
    if kind == "resnet50v2":
        return keras.applications.ResNet50V2(input_shape=shape, include_top=False, weights=w), layers.Rescaling(1 / 127.5, offset=-1.0)
    if kind == "inceptionv3":
        return keras.applications.InceptionV3(input_shape=shape, include_top=False, weights=w), layers.Rescaling(1 / 127.5, offset=-1.0)
    if kind == "efficientnetb0":
        return keras.applications.EfficientNetB0(input_shape=shape, include_top=False, weights=w), layers.Rescaling(1.0)
    raise ValueError(f"Unknown model '{kind}'. Choose from {ALL_KINDS}")


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def build_model(kind, n_classes, img_size=224, pretrained=True, lr=1e-3, dropout=0.4, units=256):
    """Build + compile a model. Transfer models start with a FROZEN backbone and a new dense head."""
    inp = keras.Input((img_size, img_size, 3), name="image")
    x = augmentation_block()(inp)
    if kind == "custom_cnn":
        x = layers.Rescaling(1 / 255.0)(x)                      # 0-1 normalisation
        for f in (32, 64, 128, 256):                            # 4 blocks of (Conv-BN-ReLU) x2 + pool + dropout
            for _ in range(2):
                x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
            x = layers.MaxPooling2D()(x)
            x = layers.Dropout(0.25)(x)
    else:
        base, scale = _backbone(kind, img_size, pretrained)
        base.trainable = False
        x = base(scale(x), training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out, name=kind)
    compile_model(model, lr)
    return model


def unfreeze_top(model, n_layers=30, lr=1e-5):
    """Fine-tuning stage: unfreeze the top n layers of the backbone (BatchNorm stays frozen)."""
    base = [l for l in model.layers if isinstance(l, keras.Model) and l.name != "augment"][-1]
    base.trainable = True
    for l in base.layers[:-n_layers]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    compile_model(model, lr)
