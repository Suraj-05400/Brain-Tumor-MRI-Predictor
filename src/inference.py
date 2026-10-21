"""Inference helpers shared by the Streamlit app and the analysis scripts: preprocessing, prediction, Grad-CAM."""
import numpy as np
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image
from tensorflow.keras import layers


def preprocess(img, size):
    """PIL image -> float32 array (size, size, 3) in 0-255. Scaling to the model's range happens INSIDE the model."""
    return np.asarray(img.convert("RGB").resize((size, size)), dtype="float32")


def predict(model, x):
    """Class probabilities for one preprocessed image.
    Calling the model directly is much faster than model.predict() for a single image
    (predict builds a data pipeline and a progress loop on every call)."""
    return np.asarray(model(x[None, ...], training=False))[0]


def grad_cam(model, x, class_idx=None):
    """Grad-CAM heat map in [0, 1] (same H x W as x) for the predicted (or a given) class.
    Works for every model of this project: it takes the feature map right before global average pooling."""
    t = tf.convert_to_tensor(x[None, ...], tf.float32)
    with tf.GradientTape() as tape:
        h, conv = t, None
        for layer in model.layers[1:]:                       # skip the InputLayer
            if isinstance(layer, layers.GlobalAveragePooling2D):
                conv = h
                tape.watch(conv)
            h = layer(h, training=False)
        idx = int(tf.argmax(h[0])) if class_idx is None else int(class_idx)
        score = h[:, idx]
    if conv is None:
        raise ValueError("Model has no GlobalAveragePooling2D layer; cannot compute Grad-CAM.")
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv, axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., None], x.shape[:2]).numpy()[..., 0]
    return cam, idx, h[0].numpy()


def overlay(x, cam, alpha=0.45):
    """Blend the heat map over the image; returns a PIL image."""
    heat = (colormaps["jet"](cam)[..., :3] * 255).astype("float32")
    mix = x.astype("float32") * (1 - alpha) + heat * alpha
    return Image.fromarray(np.clip(mix, 0, 255).astype("uint8"))
