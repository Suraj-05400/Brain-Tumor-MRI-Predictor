"""Dataset utilities: scan + clean + split the image folders and build tf.data pipelines.

Expected layout (the PARENT FOLDER of every image is its class name):
    <data_dir>/Training/<class>/*.jpg   and   <data_dir>/Testing/<class>/*.jpg
    or simply <data_dir>/<class>/*.jpg
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_ALIASES = {"training": "train", "train": "train", "testing": "test", "test": "test",
                 "valid": "val", "validation": "val", "val": "val"}


def scan_dataset(data_dir):
    """List all images, drop unreadable files and exact duplicates (prevents train/test leakage)."""
    rows = []
    for p in sorted(Path(data_dir).rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        parts = p.relative_to(data_dir).parts
        split = SPLIT_ALIASES.get(parts[0].lower(), "all") if len(parts) > 2 else "all"
        try:
            with Image.open(p) as im:
                im.verify()                                   # raises on corrupt files
            rows.append({"path": str(p), "label": p.parent.name, "orig_split": split,
                         "md5": hashlib.md5(p.read_bytes()).hexdigest()})
        except Exception:
            print("Skipping unreadable file:", p)
    if not rows:
        raise FileNotFoundError(f"No images found under '{data_dir}'. Check the dataset path/layout.")
    df = pd.DataFrame(rows)
    n = len(df)
    df = df.drop_duplicates("md5").reset_index(drop=True)
    print(f"Found {n} readable images; removed {n - len(df)} exact duplicates -> {len(df)} images")
    return df


def split_dataset(df, seed=42):
    """Return (train, val, test, class_names). Uses the dataset's own Testing folder when present,
    otherwise a stratified 70/15/15 split. Validation is always a stratified slice of the rest."""
    class_names = sorted(df["label"].unique())
    df = df.assign(label_id=df["label"].map({c: i for i, c in enumerate(class_names)}))
    if (df["orig_split"] == "test").any() and (df["orig_split"] != "test").any():
        test = df[df["orig_split"] == "test"]
        rest = df[df["orig_split"] != "test"]
        train, val = train_test_split(rest, test_size=0.15, stratify=rest["label_id"], random_state=seed)
    else:
        train, temp = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=seed)
        val, test = train_test_split(temp, test_size=0.50, stratify=temp["label_id"], random_state=seed)
    return [d.reset_index(drop=True) for d in (train, val, test)] + [class_names]


def _load(path, label, img_size):
    """Decode + resize to img_size x img_size. Pixels stay in 0-255; each model scales them itself."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, (img_size, img_size)), label


def make_ds(df, img_size, batch, training=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label_id"].values.astype("int32")))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: _load(p, l, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def get_class_weights(train_df, n_classes):
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=train_df["label_id"].values)
    return {i: float(v) for i, v in enumerate(w)}
