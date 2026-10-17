"""Random-search hyper-parameter tuning + stratified K-fold cross-validation for ONE model, in its own process
(keeps memory low when several models are processed one after another).

    python -m src.tune_cv --data_dir /content/data --kind resnet50v2 --img_size 224
Writes results/tune_cv_<kind>.json  (same train/validation split as src.train: same function, same seed)
"""
import argparse
import gc
import itertools
import json
import os
import random

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

from src.data import get_class_weights, make_ds, scan_dataset, split_dataset
from src.models import ALL_KINDS, build_model

DEFAULT_CFG = dict(lr=1e-3, dropout=0.4, units=256)     # configuration used for stage 1 of the saved models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--kind", required=True, choices=ALL_KINDS)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--trials", type=int, default=3, help="random-search trials (the default config is always trial 0)")
    ap.add_argument("--tune_epochs", type=int, default=4)
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--cv_epochs", type=int, default=3)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results")
    a = ap.parse_args()
    tf.keras.utils.set_random_seed(a.seed)
    pretrained = not a.no_pretrained

    train_df, val_df, _, names = split_dataset(scan_dataset(a.data_dir), a.seed)
    n = len(names)
    cw = get_class_weights(train_df, n)
    train_ds = make_ds(train_df, a.img_size, a.batch, training=True, seed=a.seed)
    val_ds = make_ds(val_df, a.img_size, a.batch)

    # ---- random search at an equal small budget (default config = trial 0) ----
    space = list(itertools.product([1e-3, 5e-4, 1e-4], [0.2, 0.4, 0.5], [128, 256, 512]))
    random.Random(a.seed).shuffle(space)
    cfgs = [("default", DEFAULT_CFG)] + [("random", dict(lr=lr, dropout=do, units=un)) for lr, do, un in space
                                          if (lr, do, un) != (1e-3, 0.4, 256)][:a.trials]
    rows = []
    for tag, cfg in cfgs:
        m = build_model(a.kind, n, a.img_size, pretrained, **cfg)
        h = m.fit(train_ds, validation_data=val_ds, epochs=a.tune_epochs, class_weight=cw, verbose=0)
        rows.append({"config": tag, **cfg, "best_val_accuracy": float(max(h.history["val_accuracy"])),
                     "best_val_loss": float(min(h.history["val_loss"]))})
        print(f"  trial {tag:7s} {cfg} -> best val accuracy {rows[-1]['best_val_accuracy']:.4f}", flush=True)
        del m
        keras.backend.clear_session(); gc.collect()

    # ---- stratified K-fold cross-validation of the default configuration (training set only) ----
    skf = StratifiedKFold(a.cv_folds, shuffle=True, random_state=a.seed)
    accs, f1s = [], []
    for i, (tr, te) in enumerate(skf.split(train_df, train_df["label_id"]), 1):
        m = build_model(a.kind, n, a.img_size, pretrained, **DEFAULT_CFG)
        m.fit(make_ds(train_df.iloc[tr], a.img_size, a.batch, training=True, seed=a.seed), epochs=a.cv_epochs, class_weight=cw, verbose=0)
        te_df = train_df.iloc[te]
        pred = m.predict(make_ds(te_df, a.img_size, a.batch), verbose=0).argmax(1)
        accs.append(accuracy_score(te_df["label_id"], pred))
        f1s.append(f1_score(te_df["label_id"], pred, average="macro", zero_division=0))
        print(f"  fold {i}: accuracy {accs[-1]:.4f} | macro-F1 {f1s[-1]:.4f}", flush=True)
        del m
        keras.backend.clear_session(); gc.collect()
    cv = {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)), "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
    print(f"{a.cv_folds}-fold CV -> accuracy {cv['acc_mean']:.4f} ± {cv['acc_std']:.4f} | macro-F1 {cv['f1_mean']:.4f} ± {cv['f1_std']:.4f}", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"kind": a.kind, "tuning": rows, "cv": cv, "folds": {"accuracy": accs, "f1": f1s},
               "settings": {"trials": a.trials, "tune_epochs": a.tune_epochs, "cv_folds": a.cv_folds, "cv_epochs": a.cv_epochs}},
              open(f"{a.out_dir}/tune_cv_{a.kind}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
