"""Train + evaluate + compare the models, then export the best one for the Streamlit app.

Example (Colab, GPU):
    python -m src.train --data_dir /content/data --epochs 25 --ft_epochs 10
Smoke test (CPU):
    python -m src.train --data_dir data --img_size 64 --epochs 1 --ft_epochs 1 --no_pretrained
"""
import argparse
import json
import os
import shutil

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data import get_class_weights, make_ds, scan_dataset, split_dataset
from src.evaluate import compute_metrics, plot_comparison, plot_history, predict, save_report
from src.models import ALL_KINDS, build_model, unfreeze_top


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder whose sub-folders are the classes")
    ap.add_argument("--models", default="custom_cnn,mobilenetv2,resnet50v2,efficientnetb0",
                    help=f"comma separated, from {ALL_KINDS}")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25, help="stage-1 epochs (EarlyStopping applies)")
    ap.add_argument("--ft_epochs", type=int, default=10, help="fine-tuning epochs for transfer models")
    ap.add_argument("--no_pretrained", action="store_true", help="skip ImageNet weights (smoke tests only)")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def make_callbacks(path, init_best=None):
    """EarlyStopping + ModelCheckpoint(best val_loss -> .h5) + LR reduction."""
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True,
                                            initial_value_threshold=init_best),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7)]


def train_one(kind, n_classes, args, train_ds, val_ds, class_weight, ckpt):
    """Stage 1: train (head only for transfer models). Stage 2: fine-tune the top backbone layers."""
    model = build_model(kind, n_classes, args.img_size, pretrained=not args.no_pretrained)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, class_weight=class_weight,
                   callbacks=make_callbacks(ckpt), verbose=2)
    hist = {k: list(v) for k, v in h1.history.items()}
    if kind != "custom_cnn" and args.ft_epochs > 0:
        unfreeze_top(model, 30, lr=1e-5)
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.ft_epochs, class_weight=class_weight,
                       callbacks=make_callbacks(ckpt, init_best=min(h1.history["val_loss"])), verbose=2)
        for k, v in h2.history.items():
            hist[k] += list(v)
    return hist


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    models_dir, res_dir = f"{args.out_dir}/models", f"{args.out_dir}/results"
    os.makedirs(models_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)

    # ---- data ----
    df = scan_dataset(args.data_dir)
    train_df, val_df, test_df, class_names = split_dataset(df, args.seed)
    n_classes = len(class_names)
    print("Classes:", class_names)
    print(pd.DataFrame({"train": train_df.label.value_counts(), "val": val_df.label.value_counts(),
                        "test": test_df.label.value_counts()}).reindex(class_names))
    json.dump(class_names, open(f"{models_dir}/class_names.json", "w"))
    train_ds = make_ds(train_df, args.img_size, args.batch, training=True, seed=args.seed)
    val_ds, test_ds = make_ds(val_df, args.img_size, args.batch), make_ds(test_df, args.img_size, args.batch)
    cw = get_class_weights(train_df, n_classes)

    # ---- train + evaluate every model ----
    rows = []
    for kind in [k.strip() for k in args.models.split(",") if k.strip()]:
        print(f"\n{'=' * 20} {kind} {'=' * 20}")
        ckpt = f"{models_dir}/{kind}.h5"
        hist = train_one(kind, n_classes, args, train_ds, val_ds, cw, ckpt)
        json.dump(hist, open(f"{res_dir}/{kind}_history.json", "w"))
        plot_history(kind, hist, res_dir)

        best = keras.models.load_model(ckpt, compile=False)          # best checkpoint on disk
        val_pred, _ = predict(best, val_ds)
        test_pred, ms = predict(best, test_ds)
        save_report(kind, test_df["label_id"].values, test_pred, class_names, res_dir)
        m = compute_metrics(test_df["label_id"].values, test_pred)
        rows.append({"model": kind, **m, "val_f1": compute_metrics(val_df["label_id"].values, val_pred)["f1"],
                     "params_M": best.count_params() / 1e6, "size_MB": os.path.getsize(ckpt) / 1e6, "ms_per_image": ms})
        print({k: round(float(v), 4) for k, v in rows[-1].items() if k != "model"})
        keras.backend.clear_session()

    # ---- compare + export best model (selected on VALIDATION macro-F1) ----
    comp = pd.DataFrame(rows).round(4)
    comp.to_csv(f"{res_dir}/model_comparison.csv", index=False)
    plot_comparison(comp, res_dir)
    best_name = comp.loc[comp["val_f1"].idxmax(), "model"]
    shutil.copy(f"{models_dir}/{best_name}.h5", f"{models_dir}/best_model.h5")
    print("\n", comp.to_string(index=False))
    print(f"\nBest model (validation macro-F1): {best_name} -> {models_dir}/best_model.h5")


if __name__ == "__main__":
    main()
