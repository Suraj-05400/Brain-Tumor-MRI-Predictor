"""Evaluation helpers: metrics, confusion matrix, training curves and efficiency numbers."""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)


def compute_metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def predict(model, ds):
    """Return predicted class ids and the average inference time per image (ms)."""
    t0 = time.perf_counter()
    prob = model.predict(ds, verbose=0)
    ms = (time.perf_counter() - t0) / len(prob) * 1000
    return prob.argmax(1), ms


def save_report(name, y_true, y_pred, class_names, out_dir):
    """Write the classification report (txt) and the confusion-matrix figure (png)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{name}_classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{name} - confusion matrix (test)"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(f"{out_dir}/{name}_confusion_matrix.png", dpi=120); plt.close()


def plot_history(name, hist, out_dir):
    """Training vs validation accuracy and loss."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["accuracy"], label="train"); ax[0].plot(hist["val_accuracy"], label="val"); ax[0].set_title(f"{name} accuracy")
    ax[1].plot(hist["loss"], label="train"); ax[1].plot(hist["val_loss"], label="val"); ax[1].set_title(f"{name} loss")
    for a in ax:
        a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(f"{out_dir}/{name}_history.png", dpi=120); plt.close()


def plot_comparison(df, out_dir):
    cols = ["accuracy", "precision", "recall", "f1"]
    ax = df.set_index("model")[cols].plot(kind="bar", figsize=(10, 4.5), rot=0)
    ax.set_ylim(0, 1.05); ax.set_title("Model comparison (test set)")
    plt.tight_layout(); plt.savefig(f"{out_dir}/model_comparison.png", dpi=120); plt.close()
