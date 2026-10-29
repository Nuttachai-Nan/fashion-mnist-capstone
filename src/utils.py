"""Shared metrics and plotting helpers."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .data import CLASS_NAMES, MEAN, STD


def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": np.asarray(y_true, dtype=int),
        "y_pred": np.asarray(y_pred, dtype=int),
        "y_prob": np.asarray(y_prob, dtype=float),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(10))),
    }


def write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_learning_curves(histories: dict[str, list[dict]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, rows in histories.items():
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["train_loss"] for r in rows], marker="o", label=f"{name} train")
        axes[0].plot(epochs, [r["val_loss"] for r in rows], marker="o", linestyle="--", label=f"{name} val")
        axes[1].plot(epochs, [r["train_accuracy"] for r in rows], marker="o", label=f"{name} train")
        axes[1].plot(epochs, [r["val_accuracy"] for r in rows], marker="o", linestyle="--", label=f"{name} val")
    axes[0].set_title("Cross-entropy loss")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Fraction correct")
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(CLASS_NAMES)), yticks=np.arange(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           ylabel="True label", xlabel="Predicted label", title="CNN test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = cm.max() * 0.55 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_failure_cases(test_set, result: dict, output: Path, json_output: Path, limit: int = 12) -> list[dict]:
    wrong = np.flatnonzero(result["y_true"] != result["y_pred"])
    if len(wrong) == 0:
        write_json(json_output, [])
        return []
    confidence = result["y_prob"][wrong, result["y_pred"][wrong]]
    # Pick high-confidence errors: concrete examples that are often instructive to inspect.
    chosen = wrong[np.argsort(confidence)[::-1][:limit]]
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    rows = []
    for ax, idx in zip(axes.flat, chosen):
        image, true_label = test_set[int(idx)]
        pred_label = int(result["y_pred"][idx])
        pred_conf = float(result["y_prob"][idx, pred_label])
        # Undo normalisation for display.
        display = image.squeeze(0).cpu().numpy() * STD[0] + MEAN[0]
        ax.imshow(display, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"True: {CLASS_NAMES[int(true_label)]}\nPred: {CLASS_NAMES[pred_label]} ({pred_conf:.2f})", fontsize=8)
        ax.axis("off")
        rows.append({"test_index": int(idx), "true_label": int(true_label),
                     "true_name": CLASS_NAMES[int(true_label)], "predicted_label": pred_label,
                     "predicted_name": CLASS_NAMES[pred_label], "predicted_confidence": pred_conf})
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    fig.suptitle("High-confidence CNN mistakes (inspect; do not overgeneralise)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    write_json(json_output, rows)
    return rows


def plot_success_cases(test_set, result: dict, output: Path, json_output: Path, limit: int = 12) -> list[dict]:
    """Save concrete correct predictions for the demo and qualitative review."""
    correct = np.flatnonzero(result["y_true"] == result["y_pred"])
    if len(correct) == 0:
        write_json(json_output, [])
        return []
    confidence = result["y_prob"][correct, result["y_pred"][correct]]
    chosen = correct[np.argsort(confidence)[::-1][:limit]]
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    rows = []
    for ax, idx in zip(axes.flat, chosen):
        image, true_label = test_set[int(idx)]
        pred_label = int(result["y_pred"][idx])
        pred_conf = float(result["y_prob"][idx, pred_label])
        display = image.squeeze(0).cpu().numpy() * STD[0] + MEAN[0]
        ax.imshow(display, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"True: {CLASS_NAMES[int(true_label)]}\nPred: {CLASS_NAMES[pred_label]} ({pred_conf:.2f})", fontsize=8)
        ax.axis("off")
        rows.append({"test_index": int(idx), "true_label": int(true_label),
                     "true_name": CLASS_NAMES[int(true_label)], "predicted_label": pred_label,
                     "predicted_name": CLASS_NAMES[pred_label], "predicted_confidence": pred_conf})
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    fig.suptitle("Correct CNN predictions (high confidence; illustrative examples)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    write_json(json_output, rows)
    return rows
