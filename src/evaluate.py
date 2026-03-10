import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import load_config
from dataset import UltrasonicDataset
from model import UltrasonicCNN
from train import (
    RemappedSubset,
    _build_recording_split,
    _compute_inverse_frequency_weights,
    _evaluate_confusion,
    _parse_selected_classes,
)


def _print_confusion_matrix(conf_mat, class_names):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(conf_mat)
    print("\nPer-class Accuracy:")
    row_sums = conf_mat.sum(dim=1).cpu().numpy()
    for i, cname in enumerate(class_names):
        acc = float(conf_mat[i, i].item() / row_sums[i]) if row_sums[i] > 0 else 0.0
        print(f"{cname:<10}: {acc:.4f}")


def _plot_history(history_path, results_dir):
    if not history_path.exists():
        print(f"Training history not found: {history_path}")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        h = json.load(f)

    train_loss = h.get("train_loss", [])
    val_loss = h.get("val_loss", [])
    val_acc = h.get("val_accuracy", [])
    val_f1 = h.get("val_macro_f1", [])
    epochs = np.arange(1, len(train_loss) + 1)

    if len(epochs) == 0:
        print("Training history is empty; skipping loss/accuracy plots.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = results_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path, dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_acc, label="Val Accuracy (%)")
    plt.plot(epochs, val_f1, label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Accuracy / Macro-F1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    acc_path = results_dir / "accuracy_curve.png"
    plt.tight_layout()
    plt.savefig(acc_path, dpi=140)
    plt.close()

    print(f"Saved plot: {loss_path}")
    print(f"Saved plot: {acc_path}")


def _plot_confusion(conf_mat, class_names, out_path):
    cm = conf_mat.cpu().numpy().astype(np.float64)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def evaluate(classes=None, batch_size=None, quick=False, val_ratio=0.2, seed=42):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dataset = UltrasonicDataset(
        config["paths"]["raw_dir"],
        transform=False,
        preload=(device.type == "cpu"),
    )
    all_class_names = base_dataset.classes
    selected_label_indices = _parse_selected_classes(classes, all_class_names)
    selected_class_names = [all_class_names[i] for i in selected_label_indices]
    label_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_label_indices)}

    train_indices, val_indices, _, _ = _build_recording_split(
        base_dataset.samples,
        selected_label_indices,
        val_ratio=val_ratio,
        seed=seed,
        quick=quick,
    )

    val_ds = RemappedSubset(base_dataset, val_indices, label_map)
    eff_bs = int(batch_size if batch_size is not None else config["training"].get("batch_size", 8))
    val_loader = DataLoader(
        val_ds,
        batch_size=eff_bs,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = len(selected_class_names)
    model = UltrasonicCNN(num_classes=num_classes).to(device)
    model_path = config["paths"]["model_output"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    weights_cpu, _ = _compute_inverse_frequency_weights(
        base_dataset.samples,
        train_indices,
        label_map,
        num_classes,
    )
    criterion = nn.CrossEntropyLoss(weight=weights_cpu.to(device), label_smoothing=0.0)
    val_loss, val_acc, val_macro_f1, conf_mat = _evaluate_confusion(
        model, val_loader, criterion, num_classes, device
    )

    print("\nValidation Metrics:")
    print(f"Val Loss     : {val_loss:.4f}")
    print(f"Val Accuracy : {val_acc:.2f}%")
    print(f"Val Macro-F1 : {val_macro_f1:.4f}")
    _print_confusion_matrix(conf_mat, selected_class_names)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _plot_history(results_dir / "training_history.json", results_dir)
    _plot_confusion(conf_mat, selected_class_names, results_dir / "confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class list.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--quick", action="store_true", help="Use quick-mode recording split.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    args = parser.parse_args()
    evaluate(
        classes=args.classes,
        batch_size=args.batch_size,
        quick=args.quick,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )