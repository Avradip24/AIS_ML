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
    _compute_inverse_frequency_weights,
    _evaluate_confusion,
    _parse_selected_classes,
)

def _print_confusion_matrix(conf_mat, class_names):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = " " * 12 + " ".join([f"{c[:10]:>10}" for c in class_names])
    print(header)
    for i, class_name in enumerate(class_names):
        row_vals = " ".join([f"{int(conf_mat[i, j]):>10}" for j in range(len(class_names))])
        print(f"{class_name[:10]:>10}  {row_vals}")

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
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "loss_curve.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_acc, label="Val Accuracy (%)")
    plt.plot(epochs, val_f1, label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Accuracy / Macro-F1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(results_dir / "accuracy_curve.png", dpi=140)
    plt.close()


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
        title="Confusion Matrix (Evaluation Mode)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate(classes=None, batch_size=None, dropout=0.4):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_dir = Path(__file__).resolve().parents[1] / "results"
    
    # --- 1. Load the EXACT split used in training ---
    indices_path = results_dir / "split_indices.pth"
    if not indices_path.exists():
        print(f"Error: {indices_path} not found. You must run train.py first to save the split.")
        return
    
    print(f"Loading split from: {indices_path}")
    saved_data = torch.load(indices_path, map_location="cpu")
    train_indices = saved_data['train_indices']
    val_indices = saved_data['val_indices']
    label_map = saved_data['label_map']
    selected_label_indices = saved_data['selected_label_indices']

    # --- 2. Setup Dataset ---
    base_dataset = UltrasonicDataset(
        config["paths"]["raw_dir"],
        transform=False,
        preload=(device.type == "cpu"),
    )
    all_class_names = base_dataset.classes
    selected_class_names = [all_class_names[i] for i in selected_label_indices]

    val_ds = RemappedSubset(base_dataset, val_indices, label_map)
    eff_bs = int(batch_size if batch_size is not None else config["training"].get("batch_size", 16))
    val_loader = DataLoader(val_ds, batch_size=eff_bs, shuffle=False)

    # --- 3. Load Model with Correct Architecture ---
    num_classes = len(selected_class_names)
    # Dropout MUST match your training config (0.4 based on your JSON)
    model = UltrasonicCNN(num_classes=num_classes, dropout_rate=dropout).to(device)
    
    model_path = config["paths"]["model_output"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 4. Run Evaluation ---
    weights_cpu, _ = _compute_inverse_frequency_weights(
        base_dataset.samples, train_indices, label_map, num_classes
    )
    criterion = nn.CrossEntropyLoss(weight=weights_cpu.to(device), label_smoothing=0.1)
    
    val_loss, val_acc, val_macro_f1, conf_mat = _evaluate_confusion(
        model, val_loader, criterion, num_classes, device
    )

    print(f"\n--- FRESH EVALUATION RESULTS ---")
    print(f"Val Loss     : {val_loss:.4f}")
    print(f"Val Accuracy : {val_acc:.2f}%")
    print(f"Val Macro-F1 : {val_macro_f1:.4f}")
    
    _print_confusion_matrix(conf_mat, selected_class_names)

    # --- 5. Save Visuals ---
    _plot_history(results_dir / "training_history.json", results_dir)
    # Change "confusion_matrix.png" to "evaluation_matrix.png"
    _plot_confusion(conf_mat, selected_class_names, results_dir / "evaluation_matrix.png")
    print(f"Plots updated in: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.4) # Defaulted to your last successful run
    args = parser.parse_args()
    
    evaluate(
        classes=args.classes,
        batch_size=args.batch_size,
        dropout=args.dropout
    )