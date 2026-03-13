import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from collections import defaultdict
import random

from dataset import UltrasonicDataset
from model import UltrasonicCNN
from data_loader import load_config

"""
Hierarchical CNN Evaluation Script for FIUS Project

This script evaluates the trained hierarchical CNN models on the validation set.
It loads the 3 trained models and performs end-to-end hierarchical classification.

Usage:
    python src/evaluate_hierarchical.py

Output:
    Prints validation metrics including:
    - Group classifier accuracy
    - Final hierarchical accuracy
    - Final hierarchical macro-F1
    - Per-class accuracy
    - Confusion matrix
"""


class RemappedSubset(Dataset):
    def __init__(self, base_dataset, indices, label_map):
        self.base_dataset = base_dataset
        self.indices = indices
        self.label_map = label_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.indices[idx]]
        mapped = self.label_map[int(y.item())]
        return x, torch.tensor(mapped).long()


def _compute_macro_f1(conf_mat):
    eps = 1e-8
    tp = conf_mat.diag().float()
    fp = conf_mat.sum(dim=0).float() - tp
    fn = conf_mat.sum(dim=1).float() - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return float(f1.mean().item())


def _build_recording_split(samples, selected_labels, val_ratio=0.2, seed=42, quick=False):
    file_to_indices = defaultdict(list)
    file_to_label = {}

    for idx, (file_path, label, _segment_idx) in enumerate(samples):
        if label not in selected_labels:
            continue
        file_to_indices[file_path].append(idx)
        file_to_label[file_path] = label

    class_to_files = defaultdict(list)
    for file_path, label in file_to_label.items():
        class_to_files[label].append(file_path)

    rng = random.Random(seed)
    train_files = set()
    val_files = set()
    train_recordings = defaultdict(int)
    val_recordings = defaultdict(int)

    for label in selected_labels:
        files = sorted(class_to_files.get(label, []))
        rng.shuffle(files)
        if quick:
            files = files[: min(len(files), 3)]

        if not files:
            continue

        if len(files) == 1:
            class_val = []
            class_train = files
        else:
            val_count = max(1, int(round(len(files) * val_ratio)))
            val_count = min(val_count, len(files) - 1)
            class_val = files[:val_count]
            class_train = files[val_count:]

        train_files.update(class_train)
        val_files.update(class_val)
        train_recordings[label] += len(class_train)
        val_recordings[label] += len(class_val)

    train_indices = []
    val_indices = []
    for file_path, indices in file_to_indices.items():
        if file_path in train_files:
            train_indices.extend(indices)
        elif file_path in val_files:
            val_indices.extend(indices)

    if not train_indices:
        raise ValueError("No training segments found after recording-level split.")
    if not val_indices:
        raise ValueError("No validation segments found after recording-level split.")

    return train_indices, val_indices, train_recordings, val_recordings


def _evaluate_confusion(model, data_loader, criterion, num_classes, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for signals, labels in data_loader:
            signals, labels = signals.to(device), labels.to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            _, predicted = logits.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

            labels_cpu = labels.detach().cpu()
            preds_cpu = predicted.detach().cpu()
            linear_idx = labels_cpu * num_classes + preds_cpu
            conf_mat += torch.bincount(linear_idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)

    avg_val_loss = val_loss / max(1, len(data_loader))
    val_acc = 100.0 * val_correct / max(1, val_total)
    val_macro_f1 = _compute_macro_f1(conf_mat)
    return avg_val_loss, val_acc, val_macro_f1, conf_mat


def _print_confusion_matrix(conf_mat, selected_class_names):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = " " * 12 + " ".join([f"{c[:10]:>10}" for c in selected_class_names])
    print(header)
    for i, class_name in enumerate(selected_class_names):
        row_vals = " ".join([f"{int(conf_mat[i, j]):>10}" for j in range(len(selected_class_names))])
        print(f"{class_name[:10]:>10}  {row_vals}")


def _create_group_datasets(base_dataset, train_indices, val_indices):
    """Create datasets for group classifier and fine classifiers."""
    all_class_names = base_dataset.classes

    # Define group mappings
    group0_classes = ["person", "chair", "plant"]
    group1_classes = ["wall", "backpack", "bigtable"]

    group0_indices = [i for i, name in enumerate(all_class_names) if name in group0_classes]
    group1_indices = [i for i, name in enumerate(all_class_names) if name in group1_classes]

    # Group classifier: group0 -> 0, group1 -> 1
    group_label_map = {}
    for idx in group0_indices:
        group_label_map[idx] = 0
    for idx in group1_indices:
        group_label_map[idx] = 1

    # Fine classifier mappings
    group0_fine_map = {orig_idx: local_idx for local_idx, orig_idx in enumerate(group0_indices)}
    group1_fine_map = {orig_idx: local_idx for local_idx, orig_idx in enumerate(group1_indices)}

    # Create subsets
    group_train_ds = RemappedSubset(base_dataset, train_indices, group_label_map)
    group_val_ds = RemappedSubset(base_dataset, val_indices, group_label_map)

    group0_train_indices = [idx for idx in train_indices if base_dataset.samples[idx][1] in group0_indices]
    group0_val_indices = [idx for idx in val_indices if base_dataset.samples[idx][1] in group0_indices]
    group0_train_ds = RemappedSubset(base_dataset, group0_train_indices, group0_fine_map)
    group0_val_ds = RemappedSubset(base_dataset, group0_val_indices, group0_fine_map)

    group1_train_indices = [idx for idx in train_indices if base_dataset.samples[idx][1] in group1_indices]
    group1_val_indices = [idx for idx in val_indices if base_dataset.samples[idx][1] in group1_indices]
    group1_train_ds = RemappedSubset(base_dataset, group1_train_indices, group1_fine_map)
    group1_val_ds = RemappedSubset(base_dataset, group1_val_indices, group1_fine_map)

    return (group_train_ds, group_val_ds), (group0_train_ds, group0_val_ds), (group1_train_ds, group1_val_ds)


def evaluate_hierarchical():
    """Evaluate hierarchical CNN models on validation set."""
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dataset = UltrasonicDataset(
        config["paths"]["raw_dir"],
        transform=False,
        preload=(device.type == "cpu"),
    )
    all_class_names = base_dataset.classes
    print(f"Detected Classes: {all_class_names}")

    if len(base_dataset) == 0:
        print("Dataset is empty. Check your data paths.")
        return

    # Use all 6 classes
    selected_label_indices = list(range(len(all_class_names)))

    # Get validation split (same as training)
    train_indices, val_indices, _, _ = _build_recording_split(
        base_dataset.samples,
        selected_label_indices,
        val_ratio=0.2,
        seed=42,
        quick=False,
    )

    # Create validation dataset - use RemappedSubset with identity mapping for proper subset
    identity_map = {i: i for i in range(len(all_class_names))}
    val_ds = RemappedSubset(base_dataset, val_indices, identity_map)

    # Load models
    models_dir = Path(__file__).resolve().parents[1] / "models"
    group_model_path = models_dir / "fius_group_cnn.pth"
    group0_model_path = models_dir / "fius_group0_cnn.pth"
    group1_model_path = models_dir / "fius_group1_cnn.pth"

    if not all(p.exists() for p in [group_model_path, group0_model_path, group1_model_path]):
        print("Missing trained models. Please run train_hierarchical.py first.")
        return

    group_model = UltrasonicCNN(num_classes=2).to(device)
    group_model.load_state_dict(torch.load(group_model_path, map_location=device))
    group_model.eval()

    group0_model = UltrasonicCNN(num_classes=3).to(device)
    group0_model.load_state_dict(torch.load(group0_model_path, map_location=device))
    group0_model.eval()

    group1_model = UltrasonicCNN(num_classes=3).to(device)
    group1_model.load_state_dict(torch.load(group1_model_path, map_location=device))
    group1_model.eval()

    # Define class mappings
    group0_classes = ["person", "chair", "plant"]
    group1_classes = ["wall", "backpack", "bigtable"]

    group0_indices = [i for i, name in enumerate(all_class_names) if name in group0_classes]
    group1_indices = [i for i, name in enumerate(all_class_names) if name in group1_classes]

    # Group to original class mapping for fine classifiers
    group0_to_orig = {local: orig for local, orig in enumerate(group0_indices)}
    group1_to_orig = {local: orig for local, orig in enumerate(group1_indices)}

    # Create validation data loader
    effective_batch_size = int(config["training"].get("batch_size", 8))
    val_loader = DataLoader(
        val_ds,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("Evaluating hierarchical classifier...")

    # Evaluation metrics
    group_correct = 0
    group_total = 0
    final_correct = 0
    final_total = 0
    num_classes = len(all_class_names)
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)

            # Step 1: Group classification
            group_logits = group_model(signals)
            group_probs = torch.softmax(group_logits, dim=1)
            group_preds = torch.argmax(group_probs, dim=1)

            # Count group accuracy
            true_groups = torch.zeros_like(labels)
            for i, label in enumerate(labels):
                if label.item() in group0_indices:
                    true_groups[i] = 0
                else:
                    true_groups[i] = 1
            group_correct += (group_preds == true_groups).sum().item()
            group_total += labels.size(0)

            # Step 2: Route to fine classifiers
            final_preds = torch.zeros_like(labels)

            # Group 0 samples
            group0_mask = (group_preds == 0)
            if group0_mask.any():
                group0_signals = signals[group0_mask]
                group0_logits = group0_model(group0_signals)
                group0_probs = torch.softmax(group0_logits, dim=1)
                group0_fine_preds = torch.argmax(group0_probs, dim=1)

                # Map back to original classes using proper indexing
                group0_indices_tensor = torch.where(group0_mask)[0]
                for i, fine_pred in enumerate(group0_fine_preds):
                    final_preds[group0_indices_tensor[i]] = group0_to_orig[fine_pred.item()]

            # Group 1 samples
            group1_mask = (group_preds == 1)
            if group1_mask.any():
                group1_signals = signals[group1_mask]
                group1_logits = group1_model(group1_signals)
                group1_probs = torch.softmax(group1_logits, dim=1)
                group1_fine_preds = torch.argmax(group1_probs, dim=1)

                # Map back to original classes using proper indexing
                group1_indices_tensor = torch.where(group1_mask)[0]
                for i, fine_pred in enumerate(group1_fine_preds):
                    final_preds[group1_indices_tensor[i]] = group1_to_orig[fine_pred.item()]

            # Update final accuracy
            final_correct += (final_preds == labels).sum().item()
            final_total += labels.size(0)

            # Update confusion matrix
            labels_cpu = labels.detach().cpu()
            preds_cpu = final_preds.detach().cpu()
            linear_idx = labels_cpu * 6 + preds_cpu
            conf_mat += torch.bincount(linear_idx, minlength=6 * 6).reshape(6, 6)

    # Calculate metrics
    group_acc = 100.0 * group_correct / max(1, group_total)
    final_acc = 100.0 * final_correct / max(1, final_total)
    final_macro_f1 = _compute_macro_f1(conf_mat)

    # Per-class accuracy
    per_class_acc = []
    for i in range(6):
        tp = conf_mat[i, i].item()
        total = conf_mat[i, :].sum().item()
        acc = 100.0 * tp / max(1, total)
        per_class_acc.append(acc)

    print("\nFinal Validation Metrics:")
    print(f"Group Classifier Accuracy : {group_acc:.2f}%")
    print(f"Final Hierarchical Accuracy : {final_acc:.2f}%")
    print(f"Final Hierarchical Macro-F1 : {final_macro_f1:.4f}")

    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(all_class_names):
        print(f"{class_name:<10}: {per_class_acc[i]:.2f}%")

    _print_confusion_matrix(conf_mat, all_class_names)


if __name__ == "__main__":
    evaluate_hierarchical()