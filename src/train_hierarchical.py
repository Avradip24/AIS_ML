import argparse
import time
import random
import json
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataset import UltrasonicDataset
from model import UltrasonicCNN
from data_loader import load_config

"""
Hierarchical CNN Training Script for FIUS Project

This script trains a hierarchical CNN pipeline consisting of:
1. Group classifier (2 classes): person/chair/plant vs wall/backpack/bigtable
2. Group0 fine classifier (3 classes): person, chair, plant
3. Group1 fine classifier (3 classes): wall, backpack, bigtable

Usage:
    python src/train_hierarchical.py --epochs 30 --batch_size 64 --loss ce --balanced_sampling

Arguments:
    --epochs: Number of training epochs (default: 30)
    --batch_size: Batch size for training (default: from config)
    --quick: Train on small subset for debugging
    --loss: Loss type, 'ce' or 'focal' (default: ce)
    --balanced_sampling: Enable balanced sampling (default: enabled)

Output:
    Saves models to models/fius_group_cnn.pth, models/fius_group0_cnn.pth, models/fius_group1_cnn.pth
"""

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def __call__(self, metric):
        if self.best_metric is None:
            self.best_metric = metric
            return

        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean", smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # New Logic: Apply Label Smoothing
        num_classes = logits.size(-1)
        with torch.no_grad():
            smoothed_labels = torch.full_like(logits, self.smoothing / (num_classes - 1))
            smoothed_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Focal calculation with smoothed labels
        focal_term = (1.0 - probs) ** self.gamma
        loss = -smoothed_labels * focal_term * log_probs

        if self.alpha is not None:
            loss = self.alpha.view(1, -1) * loss

        if self.reduction == "mean":
            return loss.sum(dim=1).mean()
        return loss.sum()


def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def _compute_macro_f1(conf_mat):
    eps = 1e-8
    tp = conf_mat.diag().float()
    fp = conf_mat.sum(dim=0).float() - tp
    fn = conf_mat.sum(dim=1).float() - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return float(f1.mean().item())


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


def _compute_inverse_frequency_weights_from_remapped_dataset(dataset, num_classes):
    """Compute inverse-frequency weights directly from remapped dataset labels."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, label in dataset:
        counts[label] += 1.0

    if torch.any(counts <= 0):
        raise ValueError(f"Invalid class distribution in dataset: {counts.tolist()}")

    inv = 1.0 / counts
    # Normalize to mean=1 to keep loss scale stable across runs.
    weights = inv / inv.mean()
    return weights, counts


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


def _train_single_model(model_name, train_ds, val_ds, num_classes, config, epochs, batch_size, loss_type, balanced_sampling, device):
    """Train a single CNN model."""
    print(f"Best model saved → fius_{model_name}_cnn.pth")

    # Compute weights for this subset directly from remapped dataset
    weights_cpu, class_counts = _compute_inverse_frequency_weights_from_remapped_dataset(train_ds, num_classes)
    print(f"  Class distribution: {class_counts.tolist()}")

    effective_batch_size = int(batch_size if batch_size is not None else config["training"].get("batch_size", 8))
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    train_sampler = None
    train_shuffle = True
    if balanced_sampling:
        sample_weights = []
        for _, label in train_ds:
            sample_weights.append(1.0 / max(1e-12, float(class_counts[label].item())))
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = UltrasonicCNN(num_classes=num_classes).to(device)
    model.apply(init_weights)

    learning_rate = float(config["training"].get("learning_rate", 1e-4))
    weight_decay = float(config["training"].get("weight_decay", 1e-3))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=10)

    weights = weights_cpu.to(device)
    if loss_type == "focal":
        criterion = FocalLoss(alpha=weights, gamma=2.0, reduction="mean", smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    effective_epochs = int(epochs if epochs is not None else min(30, int(config["training"].get("epochs", 30))))
    best_macro_f1 = -1.0
    best_val_loss = float("inf")

    for epoch in range(effective_epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_loss = 0.0

        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            class_logits = model(signals)
            loss = criterion(class_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss, val_acc, val_macro_f1, _ = _evaluate_confusion(
            model, val_loader, criterion, num_classes, device
        )

        scheduler.step(avg_val_loss)
        early_stopping(val_macro_f1)

        improved = False
        if val_macro_f1 > best_macro_f1 + 1e-6:
            improved = True
        elif abs(val_macro_f1 - best_macro_f1) <= 1e-6 and avg_val_loss < best_val_loss - 1e-6:
            improved = True

        if improved:
            best_macro_f1 = val_macro_f1
            best_val_loss = avg_val_loss
            model_path = Path(__file__).resolve().parents[1] / "models" / f"fius_{model_name}_cnn.pth"
            torch.save(model.state_dict(), model_path)

        epoch_time = time.perf_counter() - epoch_start
        print(
            f"  Epoch [{epoch+1:03d}/{effective_epochs}] | Loss: {avg_train_loss:.3f} | "
            f"Val Loss: {avg_val_loss:.3f} | Val Acc: {val_acc:.1f}% | "
            f"Val Macro-F1: {val_macro_f1:.4f} | Time: {epoch_time:.1f}s"
        )

        if early_stopping.early_stop:
            print("  Early stopping triggered.")
            break

    print(f"  {model_name} training complete! Best Val Macro-F1: {best_macro_f1:.4f}")
    return best_macro_f1


def train_hierarchical(epochs=None, batch_size=None, quick=False, loss_type="ce", balanced_sampling=True):
    """Train hierarchical CNN models."""
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

    # Use all 6 classes for hierarchical training
    selected_label_indices = list(range(len(all_class_names)))

    train_indices, val_indices, train_rec_counts, val_rec_counts = _build_recording_split(
        base_dataset.samples,
        selected_label_indices,
        val_ratio=0.2,
        seed=42,
        quick=quick,
    )
    if quick:
        print("Quick mode enabled: limited recording count per class before splitting.")

    print("\nRecording Counts Per Class:")
    for local_idx, orig_idx in enumerate(selected_label_indices):
        class_name = all_class_names[orig_idx]
        tr = train_rec_counts.get(orig_idx, 0)
        va = val_rec_counts.get(orig_idx, 0)
        print(f"{class_name:<10} train_rec={tr:<3} val_rec={va:<3}")

    # Create group and fine datasets
    (group_train_ds, group_val_ds), (group0_train_ds, group0_val_ds), (group1_train_ds, group1_val_ds) = _create_group_datasets(
        base_dataset, train_indices, val_indices
    )

    # Train group classifier (2 classes)
    _train_single_model("group", group_train_ds, group_val_ds, 2, config, epochs, batch_size, loss_type, balanced_sampling, device)

    # Train group0 fine classifier (3 classes)
    _train_single_model("group0", group0_train_ds, group0_val_ds, 3, config, epochs, batch_size, loss_type, balanced_sampling, device)

    # Train group1 fine classifier (3 classes)
    _train_single_model("group1", group1_train_ds, group1_val_ds, 3, config, epochs, batch_size, loss_type, balanced_sampling, device)

    print("\nHierarchical training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--balanced_sampling", action="store_true", default=True)
    parser.add_argument("--no_balanced_sampling", action="store_true")

    args = parser.parse_args()

    balanced_sampling = True
    if args.no_balanced_sampling:
        balanced_sampling = False

    train_hierarchical(
        epochs=args.epochs,
        batch_size=args.batch_size,
        quick=args.quick,
        loss_type=args.loss,
        balanced_sampling=balanced_sampling,
    )