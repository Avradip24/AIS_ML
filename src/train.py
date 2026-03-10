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

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
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
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        ce_loss = -target_log_probs
        focal_term = (1.0 - target_probs) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

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


def _parse_selected_classes(classes_arg, available_classes):
    if not classes_arg:
        return list(range(len(available_classes)))

    requested = [c.strip().lower() for c in classes_arg.split(",") if c.strip()]
    if not requested:
        return list(range(len(available_classes)))

    missing = [c for c in requested if c not in available_classes]
    if missing:
        raise ValueError(f"Unknown classes in --classes: {missing}. Available: {available_classes}")

    selected = [i for i, c in enumerate(available_classes) if c in requested]
    if len(selected) < 2:
        raise ValueError("Please select at least 2 classes for training.")
    return selected


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


def _print_recording_counts(class_names, selected_label_indices, train_counts, val_counts):
    print("\nRecording Counts Per Class:")
    for local_idx, orig_idx in enumerate(selected_label_indices):
        class_name = class_names[orig_idx]
        tr = train_counts.get(orig_idx, 0)
        va = val_counts.get(orig_idx, 0)
        print(f"{class_name:<10} train_rec={tr:<3} val_rec={va:<3}")


def _print_confusion_matrix(conf_mat, selected_class_names):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = " " * 12 + " ".join([f"{c[:10]:>10}" for c in selected_class_names])
    print(header)
    for i, class_name in enumerate(selected_class_names):
        row_vals = " ".join([f"{int(conf_mat[i, j]):>10}" for j in range(len(selected_class_names))])
        print(f"{class_name[:10]:>10}  {row_vals}")


def _compute_inverse_frequency_weights(base_samples, train_indices, label_map, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for sample_idx in train_indices:
        _file_path, original_label, _segment_idx = base_samples[sample_idx]
        mapped_label = label_map[original_label]
        counts[mapped_label] += 1.0

    if torch.any(counts <= 0):
        raise ValueError(f"Invalid class distribution in training split: {counts.tolist()}")

    inv = 1.0 / counts
    # Normalize to mean=1 to keep loss scale stable across runs.
    weights = inv / inv.mean()
    return weights, counts


def train(epochs=None, batch_size=None, quick=False, classes=None, loss_type="ce", balanced_sampling=False):
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

    selected_label_indices = _parse_selected_classes(classes, all_class_names)
    selected_class_names = [all_class_names[i] for i in selected_label_indices]
    label_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(selected_label_indices)}
    print(f"Training classes: {selected_class_names}")

    train_indices, val_indices, train_rec_counts, val_rec_counts = _build_recording_split(
        base_dataset.samples,
        selected_label_indices,
        val_ratio=0.2,
        seed=42,
        quick=quick,
    )
    if quick:
        print("Quick mode enabled: limited recording count per class before splitting.")

    _print_recording_counts(all_class_names, selected_label_indices, train_rec_counts, val_rec_counts)

    train_ds = RemappedSubset(base_dataset, train_indices, label_map)
    val_ds = RemappedSubset(base_dataset, val_indices, label_map)

    # Compute class-frequency weights once (used by loss and optional balanced sampling).
    weights_cpu, class_counts = _compute_inverse_frequency_weights(
        base_dataset.samples,
        train_indices,
        label_map,
        len(selected_class_names),
    )
    print("\nComputed Class Weights (inverse-frequency from training split):")
    for i, class_name in enumerate(selected_class_names):
        print(f"{class_name:<10} count={int(class_counts[i].item()):<6} weight={weights_cpu[i].item():.4f}")

    effective_batch_size = int(batch_size if batch_size is not None else config["training"].get("batch_size", 8))
    num_workers = 0  # Windows-safe default for reproducibility and fewer worker startup costs.
    pin_memory = torch.cuda.is_available()
    train_sampler = None
    train_shuffle = True

    if balanced_sampling:
        sample_weights = []
        for sample_idx in train_ds.indices:
            _file_path, original_label, _segment_idx = base_dataset.samples[sample_idx]
            mapped_label = label_map[original_label]
            sample_weights.append(1.0 / max(1e-12, float(class_counts[mapped_label].item())))
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    print(f"Balanced sampling: {'enabled' if balanced_sampling else 'disabled'}")

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

    num_classes = len(selected_class_names)
    model = UltrasonicCNN(num_classes=num_classes).to(device)
    model.apply(init_weights)

    learning_rate = float(config["training"].get("learning_rate", 1e-4))
    weight_decay = float(config["training"].get("weight_decay", 1e-3))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=10)

    weights = weights_cpu.to(device)
    if loss_type == "focal":
        criterion = FocalLoss(alpha=weights, gamma=2.0, reduction="mean")
        print("Using loss: focal")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.0)
        print("Using loss: ce")
    effective_epochs = int(epochs if epochs is not None else min(30, int(config["training"].get("epochs", 30))))
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "class_names": selected_class_names,
    }

    total_start = time.perf_counter()

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
            torch.save(model.state_dict(), config["paths"]["model_output"])

        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch [{epoch+1:03d}/{effective_epochs}] | Loss: {avg_train_loss:.3f} | "
            f"Val Loss: {avg_val_loss:.3f} | Val Acc: {val_acc:.1f}% | "
            f"Val Macro-F1: {val_macro_f1:.4f} | Time: {epoch_time:.1f}s"
        )
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["val_accuracy"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_macro_f1))

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    total_time = time.perf_counter() - total_start
    print(f"\nTraining complete! Best Val Macro-F1: {best_macro_f1:.4f}")
    print(f"Total training time: {total_time:.1f}s")

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history: {history_path}")

    # Evaluate best checkpoint for trustworthy final metrics.
    model.load_state_dict(torch.load(config["paths"]["model_output"], map_location=device))
    final_val_loss, final_val_acc, final_macro_f1, final_conf_mat = _evaluate_confusion(
        model, val_loader, criterion, num_classes, device
    )

    print("\nFinal Validation Metrics:")
    print(f"Val Loss     : {final_val_loss:.4f}")
    print(f"Val Accuracy : {final_val_acc:.2f}%")
    print(f"Val Macro-F1 : {final_macro_f1:.4f}")
    _print_confusion_matrix(final_conf_mat, selected_class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: 30).")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--quick", action="store_true", help="Train on a small random subset for fast debugging.")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class list, e.g. person,bigtable")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"], help="Loss type.")
    parser.add_argument("--balanced_sampling", action="store_true", help="Enable balanced class sampling in train loader.")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        quick=args.quick,
        classes=args.classes,
        loss_type=args.loss,
        balanced_sampling=args.balanced_sampling,
    )