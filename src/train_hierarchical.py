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
from model import UltrasonicHierarchicalSoftCNN
from data_loader import load_config

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


GROUP_PRESETS = {
    "preset_a": {
        "group0": ["person", "backpack", "plant"],
        "group1": ["wall", "chair"],
    },
    "preset_b": {
        "group0": ["person", "chair", "plant"],
        "group1": ["wall", "backpack"],
    },
    "preset_c": {
        "group0": ["wall", "chair"],
        "group1": ["person", "backpack", "plant"],
    },
}


def get_grouping(grouping, available_classes):
    if grouping not in GROUP_PRESETS:
        raise ValueError(f"Invalid grouping preset: {grouping}. Valid: {list(GROUP_PRESETS.keys())}")
    group0 = [c for c in GROUP_PRESETS[grouping]["group0"] if c in available_classes]
    group1 = [c for c in GROUP_PRESETS[grouping]["group1"] if c in available_classes]
    if not group0 or not group1:
        raise ValueError(f"Grouping {grouping} does not match available classes: {available_classes}")
    return group0, group1


def build_group_label_maps(group0_names, group1_names, all_classes):
    label_to_group = {}
    label_to_fine_local = {}
    group0_indices = []
    group1_indices = []

    for idx, c in enumerate(all_classes):
        if c in group0_names:
            label_to_group[idx] = 0
            label_to_fine_local[idx] = group0_names.index(c)
            group0_indices.append(idx)
        elif c in group1_names:
            label_to_group[idx] = 1
            label_to_fine_local[idx] = group1_names.index(c)
            group1_indices.append(idx)
        else:
            raise ValueError(f"Class {c} is not in group0 or group1 for hierarchical preset")

    return label_to_group, label_to_fine_local, group0_indices, group1_indices


def ensure_split_indices(path, samples, selected_label_indices, val_ratio=0.2, seed=42, quick=False):
    if path.exists():
        data = torch.load(path)
        return data["train_indices"], data["val_indices"]

    train_indices, val_indices, _, _ = _build_recording_split(samples, selected_label_indices, val_ratio=val_ratio, seed=seed, quick=quick)
    data = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "selected_label_indices": selected_label_indices,
    }
    torch.save(data, path)
    return train_indices, val_indices


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


def train_hierarchical(epochs=None, batch_size=None, quick=False, grouping="preset_a", loss_type="ce", balanced_sampling=False):
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

    if len(all_class_names) != 5:
        raise ValueError("Hierarchical training requires exactly 5 merged classes (wall, person, chair, backpack, plant).")

    group0_names, group1_names = get_grouping(grouping, all_class_names)
    print(f"Grouping preset: {grouping}")
    print(f"  group0: {group0_names}")
    print(f"  group1: {group1_names}")

    # all classes selected by default for hierarchical training.
    selected_label_indices = list(range(len(all_class_names)))
    selected_class_names = [all_class_names[i] for i in selected_label_indices]

    group_label_map, fine_local_map, group0_indices, group1_indices = build_group_label_maps(group0_names, group1_names, selected_class_names)

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    split_path = results_dir / "split_indices_hierarchical.pth"

    train_indices, val_indices = ensure_split_indices(split_path, base_dataset.samples, selected_label_indices, val_ratio=0.2, seed=42, quick=quick)

    print("Recording Counts Per Class:")
    class_rec_counts = {}
    for class_idx in selected_label_indices:
        train_rec = sum(1 for i in train_indices if base_dataset.samples[i][1] == class_idx)
        val_rec = sum(1 for i in val_indices if base_dataset.samples[i][1] == class_idx)
        class_rec_counts[selected_class_names[class_idx]] = {"train": train_rec, "val": val_rec}
        print(f"{selected_class_names[class_idx]:<10} train_rec={train_rec} val_rec={val_rec}")

    train_ds = RemappedSubset(base_dataset, train_indices, {idx: idx for idx in selected_label_indices})
    val_ds   = RemappedSubset(base_dataset, val_indices, {idx: idx for idx in selected_label_indices})

    num_classes = len(selected_class_names)
    num_group0 = len(group0_names)
    num_group1 = len(group1_names)

    weights_cpu, class_counts = _compute_inverse_frequency_weights(base_dataset.samples, train_indices, {idx: idx for idx in selected_label_indices}, num_classes)
    print("\nComputed Class Weights (inverse-frequency from training split):")
    for i, class_name in enumerate(selected_class_names):
        print(f"{class_name:<10} count={int(class_counts[i].item()):<6} weight={weights_cpu[i].item():.4f}")

    train_sampler = None
    train_shuffle = True
    if balanced_sampling:
        sample_weights = []
        for sample_idx in train_ds.indices:
            _, original_label, _ = base_dataset.samples[sample_idx]
            sample_weights.append(1.0 / max(1e-12, float(class_counts[original_label].item())))
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_shuffle = False

    effective_batch_size = int(batch_size if batch_size is not None else config["training"].get("batch_size", 8))
    num_workers=0
    pin_memory=torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=train_shuffle, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=effective_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = UltrasonicHierarchicalSoftCNN(num_group0=num_group0, num_group1=num_group1).to(device)
    model.apply(init_weights)

    if loss_type == "focal":
        group_criterion = FocalLoss(gamma=2.0, smoothing=0.1)
        fine0_criterion = FocalLoss(gamma=2.0, smoothing=0.1)
        fine1_criterion = FocalLoss(gamma=2.0, smoothing=0.1)
        print("Using focal loss for all heads")
    else:
        group_criterion = nn.CrossEntropyLoss()
        fine0_criterion = nn.CrossEntropyLoss()
        fine1_criterion = nn.CrossEntropyLoss()
        print("Using CE loss for all heads")

    optimizer = optim.Adam(model.parameters(), lr=float(config["training"].get("learning_rate", 1e-4)), weight_decay=float(config["training"].get("weight_decay", 1e-3)))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=10)

    group_target_map = torch.tensor([group_label_map[i] for i in selected_label_indices], dtype=torch.long, device=device)
    fine_target_map = torch.tensor([fine_local_map[i] for i in selected_label_indices], dtype=torch.long, device=device)

    best_metric = -1.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_group_acc": [],
        "val_soft_acc": [],
        "val_soft_macro_f1": [],
        "class_names": selected_class_names,
        "grouping": grouping,
        "group0": group0_names,
        "group1": group1_names,
        "split_path": str(split_path),
    }

    ckpt_path = Path(__file__).resolve().parents[1] / "models" / "fius_hierarchical_soft_cnn.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    for epoch in range(int(epochs if epochs is not None else min(30, int(config["training"].get("epochs", 30))))):
        model.train(); epoch_loss=0.0
        for signals, labels in train_loader:
            signals=signals.to(device);labels=labels.to(device)
            optimizer.zero_grad()
            g_logits, f0_logits, f1_logits = model(signals)
            target_group = group_target_map[labels]
            loss_group = group_criterion(g_logits, target_group)

            mask0 = target_group==0
            mask1 = target_group==1
            loss_f0 = fine0_criterion(f0_logits[mask0], fine_target_map[labels[mask0]]) if mask0.any() else torch.tensor(0.0, device=device)
            loss_f1 = fine1_criterion(f1_logits[mask1], fine_target_map[labels[mask1]]) if mask1.any() else torch.tensor(0.0, device=device)

            loss = loss_group + loss_f0 + loss_f1
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss=epoch_loss/max(1,len(train_loader))

        # validation
        model.eval(); val_loss=0.0; all_preds=[]; all_labels=[]; group_correct=0; group_total=0
        conf_mat=torch.zeros((num_classes,num_classes),dtype=torch.long)
        with torch.no_grad():
            for signals, labels in val_loader:
                signals=signals.to(device);labels=labels.to(device)
                g_logits,f0_logits,f1_logits=model(signals)
                target_group=group_target_map[labels]
                vloss_group=group_criterion(g_logits,target_group)
                mask0=target_group==0; mask1=target_group==1
                vloss_f0=fine0_criterion(f0_logits[mask0], fine_target_map[labels[mask0]]) if mask0.any() else torch.tensor(0.0, device=device)
                vloss_f1=fine1_criterion(f1_logits[mask1], fine_target_map[labels[mask1]]) if mask1.any() else torch.tensor(0.0, device=device)
                vloss=vloss_group+vloss_f0+vloss_f1
                val_loss += vloss.item()

                g_probs=torch.softmax(g_logits,dim=1)
                f0_probs=torch.softmax(f0_logits,dim=1)
                f1_probs=torch.softmax(f1_logits,dim=1)

                final_probs=torch.zeros((labels.size(0),num_classes),device=device)
                if group0_indices:
                    final_probs[:,group0_indices]=g_probs[:,0:1]*f0_probs
                if group1_indices:
                    final_probs[:,group1_indices]=g_probs[:,1:2]*f1_probs

                pred=final_probs.argmax(dim=1)
                all_preds.append(pred.cpu())
                all_labels.append(labels.cpu())

                group_pred=g_probs.argmax(dim=1)
                group_correct += (group_pred==target_group).sum().item(); group_total += labels.size(0)

                linear_idx=labels.cpu()*num_classes+pred.cpu()
                conf_mat += torch.bincount(linear_idx, minlength=num_classes*num_classes).reshape(num_classes,num_classes)

        val_acc=(sum((p==t).sum().item() for p,t in zip(all_preds,all_labels))/sum(len(t) for t in all_labels))*100.0
        val_macro_f1=_compute_macro_f1(conf_mat)
        val_group_acc=100.0*group_correct/max(1,group_total)
        avg_val_loss=val_loss/max(1,len(val_loader))

        history['train_loss'].append(avg_train_loss);history['val_loss'].append(avg_val_loss);history['val_group_acc'].append(val_group_acc);history['val_soft_acc'].append(val_acc);history['val_soft_macro_f1'].append(val_macro_f1)

        improved=False
        if val_macro_f1 > best_metric:
            best_metric = val_macro_f1
            improved = True
            metadata = {
                "class_names": selected_class_names,
                "grouping": grouping,
                "group0": group0_names,
                "group1": group1_names,
                "group_label_map": group_label_map,
                "fine_local_map": fine_local_map,
                "group0_indices": group0_indices,
                "group1_indices": group1_indices,
                "split_path": str(split_path),
            }
            torch.save({"model_state": model.state_dict(), "metadata": metadata}, ckpt_path)

        if scheduler is not None:
            scheduler.step(avg_val_loss)
        early_stopping(val_group_acc)

        print(f"Epoch [{epoch+1}/{int(epochs if epochs else min(30,int(config['training'].get('epochs',30))))}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Group Acc: {val_group_acc:.2f}% | Soft Acc: {val_acc:.2f}% | Soft Macro-F1: {val_macro_f1:.4f}")

        if early_stopping.early_stop: print('Early stopping'); break

    # save growth history
    history_path = results_dir / 'training_history_hierarchical.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history: {history_path}")

    split_history_path = results_dir / 'split_indices_hierarchical.pth'
    # ensure split saved under hierarchical file (already done in ensure_split_indices)
    if split_path.exists():
        torch.save(torch.load(split_path), split_history_path)
    print(f"Saved split indices: {split_history_path}")

    print(f"Saved checkpoint: {ckpt_path}")

    # end of train_hierarchical


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (default: 30).")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--quick", action="store_true", help="Train on a small random subset for fast debugging.")
    parser.add_argument("--grouping", type=str, default="preset_a", choices=["preset_a", "preset_b", "preset_c"], help="Grouping preset for hierarchical training.")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"], help="Loss type.")
    parser.add_argument("--balanced_sampling", action="store_true", help="Enable balanced class sampling in train loader.")
    args = parser.parse_args()
    train_hierarchical(
        epochs=args.epochs,
        batch_size=args.batch_size,
        quick=args.quick,
        grouping=args.grouping,
        loss_type=args.loss,
        balanced_sampling=args.balanced_sampling,
    )