import argparse
import time
import random
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

# ---------------------------------------------------------------------------
# Acoustic grouping (v2 - KEY IMPROVEMENT)
# Group 0 (soft/absorbing):  person, backpack, plant
#   -> these scatter/absorb ultrasound: weaker, broader echo peaks
# Group 1 (hard/reflective): wall, chair
#   -> these reflect cleanly: sharp, strong echo peaks
# Note: bigtable is merged into wall in dataset.py (acoustically identical).
# This replaces the previous arbitrary split which had no acoustic basis.
# ---------------------------------------------------------------------------
GROUP0_CLASSES = ["person", "backpack", "plant"]   # soft / absorbing
GROUP1_CLASSES = ["wall", "chair"]                 # hard / reflective


def augment_signal(x):
    """Random augmentation for a [4, length] signal tensor."""
    x = x.clone()
    length = x.shape[1]
    # Gaussian noise on ADC
    if random.random() < 0.7:
        x[0] += torch.randn_like(x[0]) * 0.03
        x[1] += torch.randn_like(x[1]) * 0.01
    # Gaussian noise on FFT
    if random.random() < 0.5:
        x[2] += torch.randn_like(x[2]) * 0.015
        x[3] += torch.randn_like(x[3]) * 0.005
    # Random time shift (circular roll on ADC channels)
    if random.random() < 0.5:
        shift = random.randint(-int(length * 0.05), int(length * 0.05))
        x[0] = torch.roll(x[0], shift)
        x[1] = torch.roll(x[1], shift)
    # Random amplitude scaling
    if random.random() < 0.6:
        scale = random.uniform(0.85, 1.15)
        x[0] *= scale
        x[2] *= scale
    return x


class RemappedSubset(Dataset):
    def __init__(self, base_dataset, indices, label_map, augment=False):
        self.base_dataset = base_dataset
        self.indices = indices
        self.label_map = label_map
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base_dataset[self.indices[idx]]
        if self.augment:
            x = augment_signal(x)
        mapped = self.label_map[int(y.item())]
        return x, torch.tensor(mapped).long()


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
        num_classes = logits.size(-1)
        with torch.no_grad():
            smoothed = torch.full_like(logits, self.smoothing / (num_classes - 1))
            smoothed.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        focal_term = (1.0 - probs) ** self.gamma
        loss = -smoothed * focal_term * log_probs
        if self.alpha is not None:
            loss = self.alpha.view(1, -1) * loss
        return loss.sum(dim=1).mean() if self.reduction == "mean" else loss.sum()


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def _compute_macro_f1(conf_mat):
    eps = 1e-8
    tp = conf_mat.diag().float()
    fp = conf_mat.sum(dim=0).float() - tp
    fn = conf_mat.sum(dim=1).float() - tp
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    return float(f1.mean().item())


def _compute_inverse_frequency_weights(dataset, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, label in dataset:
        counts[label] += 1.0
    if torch.any(counts <= 0):
        raise ValueError(f"Empty class in dataset: {counts.tolist()}")
    inv = 1.0 / counts
    return inv / inv.mean(), counts


def _build_recording_split(samples, selected_labels, val_ratio=0.2, seed=42, quick=False):
    file_to_indices = defaultdict(list)
    file_to_label = {}
    for idx, (file_path, label, _seg) in enumerate(samples):
        if label not in selected_labels:
            continue
        file_to_indices[file_path].append(idx)
        file_to_label[file_path] = label

    class_to_files = defaultdict(list)
    for fp, label in file_to_label.items():
        class_to_files[label].append(fp)

    rng = random.Random(seed)
    train_files, val_files = set(), set()
    train_counts, val_counts = defaultdict(int), defaultdict(int)

    for label in selected_labels:
        files = sorted(class_to_files.get(label, []))
        rng.shuffle(files)
        if quick:
            files = files[:min(len(files), 3)]
        if not files:
            continue
        if len(files) == 1:
            class_val, class_train = [], files
        else:
            n_val = max(1, int(round(len(files) * val_ratio)))
            n_val = min(n_val, len(files) - 1)
            class_val = files[:n_val]
            class_train = files[n_val:]
        train_files.update(class_train)
        val_files.update(class_val)
        train_counts[label] += len(class_train)
        val_counts[label] += len(class_val)

    train_idx, val_idx = [], []
    for fp, indices in file_to_indices.items():
        if fp in train_files:
            train_idx.extend(indices)
        elif fp in val_files:
            val_idx.extend(indices)

    if not train_idx:
        raise ValueError("No training segments after recording-level split.")
    if not val_idx:
        raise ValueError("No validation segments after recording-level split.")
    return train_idx, val_idx, train_counts, val_counts


def _create_group_datasets(base_dataset, train_indices, val_indices):
    all_classes = base_dataset.classes
    group0_indices = [i for i, n in enumerate(all_classes) if n in GROUP0_CLASSES]
    group1_indices = [i for i, n in enumerate(all_classes) if n in GROUP1_CLASSES]

    if not group0_indices or not group1_indices:
        raise ValueError(
            f"Could not find all required classes.\n"
            f"  Expected group0: {GROUP0_CLASSES}\n"
            f"  Expected group1: {GROUP1_CLASSES}\n"
            f"  Dataset classes: {all_classes}"
        )

    print(f"\nAcoustic group mapping:")
    print(f"  Group 0 (soft/absorbing) : {[all_classes[i] for i in group0_indices]}")
    print(f"  Group 1 (hard/reflective): {[all_classes[i] for i in group1_indices]}")

    group_label_map = {i: 0 for i in group0_indices}
    group_label_map.update({i: 1 for i in group1_indices})
    group0_fine_map = {orig: local for local, orig in enumerate(group0_indices)}
    group1_fine_map = {orig: local for local, orig in enumerate(group1_indices)}

    group_train_ds = RemappedSubset(base_dataset, train_indices, group_label_map, augment=True)
    group_val_ds   = RemappedSubset(base_dataset, val_indices,   group_label_map, augment=False)

    g0_tr = [i for i in train_indices if base_dataset.samples[i][1] in group0_indices]
    g0_vl = [i for i in val_indices   if base_dataset.samples[i][1] in group0_indices]
    g1_tr = [i for i in train_indices if base_dataset.samples[i][1] in group1_indices]
    g1_vl = [i for i in val_indices   if base_dataset.samples[i][1] in group1_indices]

    return (
        (group_train_ds, group_val_ds),
        (RemappedSubset(base_dataset, g0_tr, group0_fine_map, augment=True),
         RemappedSubset(base_dataset, g0_vl, group0_fine_map, augment=False)),
        (RemappedSubset(base_dataset, g1_tr, group1_fine_map, augment=True),
         RemappedSubset(base_dataset, g1_vl, group1_fine_map, augment=False)),
    )


def _evaluate_confusion(model, loader, criterion, num_classes, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            logits = model(signals)
            total_loss += criterion(logits, labels).item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            lin = labels.cpu() * num_classes + predicted.cpu()
            conf_mat += torch.bincount(lin, minlength=num_classes**2).reshape(num_classes, num_classes)
    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total), _compute_macro_f1(conf_mat), conf_mat


def _train_single_model(model_name, train_ds, val_ds, num_classes,
                         config, epochs, batch_size, loss_type,
                         balanced_sampling, device):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}  ({num_classes} classes)")
    print(f"{'='*60}")

    weights_cpu, class_counts = _compute_inverse_frequency_weights(train_ds, num_classes)
    print(f"  Class distribution: {class_counts.tolist()}")

    eff_batch = int(batch_size or config["training"].get("batch_size", 8))
    pin_mem = torch.cuda.is_available()
    train_sampler = None
    train_shuffle = True

    if balanced_sampling:
        sw = [1.0 / max(1e-12, float(class_counts[lbl].item())) for _, lbl in train_ds]
        train_sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), len(sw), replacement=True)
        train_shuffle = False

    train_loader = DataLoader(train_ds, batch_size=eff_batch, shuffle=train_shuffle,
                               sampler=train_sampler, num_workers=0, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds, batch_size=eff_batch, shuffle=False,
                               num_workers=0, pin_memory=pin_mem)

    model = UltrasonicCNN(num_classes=num_classes).to(device)
    model.apply(init_weights)

    lr = float(config["training"].get("learning_rate", 1e-4))
    wd = float(config["training"].get("weight_decay", 1e-3))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5, factor=0.5)
    stopper   = EarlyStopping(patience=12)

    weights = weights_cpu.to(device)
    criterion = (FocalLoss(alpha=weights, gamma=2.0, smoothing=0.1)
                 if loss_type == "focal"
                 else nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1))

    eff_epochs = int(epochs or min(50, int(config["training"].get("epochs", 50))))
    best_f1 = -1.0
    best_val_loss = float("inf")
    model_path = Path(__file__).resolve().parents[1] / "models" / f"fius_{model_name}_cnn.pth"

    for epoch in range(eff_epochs):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(signals), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / max(1, len(train_loader))
        avg_val, val_acc, f1, _ = _evaluate_confusion(model, val_loader, criterion, num_classes, device)
        scheduler.step(avg_val)
        stopper(f1)

        improved = (f1 > best_f1 + 1e-6) or (abs(f1 - best_f1) <= 1e-6 and avg_val < best_val_loss - 1e-6)
        if improved:
            best_f1 = f1
            best_val_loss = avg_val
            torch.save(model.state_dict(), model_path)

        print(
            f"  Ep [{epoch+1:03d}/{eff_epochs}] "
            f"Loss: {avg_train:.3f} | Val: {avg_val:.3f} | "
            f"Acc: {val_acc:.1f}% | F1: {f1:.4f} | "
            f"[{'SAVED' if improved else '     '}] {time.perf_counter()-t0:.1f}s"
        )
        if stopper.early_stop:
            print("  Early stopping triggered.")
            break

    print(f"  {model_name} done. Best Val F1: {best_f1:.4f}  -> {model_path}")
    return best_f1


def train_hierarchical(epochs=None, batch_size=None, quick=False,
                        loss_type="focal", balanced_sampling=True):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_ds = UltrasonicDataset(config["paths"]["raw_dir"], transform=False,
                                 preload=(device.type == "cpu"))
    all_classes = base_ds.classes
    print(f"Detected classes: {all_classes}")

    if len(base_ds) == 0:
        print("Dataset is empty. Check your data paths.")
        return

    missing = [c for c in GROUP0_CLASSES + GROUP1_CLASSES if c not in all_classes]
    if missing:
        raise ValueError(f"Missing classes: {missing}. Available: {all_classes}")

    selected = list(range(len(all_classes)))
    tr_idx, vl_idx, tr_rec, vl_rec = _build_recording_split(
        base_ds.samples, selected, val_ratio=0.2, seed=42, quick=quick
    )

    print("\nRecording counts per class:")
    for i, name in enumerate(all_classes):
        print(f"  {name:<10} train={tr_rec.get(i,0):<3}  val={vl_rec.get(i,0):<3}")
    print(f"Total train segments: {len(tr_idx)}  |  val segments: {len(vl_idx)}")

    (g_tr, g_vl), (g0_tr, g0_vl), (g1_tr, g1_vl) = _create_group_datasets(base_ds, tr_idx, vl_idx)

    f1_g  = _train_single_model("group",  g_tr,  g_vl,  2, config, epochs, batch_size, loss_type, balanced_sampling, device)
    f1_g0 = _train_single_model("group0", g0_tr, g0_vl, 3, config, epochs, batch_size, loss_type, balanced_sampling, device)
    f1_g1 = _train_single_model("group1", g1_tr, g1_vl, len(GROUP1_CLASSES), config, epochs, batch_size, loss_type, balanced_sampling, device)

    print("\n" + "="*60)
    print("Hierarchical training complete!")
    print(f"  Group classifier  Val F1: {f1_g:.4f}")
    print(f"  Fine classifier 0 Val F1: {f1_g0:.4f}")
    print(f"  Fine classifier 1 Val F1: {f1_g1:.4f}")
    print("Run evaluate_hierarchical.py to see end-to-end accuracy.")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",               type=int,  default=None)
    parser.add_argument("--batch_size",           type=int,  default=None)
    parser.add_argument("--quick",                action="store_true")
    parser.add_argument("--loss",                 type=str,  default="focal", choices=["ce", "focal"])
    parser.add_argument("--balanced_sampling",    action="store_true", default=True)
    parser.add_argument("--no_balanced_sampling", action="store_true")
    args = parser.parse_args()
    train_hierarchical(
        epochs=args.epochs,
        batch_size=args.batch_size,
        quick=args.quick,
        loss_type=args.loss,
        balanced_sampling=not args.no_balanced_sampling,
    )