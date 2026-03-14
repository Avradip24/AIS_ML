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
Hierarchical CNN Evaluation Script (v2 — soft gating + acoustic grouping)

Key improvement: SOFT gating instead of hard gating.

  v1 (hard gating):  pick group 0 or 1, route 100% to that fine classifier.
                     One wrong group prediction = guaranteed wrong final answer.

  v2 (soft gating):  compute a WEIGHTED SUM of both fine classifiers' logits,
                     weighted by the group classifier's confidence:
                       final_logit = p(group=0) * fine0_logit
                                   + p(group=1) * fine1_logit
                     This way, even if the group classifier is uncertain, the
                     fine classifiers both contribute and the correct class
                     can still win.

Also uses the new acoustic grouping:
  Group 0 (soft/absorbing):  person, backpack, plant
  Group 1 (hard/reflective): wall, chair, bigtable
"""

GROUP0_CLASSES = ["person", "backpack", "plant"]   # soft / absorbing
GROUP1_CLASSES = ["wall", "chair"]                 # hard / reflective (bigtable merged into wall)


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


def _build_recording_split(samples, selected_labels, val_ratio=0.2, seed=42):
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
        val_counts[label]   += len(class_val)

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


def _print_confusion_matrix(conf_mat, class_names):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = " " * 12 + " ".join([f"{c[:10]:>10}" for c in class_names])
    print(header)
    for i, name in enumerate(class_names):
        row = " ".join([f"{int(conf_mat[i, j]):>10}" for j in range(len(class_names))])
        print(f"{name[:10]:>10}  {row}")


def evaluate_hierarchical():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_ds = UltrasonicDataset(config["paths"]["raw_dir"], transform=False,
                                 preload=(device.type == "cpu"))
    all_classes = base_ds.classes
    print(f"Detected classes: {all_classes}")

    if len(base_ds) == 0:
        print("Dataset is empty.")
        return

    # Build same val split as training (seed=42)
    selected = list(range(len(all_classes)))
    _, val_idx, _, _ = _build_recording_split(base_ds.samples, selected, val_ratio=0.2, seed=42)

    # Load models
    models_dir = Path(__file__).resolve().parents[1] / "models"
    paths = {
        "group":  models_dir / "fius_group_cnn.pth",
        "group0": models_dir / "fius_group0_cnn.pth",
        "group1": models_dir / "fius_group1_cnn.pth",
    }
    missing_models = [k for k, p in paths.items() if not p.exists()]
    if missing_models:
        print(f"Missing trained models: {missing_models}. Run train_hierarchical.py first.")
        return

    group_model  = UltrasonicCNN(num_classes=2).to(device)
    group0_model = UltrasonicCNN(num_classes=3).to(device)
    group1_model = UltrasonicCNN(num_classes=len(GROUP1_CLASSES)).to(device)
    group_model.load_state_dict(torch.load(paths["group"],  map_location=device, weights_only=True))
    group0_model.load_state_dict(torch.load(paths["group0"], map_location=device, weights_only=True))
    group1_model.load_state_dict(torch.load(paths["group1"], map_location=device, weights_only=True))
    group_model.eval()
    group0_model.eval()
    group1_model.eval()

    # Class index mappings
    group0_indices = [i for i, n in enumerate(all_classes) if n in GROUP0_CLASSES]
    group1_indices = [i for i, n in enumerate(all_classes) if n in GROUP1_CLASSES]
    print(f"\nGroup 0 (soft/absorbing) : {[all_classes[i] for i in group0_indices]}")
    print(f"Group 1 (hard/reflective): {[all_classes[i] for i in group1_indices]}")

    # Map fine classifier local indices -> original class indices
    group0_to_orig = {local: orig for local, orig in enumerate(group0_indices)}
    group1_to_orig = {local: orig for local, orig in enumerate(group1_indices)}

    # Validation loader — identity map (keep original labels)
    identity_map = {i: i for i in range(len(all_classes))}
    val_ds = RemappedSubset(base_ds, val_idx, identity_map)
    eff_batch = int(config["training"].get("batch_size", 8))
    val_loader = DataLoader(val_ds, batch_size=eff_batch, shuffle=False,
                             num_workers=0, pin_memory=torch.cuda.is_available())

    # Hard gating: route each sample to exactly one fine classifier based on group prediction.
    # Soft blending was tested but collapsed — group0 has 3 classes (person/backpack/plant)
    # and group1 has 2 (wall/chair), so person systematically wins the blend regardless of
    # the true label (wall=0%, chair=0% under soft blend). Hard gating at 47.6% > soft at 35.9%.
    print("\nEvaluating with HARD gating...")
    print("  Each sample routed to one fine classifier based on argmax(group_probs).")

    num_classes = len(all_classes)
    conf_mat    = torch.zeros((num_classes, num_classes), dtype=torch.long)
    group_correct = 0
    total         = 0

    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)
            bs = signals.size(0)

            # Step 1: group classification
            group_logits = group_model(signals)
            group_probs  = torch.softmax(group_logits, dim=1)   # [bs, 2]
            p_group0     = group_probs[:, 0]
            p_group1     = group_probs[:, 1]
            hard_group   = group_probs.argmax(dim=1)

            # Group accuracy tracking
            true_groups = torch.zeros(bs, dtype=torch.long, device=device)
            for i, lbl in enumerate(labels):
                true_groups[i] = 0 if lbl.item() in group0_indices else 1
            group_correct += (hard_group == true_groups).sum().item()
            total += bs

            # Step 2: both fine classifiers run on ALL samples
            fine0_logits = group0_model(signals)   # [bs, 3]
            fine1_logits = group1_model(signals)   # [bs, len(GROUP1_CLASSES)]

            # Step 3: hard gating — commit to the predicted group branch
            preds = torch.zeros(bs, dtype=torch.long, device=device)
            for i in range(bs):
                if hard_group[i] == 0:
                    local = fine0_logits[i].argmax().item()
                    preds[i] = group0_to_orig[local]
                else:
                    local = fine1_logits[i].argmax().item()
                    preds[i] = group1_to_orig[local]

            labels_cpu = labels.cpu()
            preds_cpu  = preds.cpu()
            conf_mat += torch.bincount(
                labels_cpu * num_classes + preds_cpu,
                minlength=num_classes ** 2
            ).reshape(num_classes, num_classes)

    # --- Metrics ---
    group_acc    = 100.0 * group_correct / max(1, total)
    soft_total   = conf_mat.sum().item()
    soft_correct = conf_mat.diag().sum().item()
    soft_acc     = 100.0 * soft_correct / max(1, soft_total)
    soft_f1      = _compute_macro_f1(conf_mat)

    import time
    tp_h = torch.diag(conf_mat).float()
    fp_h = (conf_mat.sum(0) - torch.diag(conf_mat)).float()
    fn_h = (conf_mat.sum(1) - torch.diag(conf_mat)).float()
    prec_h = tp_h / (tp_h + fp_h + 1e-8)
    rec_h  = tp_h / (tp_h + fn_h + 1e-8)
    f1_h   = 2 * prec_h * rec_h / (prec_h + rec_h + 1e-8)

    # Inference timing
    sample_signals = next(iter(val_loader))[0].to(device)
    for m in [group_model, group0_model, group1_model]:
        m.eval()
        with torch.no_grad():
            for _ in range(5): m(sample_signals)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(50):
            group_model(sample_signals)
            group0_model(sample_signals)
            group1_model(sample_signals)
    inf_ms = (time.perf_counter() - t0) / 50 * 1000

    print("\n" + "=" * 62)
    print("HIERARCHICAL CNN — EVALUATION REPORT (hard gating)")
    print("=" * 62)
    print(f"  Group classifier accuracy : {group_acc:.2f}%")
    print(f"  Hard-gate accuracy        : {soft_acc:.2f}%")
    print(f"  Hard-gate macro-F1        : {soft_f1:.4f}")
    print(f"  Inference time            : {inf_ms:.2f} ms/batch (3 models)")
    print("-" * 62)
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Acc':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i, name in enumerate(all_classes):
        class_total   = conf_mat[i, :].sum().item()
        class_correct = conf_mat[i, i].item()
        acc_i = class_correct / max(1, class_total)
        marker = " <-- SAFETY" if name == "person" else ""
        print(f"  {name:<12} {prec_h[i].item():>10.4f} {rec_h[i].item():>10.4f} {f1_h[i].item():>10.4f} {acc_i:>10.4f}{marker}")
    print("=" * 62)
    _print_confusion_matrix(conf_mat, all_classes)


if __name__ == "__main__":
    evaluate_hierarchical()