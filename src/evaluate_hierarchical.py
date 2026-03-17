import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from collections import defaultdict
import random

from dataset import UltrasonicDataset
from model import UltrasonicHierarchicalSoftCNN
from data_loader import load_config

"""
Hierarchical CNN Evaluation Script (soft routing, single-model)
"""


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
        val_counts[label] += len(class_val)

    train_idx, val_idx = [], []
    for fp, indices in file_to_indices.items():
        if fp in train_files:
            train_idx.extend(indices)
        elif fp in val_files:
            val_idx.extend(indices)

    if not train_idx or not val_idx:
        raise ValueError("No recording split available")

    return train_idx, val_idx, train_counts, val_counts


def _print_confusion_matrix(conf_mat, class_names):
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = " " * 12 + " ".join([f"{c[:10]:>10}" for c in class_names])
    print(header)
    for i, name in enumerate(class_names):
        row = " ".join([f"{int(conf_mat[i, j]):>10}" for j in range(len(class_names))])
        print(f"{name[:10]:>10}  {row}")


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


def evaluate_hierarchical(grouping='preset_a'):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_ds = UltrasonicDataset(config["paths"]["raw_dir"], transform=False, preload=(device.type == "cpu"))
    all_classes = base_ds.classes
    print(f"Detected classes: {all_classes}")

    if len(all_classes) != 5:
        raise ValueError("Hierarchical evaluation requires exactly 5 merged classes.")

    ckpt_path = Path(__file__).resolve().parents[1] / "models" / "fius_hierarchical_soft_cnn.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train_hierarchical.py first.")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" not in ckpt or "metadata" not in ckpt:
        raise ValueError("Checkpoint format invalid. Expected keys: 'model_state', 'metadata'.")

    metadata = ckpt["metadata"]
    group0_names = metadata.get("group0")
    group1_names = metadata.get("group1")
    if not group0_names or not group1_names:
        raise ValueError("Checkpoint metadata missing group definitions.")

    print(f"Grouping preset: {metadata.get('grouping', grouping)}")
    print(f"group0: {group0_names}")
    print(f"group1: {group1_names}")

    group0_indices = [i for i, c in enumerate(all_classes) if c in group0_names]
    group1_indices = [i for i, c in enumerate(all_classes) if c in group1_names]

    assert len(group0_indices) > 0 and len(group1_indices) > 0

    split_path = Path(__file__).resolve().parents[1] / "results" / "split_indices_hierarchical.pth"
    if split_path.exists():
        split_data = torch.load(split_path)
        train_idx, val_idx = split_data.get("train_indices"), split_data.get("val_indices")
    else:
        selected = list(range(len(all_classes)))
        train_idx, val_idx, _, _ = _build_recording_split(base_ds.samples, selected, val_ratio=0.2, seed=42)

    val_ds = RemappedSubset(base_ds, val_idx, {i: i for i in range(len(all_classes))})
    val_loader = DataLoader(val_ds, batch_size=int(config["training"].get("batch_size", 8)), shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = UltrasonicHierarchicalSoftCNN(num_group0=len(group0_indices), num_group1=len(group1_indices)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    conf_mat = torch.zeros((len(all_classes), len(all_classes)), dtype=torch.long)
    total, group_correct = 0, 0

    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)
            group_logits, fine0_logits, fine1_logits = model(signals)

            group_probs = torch.softmax(group_logits, dim=1)
            fine0_probs = torch.softmax(fine0_logits, dim=1)
            fine1_probs = torch.softmax(fine1_logits, dim=1)

            final_probs = torch.zeros((labels.size(0), len(all_classes)), device=device)
            final_probs[:, group0_indices] = group_probs[:, 0:1] * fine0_probs
            final_probs[:, group1_indices] = group_probs[:, 1:2] * fine1_probs

            pred = final_probs.argmax(dim=1)

            # group accuracy
            true_group = torch.zeros(labels.size(0), dtype=torch.long, device=device)
            for i, lbl in enumerate(labels):
                true_group[i] = 0 if lbl.item() in group0_indices else 1
            group_pred = group_probs.argmax(dim=1)
            group_correct += (group_pred == true_group).sum().item()

            for t, p in zip(labels.cpu(), pred.cpu()):
                conf_mat[t, p] += 1
            total += labels.size(0)

    group_acc = 100.0 * group_correct / max(1, total)
    final_acc = 100.0 * conf_mat.diag().sum().item() / max(1, conf_mat.sum().item())
    final_macro_f1 = _compute_macro_f1(conf_mat)

    # per-class stats
    per_class = []
    for i, c in enumerate(all_classes):
        tp = conf_mat[i, i].item()
        fn = conf_mat[i, :].sum().item() - tp
        fp = conf_mat[:, i].sum().item() - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        acc = tp / max(1, conf_mat[i, :].sum().item())
        per_class.append({"class": c, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc})

    # output
    print("\nSOFT ROUTING HIERARCHICAL EVALUATION")
    print("Group accuracy: {:.2f}%".format(group_acc))
    print("Final accuracy: {:.2f}%".format(final_acc))
    print("Final macro-F1: {:.4f}".format(final_macro_f1))
    print("Group0 classes:", group0_names)
    print("Group1 classes:", group1_names)
    print("\nPer-class metrics:")
    for pc in per_class:
        print(f" {pc['class']:10} prec={pc['precision']:.4f} recall={pc['recall']:.4f} f1={pc['f1']:.4f} acc={pc['accuracy']:.4f}")
    _print_confusion_matrix(conf_mat, all_classes)

    out_path = Path(__file__).resolve().parents[1] / "results" / f"hierarchical_eval_{metadata.get('grouping','preset_a')}_soft.json"
    out_summary = {
        "grouping": metadata.get("grouping", grouping),
        "group0": group0_names,
        "group1": group1_names,
        "group_accuracy": group_acc,
        "final_accuracy": final_acc,
        "final_macro_f1": final_macro_f1,
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(out_summary, f, indent=2)
    print(f"Saved evaluation summary: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouping", type=str, default="preset_a", choices=["preset_a","preset_b","preset_c"], help="Grouping preset")
    args = parser.parse_args()
    evaluate_hierarchical(grouping=args.grouping)
