import argparse
from pathlib import Path
import numpy as np
import joblib
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data_loader import (
    load_config,
    infer_fft_file_path,
    _read_txt_pulses,        # Reuse existing ADC parser.
    _parse_fius_fft_file,    # Reuse existing FIUS FFT parser.
)


def _safe_skew_kurt(x):
    x = x.astype(np.float64)
    mu = np.mean(x)
    c = x - mu
    m2 = np.mean(c ** 2) + 1e-12
    m3 = np.mean(c ** 3)
    m4 = np.mean(c ** 4)
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3.0
    return float(skew), float(kurt)


def _vector_features(x):
    x = x.astype(np.float64)
    ax = np.abs(x)
    energy = ax ** 2
    n = len(x)
    if n == 0:
        return [0.0] * 19

    mean = float(np.mean(x))
    std = float(np.std(x))
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    ptp = float(np.ptp(x))
    rms = float(np.sqrt(np.mean(x ** 2)))
    sig_energy = float(np.sum(energy))
    abs_mean = float(np.mean(ax))
    skew, kurt = _safe_skew_kurt(x)

    dom_idx = int(np.argmax(ax))
    dom_idx_norm = float(dom_idx / max(1, n - 1))
    dom_val = float(ax[dom_idx])

    bins = np.arange(n, dtype=np.float64)
    spectral_centroid = float(np.sum(bins * ax) / (np.sum(ax) + 1e-12))
    spectral_centroid_norm = float(spectral_centroid / max(1, n - 1))

    # Early/Mid/Late window energies (normalized).
    third = n // 3
    e_early = float(np.sum(energy[:third]))
    e_mid = float(np.sum(energy[third : 2 * third]))
    e_late = float(np.sum(energy[2 * third :]))
    e_total = e_early + e_mid + e_late + 1e-12
    e_early /= e_total
    e_mid /= e_total
    e_late /= e_total

    # Quartile energy positions (normalized indices where cumulative energy crosses q).
    cum_e = np.cumsum(energy)
    q25 = float(np.searchsorted(cum_e, 0.25 * cum_e[-1]) / max(1, n - 1))
    q50 = float(np.searchsorted(cum_e, 0.50 * cum_e[-1]) / max(1, n - 1))
    q75 = float(np.searchsorted(cum_e, 0.75 * cum_e[-1]) / max(1, n - 1))

    return [
        mean, std, min_v, max_v, ptp, rms, sig_energy, abs_mean,
        skew, kurt, dom_idx_norm, dom_val, spectral_centroid_norm,
        e_early, e_mid, e_late, q25, q50, q75,
    ]


def _aggregate_over_pulses(mat):
    # mat: [num_pulses, n]
    pulse_feats = np.array([_vector_features(row) for row in mat], dtype=np.float64)
    feat_mean = pulse_feats.mean(axis=0)
    feat_std = pulse_feats.std(axis=0)
    return np.concatenate([feat_mean, feat_std], axis=0)


def _fit_to_bins(mat, target_bins):
    # mat: [num_pulses, bins]
    if mat.shape[1] == target_bins:
        return mat
    if mat.shape[1] > target_bins:
        return mat[:, :target_bins]
    pad = np.zeros((mat.shape[0], target_bins - mat.shape[1]), dtype=mat.dtype)
    return np.concatenate([mat, pad], axis=1)


def _extract_file_features(adc_path, input_size, require_fft=False):
    adc = _read_txt_pulses(str(adc_path), input_size)
    if adc is None:
        raise ValueError(f"No valid ADC pulses parsed: {adc_path}")

    fft_path = infer_fft_file_path(str(adc_path))
    use_paired_fft = fft_path is not None

    if use_paired_fft:
        fft = _parse_fius_fft_file(fft_path)
        fft = _fit_to_bins(fft, input_size)
    else:
        if require_fft:
            raise FileNotFoundError(f"Missing paired FFT file for {adc_path}")
        fft = np.abs(np.fft.fft(adc, axis=1)).astype(np.float32)

    num_pulses = min(len(adc), len(fft))
    adc = adc[:num_pulses]
    fft = fft[:num_pulses]

    adc_feat = _aggregate_over_pulses(adc)
    fft_feat = _aggregate_over_pulses(fft)
    combined = np.concatenate([adc_feat, fft_feat], axis=0).astype(np.float32)
    return combined, bool(use_paired_fft), fft_path


def _collect_recordings(raw_root, all_class_names):
    """
    all_class_names: the full list of folder names (e.g. ['wall', 'person', 'chair', 'backpack', 'plant', 'bigtable'])
    """
    items = []
    
    # 1. Define the Merge Logic (Same as train.py)
    # We want to map 'bigtable' to 'wall'
    merge_map = {
        'bigtable': 'wall',
        'wall': 'wall',
        'person': 'person',
        'chair': 'chair',
        'backpack': 'backpack',
        'plant': 'plant'
    }
    
    # 2. Define the Unique Target Classes (The 5 classes we want to keep)
    # This must be in the exact same order as your CNN's output
    target_classes = ['wall', 'person', 'chair', 'backpack', 'plant']
    target_to_idx = {name: i for i, name in enumerate(target_classes)}

    for folder_name in all_class_names:
        adc_dir = Path(raw_root) / folder_name / "adc_measurements"
        if not adc_dir.exists():
            continue
            
        # Determine which target index this folder belongs to
        target_name = merge_map.get(folder_name, folder_name)
        if target_name not in target_to_idx:
            print(f"Skipping {folder_name}: not in target classes.")
            continue
            
        mapped_label = target_to_idx[target_name]
        
        for adc_file in sorted(adc_dir.glob("adc_*.txt")):
            items.append((str(adc_file), mapped_label))
            
    return items, target_classes


def _print_metrics(name, y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=np.arange(len(class_names)), average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Macro F1 : {macro:.4f}")
    print("Confusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("Per-class Accuracy:")
    row_sums = cm.sum(axis=1)
    for i, cname in enumerate(class_names):
        ca = (cm[i, i] / row_sums[i]) if row_sums[i] > 0 else 0.0
        print(f"{cname:<10}: {ca:.4f}")
    return acc, macro, cm


def _build_recording_split_indices(rec_paths, y, num_classes, val_ratio=0.2, seed=42):
    class_to_indices = defaultdict(list)
    for i, label in enumerate(y):
        class_to_indices[int(label)].append(i)

    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    train_counts = defaultdict(int)
    val_counts = defaultdict(int)

    for label in range(num_classes):
        indices = sorted(class_to_indices.get(label, []))
        if not indices:
            continue
        indices = list(indices)
        rng.shuffle(indices)

        if len(indices) == 1:
            class_val = []
            class_train = indices
        else:
            val_count = max(1, int(round(len(indices) * val_ratio)))
            val_count = min(val_count, len(indices) - 1)
            class_val = indices[:val_count]
            class_train = indices[val_count:]

        train_idx.extend(class_train)
        val_idx.extend(class_val)
        train_counts[label] += len(class_train)
        val_counts[label] += len(class_val)

    if not train_idx or not val_idx:
        raise RuntimeError("Recording-level split produced empty train or validation set.")

    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64), train_counts, val_counts


def _print_recording_counts(class_names, train_counts, val_counts):
    print("Recording Counts Per Class:")
    for i, cname in enumerate(class_names):
        print(f"{cname:<10} train_rec={train_counts.get(i, 0):<3} val_rec={val_counts.get(i, 0):<3}")


def _build_models(random_state):
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
    }


def run_baselines(raw_root=None, val_ratio=0.2, seeds=None, require_fft=False):
    cfg = load_config()
    all_folders = [c.lower() for c in cfg["dataset"]["classes"]]
    input_size = int(cfg["dataset"]["input_size"])
    if seeds is None or len(seeds) == 0:
        seeds = [42]

    if raw_root is None:
        raw_root = str((Path(__file__).resolve().parents[1] / "data" / "raw").resolve())

    # Get merged recordings and the new 5-class list
    recordings, class_names = _collect_recordings(raw_root, all_folders)
    
    if not recordings:
        raise RuntimeError(f"No ADC recordings found under {raw_root}")

    X, y, rec_paths = [], [], []
    paired_count = 0

    for adc_path, label in recordings:
        try:
            feats, paired, fft_path = _extract_file_features(adc_path, input_size, require_fft=require_fft)
            X.append(feats)
            y.append(label)
            rec_paths.append(adc_path)
            if paired:
                paired_count += 1
        except Exception as e:
            print(f"Skipping {adc_path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    rec_paths = np.array(rec_paths)

    if len(X) < 4:
        raise RuntimeError("Not enough recordings for train/validation baseline.")

    print(f"Total recordings used: {len(X)}")
    print(f"Effective Classes: {class_names}")
    print(f"Recordings with paired FFT: {paired_count}/{len(X)}")
    print(f"Feature dimension per recording: {X.shape[1]}")

    best_name = None
    best_macro = -1.0
    best_acc = -1.0
    best_obj = None
    metrics_by_model = {
        "logistic_regression": {"acc": [], "macro": []},
        "svm_rbf": {"acc": [], "macro": []},
        "random_forest": {"acc": [], "macro": []},
    }

    for seed in seeds:
        print(f"\n===== Seed {seed} =====")
        # Pass len(class_names) which is now 5
        train_idx, val_idx, train_counts, val_counts = _build_recording_split_indices(
            rec_paths, y, len(class_names), val_ratio=val_ratio, seed=seed
        )
        print(f"Train recordings: {len(train_idx)} | Val recordings: {len(val_idx)}")
        _print_recording_counts(class_names, train_counts, val_counts)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        models = _build_models(seed)
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            acc, macro, cm = _print_metrics(name, y_val, pred, class_names)
            metrics_by_model[name]["acc"].append(acc)
            metrics_by_model[name]["macro"].append(macro)

            if (macro > best_macro) or (abs(macro - best_macro) < 1e-12 and acc > best_acc):
                best_macro = macro
                best_acc = acc
                best_name = name
                best_obj = {
                    "model_name": name,
                    "seed": int(seed),
                    "model": model,
                    "class_names": class_names,
                    "input_size": input_size,
                    "feature_dim": X.shape[1],
                    "val_accuracy": float(acc),
                    "val_macro_f1": float(macro),
                }

    out_dir = Path(cfg["paths"]["model_output"]).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_best.joblib"
    joblib.dump(best_obj, out_path)

    print("\nBest Baseline (Merged Classes):")
    print(f"Model    : {best_name}")
    print(f"Accuracy : {best_acc:.4f}")
    print(f"Macro F1 : {best_macro:.4f}")

    print("\nAggregate Metrics Across Seeds (5 Classes):")
    for name in ["logistic_regression", "svm_rbf", "random_forest"]:
        acc_arr = np.array(metrics_by_model[name]["acc"], dtype=np.float64)
        f1_arr = np.array(metrics_by_model[name]["macro"], dtype=np.float64)
        print(
            f"{name:<20} "
            f"acc_mean={acc_arr.mean():.4f} acc_std={acc_arr.std():.4f} "
            f"f1_mean={f1_arr.mean():.4f} f1_std={f1_arr.std():.4f}"
        )

    return best_name, best_acc, best_macro, str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, default=None, help="Path to data/raw root.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds, e.g. 42,7,21,99,123")
    parser.add_argument(
        "--require_fft",
        action="store_true",
        help="Require paired FFT files for all recordings (otherwise missing FFT falls back to ADC-derived FFT).",
    )
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    run_baselines(
        raw_root=args.raw_root,
        val_ratio=args.val_ratio,
        seeds=seeds,
        require_fft=args.require_fft,
    )