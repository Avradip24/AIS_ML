import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import UltrasonicDataset
from data_loader import load_config


def extract_features(sig_np: np.ndarray) -> list[float]:
    """
    Extract rich features from all 4 channels: ADC norm, ADC energy, FFT norm, FFT energy.
    14 stats per channel + echo envelope + zero-crossing = 72 features total.
    """
    features = []

    for ch in range(4):
        x = sig_np[ch].astype(np.float64)
        n = len(x)

        # Basic statistics
        features += [
            np.mean(x),
            np.std(x),
            np.max(x),
            np.min(x),
            np.max(x) - np.min(x),           # range
            np.percentile(x, 25),
            np.percentile(x, 75),
            np.percentile(x, 75) - np.percentile(x, 25),  # IQR
            np.sum(x ** 2) / n,               # mean power
            np.mean(np.abs(x)),               # mean absolute
        ]

        # Skewness and kurtosis (manual, no scipy dependency)
        mu = np.mean(x)
        c = x - mu
        m2 = np.mean(c ** 2) + 1e-12
        m3 = np.mean(c ** 3)
        m4 = np.mean(c ** 4)
        features += [m3 / (m2 ** 1.5), m4 / (m2 ** 2) - 3.0]

        # Zero-crossing rate (good for discriminating smooth vs rough surfaces)
        zc = np.sum(np.abs(np.diff(np.sign(x)))) / (2 * n)
        features.append(zc)

        # Peak count (number of local maxima above mean — indicates echo complexity)
        above_mean = (x > np.mean(x)).astype(int)
        peak_count = np.sum(np.diff(above_mean) == 1) / n
        features.append(peak_count)

        # First echo position (index of absolute max, normalized)
        features.append(np.argmax(np.abs(x)) / n)

        # Energy in first/second half (echo timing profile)
        half = n // 2
        e1 = np.sum(x[:half] ** 2) + 1e-12
        e2 = np.sum(x[half:] ** 2) + 1e-12
        features.append(e1 / (e1 + e2))  # fraction of energy in first half

    return [float(f) for f in features]


def run_classical_benchmark():
    config = load_config()
    dataset = UltrasonicDataset(config["paths"]["binary_dir"] 
                                if "binary_dir" in config.get("paths", {}) 
                                else config["paths"].get("raw_dir", config["paths"].get("binary_dir")))

    # Try binary dir directly
    import os
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parents[1]
    binary_dir = cfg_path / "data" / "binary"
    if binary_dir.exists():
        dataset = UltrasonicDataset(str(binary_dir))

    X, y = [], []
    print("Extracting features for classical baselines...")
    for i in range(len(dataset)):
        signal, label = dataset[i]
        sig_np = signal.numpy()
        features = extract_features(sig_np)
        X.append(features)
        y.append(label.item())

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"Feature matrix: {X.shape}  ({X.shape[1]} features per sample)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    names    = dataset.classes
    models   = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ("Random Forest",       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ("SVM (RBF)",           SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ]

    print("\n" + "=" * 60)
    print("CLASSICAL BASELINE RESULTS")
    print("=" * 60)
    summary = []
    for name, clf in models:
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        f1  = f1_score(y_test, preds, average="macro")
        summary.append((name, acc, f1))
        print(f"\n--- {name} ---")
        print(classification_report(y_test, preds, target_names=names, digits=4, zero_division=0))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10} {'Macro-F1':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    for name, acc, f1 in summary:
        print(f"  {name:<25} {acc:>9.2f}%  {f1:>10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_classical_benchmark()