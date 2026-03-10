import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset import UltrasonicDataset
from data_loader import load_config


def extract_statistical_features(sig_np: np.ndarray) -> list[float]:
    raw = sig_np[0]
    energy = sig_np[1]

    features = [
        np.mean(raw),
        np.std(raw),
        np.max(raw),
        np.min(raw),
        np.percentile(raw, 75),
        np.percentile(raw, 25),
        np.sum(raw ** 2),

        np.mean(energy),
        np.std(energy),
        np.max(energy),
        np.min(energy),
        np.percentile(energy, 75),
        np.percentile(energy, 25),
        np.sum(energy ** 2),
    ]
    return [float(f) for f in features]


def run_classical_benchmark():
    config = load_config()
    dataset = UltrasonicDataset(config["paths"]["raw_dir"])

    X, y = [], []

    print("📊 Extracting statistical features for SVM baseline...")
    for i in range(len(dataset)):
        signal, label = dataset[i]
        sig_np = signal.numpy()

        features = extract_statistical_features(sig_np)
        X.append(features)
        y.append(label.item())

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(X_train, y_train)

    print("\n--- Classical ML Baseline (SVM) Results ---")
    print(
        classification_report(
            y_test,
            clf.predict(X_test),
            target_names=config["dataset"]["classes"],
            digits=4,
        )
    )


if __name__ == "__main__":
    run_classical_benchmark()