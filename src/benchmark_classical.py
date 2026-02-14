import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset import UltrasonicDataset
from data_loader import load_config

def run_classical_benchmark():
    config = load_config()
    dataset = UltrasonicDataset(config['paths']['raw_dir'])
    
    X, y = [], []

    print("📊 Extracting statistical features for SVM baseline...")
    for i in range(len(dataset)):
        signal, label = dataset[i]
        sig_np = signal.numpy()
        
        # Methodology: Extract statistical features (Mean, Std, Max) 
        features = [
            np.mean(sig_np[0]), np.std(sig_np[0]), np.max(sig_np[0]), # Raw Channel
            np.mean(sig_np[1]), np.std(sig_np[1]), np.max(sig_np[1])  # Energy Channel
        ]
        X.append(features)
        y.append(label.item())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Train SVM as requested in ML Objectives 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(X_train, y_train)
    
    print("\n--- Classical ML Baseline (SVM) Results ---")
    print(classification_report(y_test, clf.predict(X_test), target_names=config['dataset']['classes']))

if __name__ == "__main__":
    run_classical_benchmark()