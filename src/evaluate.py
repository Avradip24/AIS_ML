import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from model import UltrasonicCNN
from dataset import UltrasonicDataset
from data_loader import load_config

def evaluate_model():
    # 1. Setup and Config
    config = load_config()
    device = torch.device("cpu") # Latency target must be validated on CPU 
    
    # 2. Load Dataset
    # Methodology: Evaluation across the required 5 perception classes [cite: 31, 32]
    dataset = UltrasonicDataset(config['paths']['raw_dir'])
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 3. Load Model
    model = UltrasonicCNN(num_classes=len(config['dataset']['classes']))
    model.load_state_dict(torch.load(config['paths']['model_output'], map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    latencies = []

    print("--- Starting AIS Evaluation Loop ---")
    print(f"Targeting Latency: < 10ms | Target Classes: {config['dataset']['classes']}")
    
    with torch.no_grad():
        for signals, labels in test_loader:
            # AIS Loop Validation: Measure end-to-end latency 
            start_time = time.perf_counter()
            
            # UNPACK TUPLE: class_logits is index 0, range_pred is index 1
            outputs, range_pred = model(signals) 
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
            
            # Now torch.max receives the classification Tensor correctly
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            
    # 4. AIS Loop Validation: Latency Profiling [cite: 41, 47]
    avg_latency = np.mean(latencies)
    print(f"\nAverage Latency: {avg_latency:.2f} ms")
    if avg_latency < 10:
        print("✅ Goal Met: Latency is under 10ms for real-time safety!")
    else:
        print("⚠️ Warning: Latency exceeds the 10ms target defined in exposé.")

    # 5. Metric Assessment: Classification Report [cite: 40]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=config['dataset']['classes']))

    # 6. Metric Assessment: Confusion Matrix [cite: 40]
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        xticklabels=config['dataset']['classes'], 
        yticklabels=config['dataset']['classes'],
        cmap='Blues'
    )
    plt.title("AIS Object Classification - Confusion Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix as a required deliverable [cite: 43, 46]
    plt.savefig("models/confusion_matrix.png")
    print("\n✅ Confusion Matrix saved to models/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model()