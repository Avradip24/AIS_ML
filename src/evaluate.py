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
    # Latency must be tested on CPU to validate the AIS Real-time target 
    device = torch.device("cpu") 
    
    # 2. Load Dataset
    # Methodology: Evaluation across the large-scale dataset classes [cite: 31, 32]
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
    print(f"Targeting Latency: < 10ms | Classes: {config['dataset']['classes']}")
    
    with torch.no_grad():
        for signals, labels in test_loader:
            # AIS Loop Validation: Measure end-to-end latency 
            start_time = time.perf_counter()
            outputs = model(signals)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000) # Convert to ms
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # 4. Latency Results [cite: 47]
    avg_latency = np.mean(latencies)
    print(f"\nAverage Latency: {avg_latency:.2f} ms")
    if avg_latency < 10:
        print("✅ Goal Met: Latency is under 10ms for real-time safety!")
    else:
        print("⚠️ Warning: Latency exceeds the 10ms AIS target.")

    # 5. Metric Assessment: Classification Report 
    print("\nClassification Report:")
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=config['dataset']['classes']
    )
    print(report)

    # 6. Metric Assessment: Confusion Matrix [cite: 40, 46]
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
    
    # Save the plot as a required deliverable for the Project Report [cite: 43, 47]
    plt.savefig("models/confusion_matrix.png")
    print("\n✅ Confusion Matrix saved to models/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    evaluate_model()