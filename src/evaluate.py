import torch
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from model import UltrasonicCNN
from dataset import UltrasonicDataset
from data_loader import load_config

def evaluate_model():
    config = load_config()
    device = torch.device("cpu") # Latency must be tested on CPU for AIS Real-time
    
    # Load Dataset
    dataset = UltrasonicDataset(config['paths']['raw_dir'])
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Load Model
    model = UltrasonicCNN(num_classes=len(config['dataset']['classes']))
    model.load_state_dict(torch.load(config['paths']['model_output'], map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    latencies = []

    print("--- Starting AIS Evaluation Loop ---")
    with torch.no_grad():
        for signals, labels in test_loader:
            # Measure Latency
            start_time = time.perf_counter()
            outputs = model(signals)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000) # Convert to ms
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Results
    avg_latency = np.mean(latencies)
    print(f"\n Average Latency: {avg_latency:.2f} ms")
    if avg_latency < 10:
        print(" Goal Met: Latency is under 10ms!")
    else:
        print(" Warning: Latency exceeds AIS target of 10ms.")

    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=config['dataset']['classes']))

if __name__ == "__main__":
    evaluate_model()