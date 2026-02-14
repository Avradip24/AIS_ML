import torch
import numpy as np
import os
from model import UltrasonicCNN 
from data_loader import load_config, process_file

def predict_with_advanced_features(file_path):
    # 1. Load Configuration
    config = load_config()
    classes = config['dataset']['classes']
    device = torch.device("cpu") # Optimized for real-time AIS loop validation

    # 2. Initialize Model with Multi-Task Heads
    model = UltrasonicCNN(num_classes=len(classes))
    model_path = config['paths']['model_output']
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}. Please train the model first!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Data Preprocessing (Normalization & Header Skipping)
    # Methodology: Segmenting relevant echo regions and applying max_abs scaling
    measurements = process_file(file_path)
    if measurements is None: 
        print("❌ Failed to process file.")
        return

    # Take the first pulse and prepare for inference
    sample = measurements[0].copy().astype(np.float32)
    
    # Matching Dataset Normalization exactly for High-Integrity Output
    r_max = np.max(np.abs(sample[0])) + 1e-8
    sample[0] /= r_max
    e_max = np.max(sample[1]) + 1e-8
    sample[1] /= e_max
    
    input_data = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

    # 4. Inference and Multi-Task Output
    with torch.no_grad():
        # Head 1: Classification | Head 2: Range Estimation
        class_logits, range_pred = model(input_data)
        
        # Uncertainty Estimation Logic (Safety Focus)
        probs = torch.nn.functional.softmax(class_logits, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)
        
        # Implementation of the "Caution" state for safety-relevant systems
        if confidence.item() < 0.70:
            final_label = "UNCERTAIN / CAUTION (Low Confidence)"
        else:
            final_label = classes[predicted_idx.item()]

    # 5. AIS Result Output
    print(f"\n" + "="*35)
    print(f"       AIS PREDICTION RESULTS")
    print("="*35)
    print(f"Object Type : {final_label}")
    print(f"Confidence  : {confidence.item()*100:.2f}%")
    print(f"Est. Range  : {range_pred.item():.2f} cm/units") # Range Estimation objective
    print("-" * 35)
    
    # Detailed Probability Breakdown for High-Integrity Output
    for i, cls in enumerate(classes):
        print(f"{cls:12}: {probs[0][i].item()*100:5.1f}%")
    print("="*35 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict object and range with uncertainty estimation")
    parser.add_argument("--input", type=str, required=True, help="Path to the .txt file to predict")
    
    args = parser.parse_args()
    predict_with_advanced_features(args.input)