import torch
import os
import numpy as np
from model import UltrasonicCNN 
import argparse
from data_loader import load_config, process_file

def predict_single_file(file_path):
    config = load_config()
    classes = config['dataset']['classes']
    device = torch.device("cpu")

    model = UltrasonicCNN(num_classes=len(classes))
    model_path = config['paths']['model_output']
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}. Train the model first!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # process_file returns [Samples, 2, 2048] with raw sig and energy
    measurements = process_file(file_path)
    if measurements is None: 
        print("❌ Failed to process file.")
        return

    # Take the first pulse and ensure it is float32
    sample = measurements[0].copy().astype(np.float32)
    
    # --- FIXED: MATCH THE DATASET NORMALIZATION EXACTLY ---
    # Channel 0: Raw Signal normalization (Max-Abs)
    r_max = np.max(np.abs(sample[0])) + 1e-8
    sample[0] /= r_max

    # Channel 1: Energy normalization (Max Scaling)
    # This was missing and caused the low-confidence 26% predictions
    e_max = np.max(sample[1]) + 1e-8
    sample[1] /= e_max
    # -------------------------------------------------------
    
    # Convert to tensor and add batch dimension [1, 2, 2048]
    input_data = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_data)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    print(f"\nAI Prediction: {classes[predicted_idx.item()]} ({confidence.item()*100:.2f}%)")
    print("-" * 30)
    for i, cls in enumerate(classes):
        print(f"{cls:12}: {probs[0][i].item()*100:5.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict object from ultrasonic data file")
    parser.add_argument("--input", type=str, required=True, help="Path to the .txt file to predict")
    
    args = parser.parse_args()
    predict_single_file(args.input)