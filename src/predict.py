import torch
import os
import numpy as np
from model import UltrasonicCNN 
from data_loader import load_config, process_file

def predict_single_file(file_path):
    config = load_config()
    classes = config['dataset']['classes']
    device = torch.device("cpu")

    model = UltrasonicCNN(num_classes=len(classes))
    model_path = config['paths']['model_output']
    
    if not os.path.exists(model_path):
        print("Train the model first!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    measurements = process_file(file_path)
    if measurements is None: return

    # Take the first window and Normalize it exactly like the dataset
    sample = measurements[0].copy()
    sample_min = sample.min(axis=1, keepdims=True)
    sample_max = sample.max(axis=1, keepdims=True)
    sample = (sample - sample_min) / (sample_max - sample_min + 1e-8)
    
    input_data = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(input_data)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    print(f"\nAI Prediction: {classes[predicted_idx.item()]} ({confidence.item()*100:.2f}%)")
    for i, cls in enumerate(classes):
        print(f"{cls:12}: {probs[0][i].item()*100:5.1f}%")

if __name__ == "__main__":
    test_file = "./data/raw/wall/adc_measurements/adc_wall_0_54m_.txt"
    predict_single_file(test_file)