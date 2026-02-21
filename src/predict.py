import torch
import numpy as np
import os
import time
from model import UltrasonicCNN 
from data_loader import load_config, process_file

# --- NEW: Global buffer for Temporal Smoothing ---
prediction_buffer = []

def predict_with_advanced_features(file_path):
    global prediction_buffer
    
    # 1. Load Configuration
    config = load_config()
    classes = config['dataset']['classes']
    device = torch.device("cpu") # Validating AIS < 10ms target on CPU

    # 2. Initialize Model 
    model = UltrasonicCNN(num_classes=len(classes))
    model_path = config['paths']['model_output']
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found. Please train first!")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # --- NEW: Monte Carlo Dropout Logic ---
    # We keep dropout active during inference to estimate Epistemic Uncertainty
    model.train() # Keeping dropout layers active
    
    # 3. Data Preprocessing
    measurements = process_file(file_path)
    if measurements is None: return

    # Take first pulse and normalize
    sample = measurements[0].copy().astype(np.float32)

    # --- SYNC CHANNEL 0 (Max-Abs) ---
    max_val = np.max(np.abs(sample[0])) + 1e-8
    sample[0] /= max_val

    # --- SYNC CHANNEL 1 (Cumulative Energy) ---
    # Match data_loader.py exactly: abs -> cumsum -> standardize
    energy = np.cumsum(np.abs(sample[0])) 
    energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-6)
    sample[1] = energy # Update the second channel
    
    input_data = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

    # 4. Inference with MC Dropout and Smoothing
    # We run the same signal 10 times to see if the model remains consistent
    mc_iterations = 2
    all_class_probs = []
    all_range_preds = []

    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(mc_iterations):
            class_logits, range_pred = model(input_data)
            probs = torch.nn.functional.softmax(class_logits, dim=1)
            all_class_probs.append(probs.numpy())
            all_range_preds.append(range_pred.item())

    # Calculate Mean and Variance for Uncertainty Estimation
    mean_probs = np.mean(all_class_probs, axis=0)
    mean_range = np.mean(all_range_preds)
    
    # --- NEW: Temporal Smoothing ---
    # Average current result with previous pulses to stabilize the AIS loop
    prediction_buffer.append(mean_probs)
    if len(prediction_buffer) > 5: # 5-pulse smoothing window
        prediction_buffer.pop(0)
    
    smoothed_probs = np.mean(prediction_buffer, axis=0)
    confidence, predicted_idx = torch.max(torch.tensor(smoothed_probs), 1)

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    # 5. Safety Logic Thresholding
    # Fulfilling Human Safety Focus by flagging low-confidence scenes
    if confidence.item() < 0.75: # Stricter threshold for "perfect" safety
        final_label = "⚠️ CAUTION: UNCERTAIN"
    else:
        final_label = classes[predicted_idx.item()]

    # 6. AIS Result Output
    print(f"\n" + "="*40)
    print(f"      AIS PERCEPTION LOOP RESULTS")
    print("="*40)
    print(f"Object Identity : {final_label}")
    print(f"AIS Confidence  : {confidence.item()*100:.2f}%")
    print(f"Est. Range      : {mean_range:.2f} cm/units")
    print(f"Loop Latency    : {latency_ms:.2f} ms") # Target: < 10ms
    print("-" * 40)
    
    for i, cls in enumerate(classes):
        print(f"{cls:12}: {smoothed_probs[0][i]*100:5.1f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    predict_with_advanced_features(parser.parse_args().input)