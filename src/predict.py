import argparse
import os
import time
import numpy as np
import torch

from model import UltrasonicCNN
from data_loader import load_config, process_file, infer_fft_file_path

def predict_file(file_path, fft_file_path=None, allow_fft_fallback=False):
    config = load_config()
    classes = config["dataset"]["classes"]
    device = torch.device("cpu")

    model = UltrasonicCNN(num_classes=len(classes)).to(device)
    model_path = config["paths"]["model_output"]
    if not os.path.exists(model_path):
        print("Model file not found. Please train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    resolved_fft = fft_file_path or infer_fft_file_path(file_path)
    if resolved_fft is None and not allow_fft_fallback:
        print("No paired FFT file found. Use --allow_fft_fallback to compute FFT from ADC.")
        return

    try:
        measurements, fft_mode = process_file(
            file_path,
            resolved_fft,
            allow_computed_fft=allow_fft_fallback,
            return_fft_mode=True,
        )
    except Exception as e:
        print(f"Prediction preprocessing failed: {e}")
        return

    input_batch = torch.from_numpy(measurements.astype(np.float32)).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.softmax(logits, dim=1)

    mean_probs = probs.mean(dim=0)
    confidence, predicted_idx = torch.max(mean_probs, dim=0)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    predicted_label = classes[int(predicted_idx.item())]

    print("\n" + "=" * 40)
    print("FIUS CLASSIFICATION RESULT")
    print("=" * 40)
    print(f"Input file      : {file_path}")
    if fft_mode == "Using FFT file":
        print("Using FFT file")
        print(f"FFT file        : {resolved_fft}")
    elif fft_mode == "Computed FFT from ADC":
        print("Computed FFT from ADC")
    else:
        print(fft_mode if fft_mode else "Unknown FFT mode")
    print(f"Pulses processed: {input_batch.shape[0]}")
    print(f"Predicted class : {predicted_label}")
    print(f"Confidence      : {confidence.item() * 100:.2f}%")
    print(f"Inference time  : {latency_ms:.2f} ms")
    print("-" * 40)

    mean_probs_np = mean_probs.detach().cpu().numpy()
    for i, cls in enumerate(classes):
        print(f"{cls:12}: {mean_probs_np[i] * 100:5.1f}%")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--fft", type=str, default=None, help="Optional explicit FFT txt file.")
    parser.add_argument(
        "--allow_fft_fallback",
        action="store_true",
        help="Allow computing FFT from ADC when paired FFT file is missing or invalid.",
    )
    args = parser.parse_args()
    predict_file(args.input, args.fft, args.allow_fft_fallback)
