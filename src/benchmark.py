import argparse
import statistics
import time

import numpy as np
import torch

from data_loader import load_config, infer_fft_file_path, process_file
from model import UltrasonicCNN


def benchmark_inference(adc_input_path, runs=100, fft_file_path=None, allow_fft_fallback=False):
    config = load_config()
    classes = config["dataset"]["classes"]
    device = torch.device("cpu")

    model = UltrasonicCNN(num_classes=5).to(device)
    model_path = config["paths"]["model_output"]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    resolved_fft = fft_file_path or infer_fft_file_path(adc_input_path)
    if resolved_fft is None and not allow_fft_fallback:
        raise FileNotFoundError(
            "No paired FFT file found. Pass --allow_fft_fallback to compute FFT from ADC."
        )

    measurements, fft_mode = process_file(
        adc_input_path,
        resolved_fft,
        allow_computed_fft=allow_fft_fallback,
        return_fft_mode=True,
    )
    input_batch = torch.from_numpy(measurements.astype(np.float32)).to(device)

    # Warm-up
    with torch.no_grad():
        _ = model(input_batch)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            logits = model(input_batch)
            _ = torch.softmax(logits, dim=1).mean(dim=0)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    mean_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    std_ms = float(np.std(latencies_ms))

    print("=" * 44)
    print("CNN INFERENCE LATENCY BENCHMARK")
    print("=" * 44)
    print(f"Input ADC file   : {adc_input_path}")
    if fft_mode == "Using FFT file":
        print(f"FFT file         : {resolved_fft}")
    else:
        print("FFT source       : Computed FFT from ADC")
    print(f"Runs             : {runs}")
    print(f"Pulses per run   : {input_batch.shape[0]}")
    print("-" * 44)
    print(f"Mean latency     : {mean_ms:.3f} ms")
    print(f"Median latency   : {median_ms:.3f} ms")
    print(f"Min latency      : {min_ms:.3f} ms")
    print(f"Max latency      : {max_ms:.3f} ms")
    print(f"Std latency      : {std_ms:.3f} ms")
    print("=" * 44)

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "runs": int(runs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to ADC txt file.")
    parser.add_argument("--runs", type=int, default=100, help="Number of repeated inference runs.")
    parser.add_argument("--fft", type=str, default=None, help="Optional explicit FFT txt file.")
    parser.add_argument(
        "--allow_fft_fallback",
        action="store_true",
        help="Allow computing FFT from ADC when paired FFT file is missing/invalid.",
    )
    args = parser.parse_args()
    benchmark_inference(
        adc_input_path=args.input,
        runs=max(1, int(args.runs)),
        fft_file_path=args.fft,
        allow_fft_fallback=args.allow_fft_fallback,
    )