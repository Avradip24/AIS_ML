import argparse
import time
from pathlib import Path

import joblib
import numpy as np

from baseline_models import _extract_file_features
from data_loader import load_config, infer_fft_file_path


def benchmark_baseline(adc_input_path, runs=100, model_path=None):
    cfg = load_config()
    input_size = int(cfg["dataset"]["input_size"])

    if model_path is None:
        model_path = str((Path(cfg["paths"]["model_output"]).resolve().parent / "baseline_best.joblib"))

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Baseline model not found: {model_path}")

    bundle = joblib.load(model_path_obj)
    model = bundle["model"]
    class_names = bundle.get("class_names", [c.lower() for c in cfg["dataset"]["classes"]])

    # Warm-up once (feature extraction + model inference).
    warm_feats, use_paired_fft, fft_path = _extract_file_features(adc_input_path, input_size, require_fft=False)
    _ = model.predict(warm_feats.reshape(1, -1))

    latencies_ms = []
    last_pred_idx = 0
    for _ in range(max(1, int(runs))):
        start = time.perf_counter()
        feats, use_paired_fft, fft_path = _extract_file_features(adc_input_path, input_size, require_fft=False)
        pred_idx = int(model.predict(feats.reshape(1, -1))[0])
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)
        last_pred_idx = pred_idx

    mean_ms = float(np.mean(latencies_ms))
    median_ms = float(np.median(latencies_ms))
    min_ms = float(np.min(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    std_ms = float(np.std(latencies_ms))

    print("=" * 46)
    print("BASELINE INFERENCE LATENCY BENCHMARK")
    print("=" * 46)
    print(f"ADC input        : {adc_input_path}")
    if use_paired_fft and fft_path:
        print(f"FFT file         : {fft_path}")
    else:
        print(f"FFT file         : NOT FOUND ({infer_fft_file_path(adc_input_path)})")
        print("FFT source       : Computed from ADC")
    print(f"Runs             : {runs}")
    print(f"Predicted class  : {class_names[last_pred_idx]}")
    print("-" * 46)
    print(f"Mean latency     : {mean_ms:.3f} ms")
    print(f"Median latency   : {median_ms:.3f} ms")
    print(f"Min latency      : {min_ms:.3f} ms")
    print(f"Max latency      : {max_ms:.3f} ms")
    print(f"Std latency      : {std_ms:.3f} ms")
    print("=" * 46)

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
    parser.add_argument("--runs", type=int, default=100, help="Number of repeated runs.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional path to baseline .joblib (default: models/baseline_best.joblib).",
    )
    args = parser.parse_args()
    benchmark_baseline(adc_input_path=args.input, runs=max(1, int(args.runs)), model_path=args.model)