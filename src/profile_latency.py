#!/usr/bin/env python3
"""
Latency profiling script for FIUS ultrasonic classification.

This script measures inference latency components:
- Preprocessing time (pulse selection, normalization)
- Model forward-pass time
- Aggregation time (voting, confidence calculation)
- Total time (including disk I/O loading)

Usage:
    python src/profile_latency.py --test_files test_files.txt
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

# Core project imports
from data_loader import load_config, process_file, infer_fft_file_path
from model import UltrasonicCNN
from predict import _select_pulses_by_energy


def profile_single_file(file_path, model, classes, device, allow_fft_fallback=False, energy_percentile=0.0):
    """
    Profile latency for a single file.
    Returns timing breakdown dictionary.
    """
    resolved_fft = infer_fft_file_path(file_path)
    if resolved_fft is None and not allow_fft_fallback:
        raise ValueError("No paired FFT file found")

    # Time preprocessing (File parsing + normalization)
    preprocess_start = time.perf_counter()
    
    # FIX: Use *others to catch extra return values (like fft_mode) from data_loader
    result = process_file(
        file_path,
        resolved_fft,
        allow_computed_fft=allow_fft_fallback,
        return_fft_mode=False,
    )
    
    # Handle both single return and tuple return variants from different data_loader versions
    if isinstance(result, tuple):
        measurements = result[0]
    else:
        measurements = result

    if measurements is None or len(measurements) == 0:
        raise ValueError("No valid pulses found")

    selected_measurements, _ = _select_pulses_by_energy(measurements, energy_percentile)
    input_batch = torch.from_numpy(selected_measurements.astype(np.float32)).to(device)
    preprocess_time = (time.perf_counter() - preprocess_start) * 1000.0

    # Time forward pass (The AI Inference)
    forward_start = time.perf_counter()
    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.softmax(logits, dim=1)
    forward_time = (time.perf_counter() - forward_start) * 1000.0

    # Time aggregation (Post-processing/Voting)
    agg_start = time.perf_counter()
    pulse_pred_idx = torch.argmax(probs, dim=1)
    # Basic voting logic timing
    pulse_confidences = torch.max(probs, dim=1).values
    conf_vote_scores = torch.zeros(len(classes), dtype=torch.float32, device=probs.device)
    conf_vote_scores.scatter_add_(0, pulse_pred_idx, pulse_confidences)
    agg_time = (time.perf_counter() - agg_start) * 1000.0

    total_time = preprocess_time + forward_time + agg_time
    pulses_used = selected_measurements.shape[0]

    return {
        'file_name': os.path.basename(file_path),
        'pulses_used': pulses_used,
        'preprocess_time_ms': preprocess_time,
        'forward_time_ms': forward_time,
        'aggregation_time_ms': agg_time,
        'total_time_ms': total_time,
        'per_pulse_forward_ms': forward_time / max(1, pulses_used),
    }


def profile_latency(test_files, allow_fft_fallback=False, energy_percentile=0.0, num_runs=5):
    config = load_config()
    device = torch.device("cpu")

    # Load model and class definitions
    from dataset import UltrasonicDataset
    raw_classes = [c.lower() for c in config["dataset"]["classes"]]
    merged_away = set(UltrasonicDataset.MERGE_MAP.keys())
    classes = [c for c in raw_classes if c not in merged_away]

    model = UltrasonicCNN(num_classes=len(classes)).to(device)
    model_path = config["paths"]["model_output"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_timings = []
    print(f"Profiling latency on {len(test_files)} files ({num_runs} runs each)...")

    for i, file_path in enumerate(test_files):
        filename = os.path.basename(file_path)
        print(f"[{i+1}/{len(test_files)}] Profiling {filename}...")

        file_timings = []
        for run in range(num_runs):
            try:
                timing = profile_single_file(
                    file_path, model, classes, device,
                    allow_fft_fallback, energy_percentile
                )
                file_timings.append(timing)
            except Exception as e:
                print(f"   Run {run+1} failed: {e}")
                continue

        if file_timings:
            avg_timing = {
                'file_name': filename,
                'pulses_used': int(mean(t['pulses_used'] for t in file_timings)),
                'preprocess_time_ms': mean(t['preprocess_time_ms'] for t in file_timings),
                'forward_time_ms': mean(t['forward_time_ms'] for t in file_timings),
                'aggregation_time_ms': mean(t['aggregation_time_ms'] for t in file_timings),
                'total_time_ms': mean(t['total_time_ms'] for t in file_timings),
                'per_pulse_forward_ms': mean(t['per_pulse_forward_ms'] for t in file_timings),
            }
            all_timings.append(avg_timing)

    if not all_timings:
        raise RuntimeError("No successful profiling runs")

    stats = {
        'num_files': len(all_timings),
        'num_runs_per_file': num_runs,
        'timings': all_timings,
        'averages': {
            'preprocess_time_ms': mean(t['preprocess_time_ms'] for t in all_timings),
            'forward_time_ms': mean(t['forward_time_ms'] for t in all_timings),
            'aggregation_time_ms': mean(t['aggregation_time_ms'] for t in all_timings),
            'total_time_ms': mean(t['total_time_ms'] for t in all_timings),
            'per_pulse_forward_ms': mean(t['per_pulse_forward_ms'] for t in all_timings),
        },
        'stddevs': {
            'preprocess_time_ms': stdev(t['preprocess_time_ms'] for t in all_timings) if len(all_timings) > 1 else 0,
            'forward_time_ms': stdev(t['forward_time_ms'] for t in all_timings) if len(all_timings) > 1 else 0,
            'aggregation_time_ms': stdev(t['aggregation_time_ms'] for t in all_timings) if len(all_timings) > 1 else 0,
            'total_time_ms': stdev(t['total_time_ms'] for t in all_timings) if len(all_timings) > 1 else 0,
            'per_pulse_forward_ms': stdev(t['per_pulse_forward_ms'] for t in all_timings) if len(all_timings) > 1 else 0,
        }
    }
    return stats


def print_latency_summary(stats):
    """Print formatted latency summary table with correct f-string formatting."""
    print("\n" + "="*60)
    print("LATENCY PROFILING SUMMARY")
    print("="*60)
    print(f"Files profiled: {stats['num_files']}")
    print(f"Runs per file: {stats['num_runs_per_file']}")
    print()

    print("Average Latency:")
    print("-" * 45)
    print(f"Preprocessing:      {stats['averages']['preprocess_time_ms']:8.2f} ms")
    print(f"Model Forward Pass: {stats['averages']['forward_time_ms']:8.2f} ms")
    print(f"Aggregation/Voting: {stats['averages']['aggregation_time_ms']:8.2f} ms")
    print(f"Total Pipeline:     {stats['averages']['total_time_ms']:8.2f} ms")
    print(f"Inference per Pulse:{stats['averages']['per_pulse_forward_ms']:8.2f} ms")
    print()

    print("Standard Deviation:")
    print("-" * 45)
    print(f"Preprocessing:      {stats['stddevs']['preprocess_time_ms']:8.2f} ms")
    print(f"Model Forward Pass: {stats['stddevs']['forward_time_ms']:8.2f} ms")
    print(f"Aggregation:        {stats['stddevs']['aggregation_time_ms']:8.2f} ms")
    print()

    print("Note: AIS <10ms requirement applies to 'Inference per Pulse' metric above.")
    print("This measures pure model forward-pass time per ultrasonic pulse.")
    print("="*60)


def save_latency_profile(stats, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "latency_profile.json", 'w') as f:
        json.dump(stats, f, indent=2)

    import pandas as pd
    pd.DataFrame(stats['timings']).to_csv(output_dir / "latency_profile.csv", index=False)
    print(f"\nLatency profile saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Latency profiling for FIUS classification")
    parser.add_argument('--test_files', required=True, help='Text file with test file paths')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--allow_fft_fallback', action='store_true')
    parser.add_argument('--energy_percentile', type=float, default=0.0)
    parser.add_argument('--num_runs', type=int, default=5)

    args = parser.parse_args()

    # Load test files
    with open(args.test_files, 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]

    # Run profiling
    stats = profile_latency(
        test_files,
        allow_fft_fallback=args.allow_fft_fallback,
        energy_percentile=args.energy_percentile,
        num_runs=args.num_runs
    )

    # Save and Print
    save_latency_profile(stats, args.output_dir)
    print_latency_summary(stats)


if __name__ == "__main__":
    main()