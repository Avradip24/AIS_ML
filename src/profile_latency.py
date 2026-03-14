#!/usr/bin/env python3
"""
Latency profiling script for FIUS ultrasonic classification.

This script measures inference latency components:
- Preprocessing time (file loading, pulse selection)
- Model forward-pass time
- Aggregation time (voting, confidence calculation)
- Total time

It runs profiling over multiple test files and computes averages.

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

    # Time preprocessing
    preprocess_start = time.perf_counter()
    measurements, _ = process_file(
        file_path,
        resolved_fft,
        allow_computed_fft=allow_fft_fallback,
        return_fft_mode=False,
    )
    if measurements is None or len(measurements) == 0:
        raise ValueError("No valid pulses found")

    selected_measurements, _ = _select_pulses_by_energy(measurements, energy_percentile)
    input_batch = torch.from_numpy(selected_measurements.astype(np.float32)).to(device)
    preprocess_time = (time.perf_counter() - preprocess_start) * 1000.0

    # Time forward pass
    forward_start = time.perf_counter()
    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.softmax(logits, dim=1)
    forward_time = (time.perf_counter() - forward_start) * 1000.0

    # Time aggregation
    agg_start = time.perf_counter()

    pulse_pred_idx = torch.argmax(probs, dim=1)
    mean_probs = probs.mean(dim=0)

    # Confidence-weighted vote
    pulse_confidences = torch.max(probs, dim=1).values
    conf_vote_scores = torch.zeros(len(classes), dtype=torch.float32, device=probs.device)
    conf_vote_scores.scatter_add_(0, pulse_pred_idx, pulse_confidences)

    # Energy-confidence weighted vote
    adc_selected = selected_measurements[:, 0, :]
    pulse_energies_np = np.sum(adc_selected.astype(np.float64) ** 2, axis=1)
    pulse_energies_np = pulse_energies_np / (np.sum(pulse_energies_np) + 1e-12)
    pulse_energies = torch.from_numpy(pulse_energies_np.astype(np.float32)).to(probs.device)

    energy_conf_weights = pulse_energies * pulse_confidences
    energy_conf_vote_scores = torch.zeros(len(classes), dtype=torch.float32, device=probs.device)
    energy_conf_vote_scores.scatter_add_(0, pulse_pred_idx, energy_conf_weights)

    # Majority vote
    counts = torch.bincount(pulse_pred_idx, minlength=len(classes))
    max_count = torch.max(counts)
    tied = torch.where(counts == max_count)[0]
    if tied.numel() == 1:
        majority_predicted_idx = tied[0]
    else:
        tied_probs = mean_probs[tied]
        majority_predicted_idx = tied[torch.argmax(tied_probs)]

    # Top-2 calculation
    sorted_mean_idx = torch.argsort(mean_probs, descending=True)
    top1_idx = int(sorted_mean_idx[0].item())
    top2_idx = int(sorted_mean_idx[1].item()) if mean_probs.numel() > 1 else top1_idx

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
    """
    Profile latency across multiple files and runs.

    Args:
        test_files: List of file paths to profile
        allow_fft_fallback: Whether to compute FFT from ADC
        energy_percentile: Energy filtering percentile
        num_runs: Number of profiling runs per file

    Returns:
        Dictionary with timing statistics
    """
    config = load_config()
    device = torch.device("cpu")

    # Load model
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
                print(f"  Run {run+1} failed: {e}")
                continue

        if file_timings:
            # Average across runs for this file
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

    # Compute overall statistics
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


def save_latency_profile(stats, output_dir):
    """Save latency profile to JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "latency_profile.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Save CSV
    csv_path = output_dir / "latency_profile.csv"
    import pandas as pd
    df = pd.DataFrame(stats['timings'])
    df.to_csv(csv_path, index=False)

    print(f"Latency profile saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")


def print_latency_summary(stats):
    """Print formatted latency summary table."""
    print("\n" + "="*60)
    print("LATENCY PROFILING SUMMARY")
    print("="*60)
    print(f"Files profiled: {stats['num_files']}")
    print(f"Runs per file: {stats['num_runs_per_file']}")
    print()

    print("Average Latency (ms):")
    print("-" * 40)
    print("2.2f")
    print("2.2f")
    print("2.2f")
    print("2.2f")
    print("2.2f")
    print()

    print("Standard Deviation (ms):")
    print("-" * 40)
    print("2.2f")
    print("2.2f")
    print("2.2f")
    print("2.2f")
    print("2.2f")
    print()

    # Check AIS requirement
    total_avg = stats['averages']['total_time_ms']
    if total_avg < 10.0:
        print("✓ Meets AIS <10ms latency requirement")
    else:
        print("✗ Exceeds AIS 10ms latency requirement")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Latency profiling for FIUS classification")
    parser.add_argument('--test_files', required=True,
                       help='Text file with one test file path per line')
    parser.add_argument('--output_dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--allow_fft_fallback', action='store_true',
                       help='Allow computing FFT from ADC when paired file missing')
    parser.add_argument('--energy_percentile', type=float, default=0.0,
                       help='Energy filtering percentile (0 = disabled)')
    parser.add_argument('--num_runs', type=int, default=5,
                       help='Number of profiling runs per file')

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

    # Save results
    save_latency_profile(stats, args.output_dir)

    # Print summary
    print_latency_summary(stats)


if __name__ == "__main__":
    main()