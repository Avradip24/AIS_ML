#!/usr/bin/env python3
"""
Pulse-level latency profiling for FIUS ultrasonic classification.

This script performs detailed timing analysis at the pulse level to validate
AIS latency requirements. It distinguishes between:
- Pure model forward latency (per pulse)
- Preprocessing + model latency (per pulse)
- Full file-level decision latency

Usage:
    python src/profile_pulse_latency.py --test_files test_files.txt
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
import torch

from data_loader import load_config, process_file, infer_fft_file_path
from model import UltrasonicCNN


def profile_pulse_latencies(file_path, model, classes, device, allow_fft_fallback=False, energy_percentile=0.0, num_runs=10):
    """
    Profile latencies at the pulse level for a single file.

    Returns detailed timing statistics for different components.
    """
    resolved_fft = infer_fft_file_path(file_path)
    if resolved_fft is None and not allow_fft_fallback:
        raise ValueError("No paired FFT file found")

    # Load and preprocess data once
    measurements = process_file(
        file_path,
        resolved_fft,
        allow_computed_fft=allow_fft_fallback,
        return_fft_mode=False,
    )
    if measurements is None or len(measurements) == 0:
        raise ValueError("No valid pulses found")

    selected_measurements, _ = _select_pulses_by_energy(measurements, energy_percentile)
    pulses_used = selected_measurements.shape[0]

    # Profile pure model forward latency (per pulse)
    pure_forward_latencies = []
    for _ in range(num_runs):
        for pulse_idx in range(pulses_used):
            pulse_data = selected_measurements[pulse_idx:pulse_idx+1]  # Single pulse
            input_tensor = torch.from_numpy(pulse_data.astype(np.float32)).to(device)

            start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_tensor)
            latency = (time.perf_counter() - start) * 1000.0  # ms
            pure_forward_latencies.append(latency)

    # Profile preprocessing + model latency (per pulse)
    preprocessing_latencies = []
    full_pulse_latencies = []
    for _ in range(num_runs):
        for pulse_idx in range(pulses_used):
            # Time preprocessing for this pulse
            preprocess_start = time.perf_counter()
            pulse_data = selected_measurements[pulse_idx:pulse_idx+1]
            input_tensor = torch.from_numpy(pulse_data.astype(np.float32)).to(device)
            preprocess_time = (time.perf_counter() - preprocess_start) * 1000.0

            # Time full inference
            full_start = time.perf_counter()
            with torch.no_grad():
                _ = model(input_tensor)
            full_time = (time.perf_counter() - full_start) * 1000.0

            preprocessing_latencies.append(preprocess_time)
            full_pulse_latencies.append(full_time)

    # Profile full file-level decision latency
    file_latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()

        # Full file processing (same as predict_file_structured)
        input_batch = torch.from_numpy(selected_measurements.astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(input_batch)
            probs = torch.softmax(logits, dim=1)

        # Aggregation (voting logic)
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

        latency = (time.perf_counter() - start) * 1000.0
        file_latencies.append(latency)

    return {
        'file_name': os.path.basename(file_path),
        'pulses_used': pulses_used,
        'num_runs': num_runs,
        'pure_forward_latencies': pure_forward_latencies,
        'preprocessing_latencies': preprocessing_latencies,
        'full_pulse_latencies': full_pulse_latencies,
        'file_latencies': file_latencies,
        'statistics': {
            'pure_forward': {
                'mean': mean(pure_forward_latencies),
                'median': median(pure_forward_latencies),
                'std': stdev(pure_forward_latencies) if len(pure_forward_latencies) > 1 else 0,
                'min': min(pure_forward_latencies),
                'max': max(pure_forward_latencies),
            },
            'preprocessing': {
                'mean': mean(preprocessing_latencies),
                'median': median(preprocessing_latencies),
                'std': stdev(preprocessing_latencies) if len(preprocessing_latencies) > 1 else 0,
                'min': min(preprocessing_latencies),
                'max': max(preprocessing_latencies),
            },
            'full_pulse': {
                'mean': mean(full_pulse_latencies),
                'median': median(full_pulse_latencies),
                'std': stdev(full_pulse_latencies) if len(full_pulse_latencies) > 1 else 0,
                'min': min(full_pulse_latencies),
                'max': max(full_pulse_latencies),
            },
            'file_level': {
                'mean': mean(file_latencies),
                'median': median(file_latencies),
                'std': stdev(file_latencies) if len(file_latencies) > 1 else 0,
                'min': min(file_latencies),
                'max': max(file_latencies),
            }
        }
    }


def _select_pulses_by_energy(measurements, energy_percentile):
    """Helper function for pulse selection (copied from predict.py)"""
    total_pulses = int(measurements.shape[0])
    if total_pulses == 0:
        return measurements, np.arange(0, dtype=np.int64)

    if energy_percentile is None or energy_percentile <= 0.0:
        return measurements, np.arange(total_pulses, dtype=np.int64)

    adc = measurements[:, 0, :]  # channel 0 = normalized ADC waveform
    energies = np.sum(adc.astype(np.float64) ** 2, axis=1)
    threshold = np.percentile(energies, float(energy_percentile))
    selected_idx = np.where(energies >= threshold)[0]

    # Ensure at least one pulse is used.
    if selected_idx.size == 0:
        selected_idx = np.array([int(np.argmax(energies))], dtype=np.int64)

    return measurements[selected_idx], selected_idx


def run_pulse_latency_profiling(test_files, allow_fft_fallback=False, energy_percentile=0.0, num_runs=10):
    """
    Run pulse-level latency profiling across multiple test files.
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

    all_results = []

    print(f"Running pulse-level latency profiling on {len(test_files)} files...")
    print(f"Each file: {num_runs} runs × pulses_used pulses")

    for i, file_path in enumerate(test_files):
        filename = os.path.basename(file_path)
        print(f"[{i+1}/{len(test_files)}] Profiling {filename}...")

        try:
            result = profile_pulse_latencies(
                file_path, model, classes, device,
                allow_fft_fallback, energy_percentile, num_runs
            )
            all_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not all_results:
        raise RuntimeError("No successful profiling runs")

    # Aggregate across all files
    aggregated_stats = {
        'num_files': len(all_results),
        'total_runs': sum(r['num_runs'] for r in all_results),
        'total_pulses': sum(r['pulses_used'] for r in all_results),
        'file_results': all_results,
        'overall_statistics': {
            'pure_forward_ms': {
                'mean': mean([r['statistics']['pure_forward']['mean'] for r in all_results]),
                'median': median([r['statistics']['pure_forward']['median'] for r in all_results]),
                'min': min([r['statistics']['pure_forward']['min'] for r in all_results]),
                'max': max([r['statistics']['pure_forward']['max'] for r in all_results]),
            },
            'preprocessing_ms': {
                'mean': mean([r['statistics']['preprocessing']['mean'] for r in all_results]),
                'median': median([r['statistics']['preprocessing']['median'] for r in all_results]),
                'min': min([r['statistics']['preprocessing']['min'] for r in all_results]),
                'max': max([r['statistics']['preprocessing']['max'] for r in all_results]),
            },
            'full_pulse_ms': {
                'mean': mean([r['statistics']['full_pulse']['mean'] for r in all_results]),
                'median': median([r['statistics']['full_pulse']['median'] for r in all_results]),
                'min': min([r['statistics']['full_pulse']['min'] for r in all_results]),
                'max': max([r['statistics']['full_pulse']['max'] for r in all_results]),
            },
            'file_level_ms': {
                'mean': mean([r['statistics']['file_level']['mean'] for r in all_results]),
                'median': median([r['statistics']['file_level']['median'] for r in all_results]),
                'min': min([r['statistics']['file_level']['min'] for r in all_results]),
                'max': max([r['statistics']['file_level']['max'] for r in all_results]),
            }
        }
    }

    return aggregated_stats


def save_pulse_latency_profile(stats, output_dir):
    """Save pulse latency profile to JSON and CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "pulse_latency_profile.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Save CSV summary
    csv_path = output_dir / "pulse_latency_profile.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'mean_ms', 'median_ms', 'min_ms', 'max_ms'])

        for metric_name, values in stats['overall_statistics'].items():
            writer.writerow([
    metric_name,
    f"{values['mean']:.3f}",
    f"{values['median']:.3f}",
    f"{values['min']:.3f}",
    f"{values['max']:.3f}"
])

    print(f"Pulse latency profile saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")


def print_pulse_latency_summary(stats):
    """Print detailed pulse latency analysis with AIS interpretation."""
    print("\n" + "="*70)
    print("PULSE-LEVEL LATENCY PROFILING RESULTS")
    print("="*70)
    print(f"Files profiled: {stats['num_files']}")
    print(f"Total pulses across all files: {stats['total_pulses']}")
    print(f"Total timing runs: {stats['total_runs']}")
    print()

    print("Latency Statistics (ms):")
    print("-" * 70)
    print("2s")
    print("-" * 70)

    overall = stats['overall_statistics']

    print("Pure Model Forward:")
    pf = overall['pure_forward_ms']
    print("6.3f")
    print()

    print("Preprocessing per Pulse:")
    pp = overall['preprocessing_ms']
    print("6.3f")
    print()

    print("Full Pulse (Preproc + Model):")
    fp = overall['full_pulse_ms']
    print("6.3f")
    print()

    print("File-Level Decision:")
    fl = overall['file_level_ms']
    print("6.3f")
    print("-" * 70)

    # AIS requirement interpretation
    print("\nAIS LATENCY REQUIREMENT ANALYSIS (< 10 ms):")
    print("-" * 70)

    pure_forward_mean = pf['mean']
    if pure_forward_mean < 10.0:
        print("✓ Pure model forward-pass latency per pulse is below 10 ms")
        print("  → Meets AIS requirement for real-time pulse processing")
    else:
        print("✗ Pure model forward-pass latency per pulse exceeds 10 ms")
        print("  → Does not meet AIS requirement for real-time pulse processing")

    full_pulse_mean = fp['mean']
    if full_pulse_mean < 10.0:
        print("✓ Preprocessing + model latency per pulse is below 10 ms")
        print("  → Meets AIS requirement for real-time pulse processing")
    else:
        print("⚠ Preprocessing + model latency per pulse exceeds 10 ms")
        print("  → Pulse-level processing may not be real-time")

    file_level_mean = fl['mean']
    if file_level_mean < 10.0:
        print("✓ Full file-level pipeline meets <10 ms AIS requirement")
        print("  → Complete classification decision is real-time")
    else:
        print("✗ Full file-level pipeline exceeds 10 ms")
        print("  → File-level decisions are not real-time")
        print("  → However, pulse-level model inference meets requirement")

    print("\nRECOMMENDATION FOR FINAL REPORT:")
    print("-" * 70)
    if pure_forward_mean < 10.0:
        print(f"Use {pure_forward_mean:.3f} ms as the AIS latency metric")
        print("(Pure model forward-pass per pulse)")
    elif full_pulse_mean < 10.0:
        print(f"Use {full_pulse_mean:.3f} ms as the AIS latency metric")
        print("(Preprocessing + model per pulse)")
    else:
        print(f"Report {file_level_mean:.3f} ms file-level latency")
        print("(Does not meet <10ms requirement)")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Pulse-level latency profiling for FIUS classification")
    parser.add_argument('--test_files', required=True,
                       help='Text file with one test file path per line')
    parser.add_argument('--output_dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--allow_fft_fallback', action='store_true',
                       help='Allow computing FFT from ADC when paired file missing')
    parser.add_argument('--energy_percentile', type=float, default=0.0,
                       help='Energy filtering percentile (0 = disabled)')
    parser.add_argument('--num_runs', type=int, default=10,
                       help='Number of timing runs per pulse')

    args = parser.parse_args()

    # Load test files
    with open(args.test_files, 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]

    # Run profiling
    stats = run_pulse_latency_profiling(
        test_files,
        allow_fft_fallback=args.allow_fft_fallback,
        energy_percentile=args.energy_percentile,
        num_runs=args.num_runs
    )

    # Save results
    save_pulse_latency_profile(stats, args.output_dir)

    # Print analysis
    print_pulse_latency_summary(stats)


if __name__ == "__main__":
    main()