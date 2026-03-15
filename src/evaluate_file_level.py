#!/usr/bin/env python3
"""
File-level evaluation script for FIUS ultrasonic classification.

This script evaluates the flat CNN model on a set of test files with known ground truth labels.
It computes file-level accuracy and per-class accuracy, saving detailed results to JSON and CSV.

Usage:
    python src/evaluate_file_level.py --test_list test_files.txt --ground_truth ground_truth.csv

Where test_files.txt contains one file path per line, and ground_truth.csv has format:
    filename,true_label
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import pandas as pd

from predict import predict_file_structured


def load_ground_truth(csv_path):
    """Load ground truth labels from CSV file with robust matching."""
    gt = {}
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    # Clean filename and force label to lowercase for matching
                    filename = os.path.basename(row[0].strip()).lower()
                    label = row[1].strip().lower()
                    gt[filename] = label
        return gt
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {csv_path}")
        return {}


def evaluate_file_level(test_files, ground_truth, allow_fft_fallback=False, energy_percentile=0.0):
    """
    Evaluate model on file-level predictions.

    Args:
        test_files: List of file paths to evaluate
        ground_truth: Dict mapping filename to true label
        allow_fft_fallback: Whether to compute FFT from ADC if missing
        energy_percentile: Energy filtering percentile

    Returns:
        List of result dictionaries
    """
    results = []

    print(f"Evaluating {len(test_files)} files...")

    for i, file_path in enumerate(test_files):
        filename = os.path.basename(file_path)
        true_label = ground_truth.get(filename)

        print(f"[{i+1}/{len(test_files)}] Processing {filename}...")

        try:
            # Get structured prediction results
            pred_result = predict_file_structured(
                file_path,
                allow_fft_fallback=allow_fft_fallback,
                energy_percentile=energy_percentile
            )

            # Add ground truth
            pred_result['true_label'] = true_label

            results.append(pred_result)

        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            # Add error result
            results.append({
                'file_name': filename,
                'true_label': true_label,
                'predicted_label': None,
                'prediction_status': 'ERROR',
                'error': str(e),
                'top1_label': None,
                'top1_prob': None,
                'top2_label': None,
                'top2_prob': None,
                'margin_percent': None,
                'pulses_total': None,
                'pulses_used': None,
                'inference_time_ms': None,
            })

    return results


def compute_metrics(results):
    """Compute file-level accuracy metrics."""
    # Filter out errors
    valid_results = [r for r in results if r['prediction_status'] != 'ERROR' and r['true_label']]

    if not valid_results:
        return {
            'file_accuracy': 0.0,
            'per_class_accuracy': {},
            'total_files': len(results),
            'valid_files': 0,
            'error_files': len([r for r in results if r['prediction_status'] == 'ERROR'])
        }

    # Overall accuracy
    correct = sum(1 for r in valid_results if r['predicted_label'] == r['true_label'])
    file_accuracy = correct / len(valid_results) if valid_results else 0.0

    # Per-class accuracy
    per_class = {}
    class_counts = {}
    for r in valid_results:
        true_label = r['true_label']
        class_counts[true_label] = class_counts.get(true_label, 0) + 1

    for r in valid_results:
        true_label = r['true_label']
        if r['predicted_label'] == true_label:
            per_class[true_label] = per_class.get(true_label, 0) + 1

    per_class_accuracy = {}
    for label, correct_count in per_class.items():
        total_count = class_counts[label]
        per_class_accuracy[label] = correct_count / total_count if total_count > 0 else 0.0

    return {
        'file_accuracy': file_accuracy,
        'per_class_accuracy': per_class_accuracy,
        'total_files': len(results),
        'valid_files': len(valid_results),
        'error_files': len(results) - len(valid_results)
    }


def save_results(results, metrics, output_dir):
    """Save results to JSON and CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results as JSON
    json_path = output_dir / "file_level_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'results': results,
            'metrics': metrics,
            'timestamp': time.time()
        }, f, indent=2)

    # Save as CSV
    csv_path = output_dir / "file_level_results.csv"
    if results:
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

    print(f"Results saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")


def print_summary(metrics):
    """Print evaluation summary."""
    print("\n" + "="*50)
    print("FILE-LEVEL EVALUATION SUMMARY")
    print("="*50)
    print(".2%")
    print(f"Valid files evaluated: {metrics['valid_files']}/{metrics['total_files']}")
    if metrics['error_files'] > 0:
        print(f"Files with errors: {metrics['error_files']}")

    print("\nPer-class file accuracy:")
    for label, acc in sorted(metrics['per_class_accuracy'].items()):
        print(".2%")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="File-level evaluation for FIUS classification")
    parser.add_argument('--test_list', required=True, help='Text file with one test file path per line')
    parser.add_argument('--ground_truth', required=True, help='CSV file with ground truth labels')
    parser.add_argument('--output_dir', default='results', help='Output directory for results')
    parser.add_argument('--allow_fft_fallback', action='store_true', help='Allow FFT computation')
    parser.add_argument('--energy_percentile', type=float, default=0.0, help='Energy filtering')

    args = parser.parse_args()

    # Load test files
    if not os.path.exists(args.test_list):
        print(f"Error: Test list not found at {args.test_list}")
        return

    with open(args.test_list, 'r') as f:
        test_files = [line.strip() for line in f if line.strip()]

    # Load ground truth
    ground_truth = load_ground_truth(args.ground_truth)

    # Run evaluation
    results = evaluate_file_level(
        test_files,
        ground_truth,
        allow_fft_fallback=args.allow_fft_fallback,
        energy_percentile=args.energy_percentile
    )

    # Compute metrics
    metrics = compute_metrics(results)

    # Save results
    save_results(results, metrics, args.output_dir)

    # --- FIX: CALL THE SUMMARY FUNCTION ---
    print_summary(metrics)

    # Print summary
def print_summary(metrics):
    """Print evaluation summary with corrected percentage formatting."""
    print("\n" + "="*50)
    print("FILE-LEVEL EVALUATION SUMMARY")
    print("="*50)
    print(f"Overall Accuracy: {metrics['file_accuracy']:.2%}")
    print(f"Valid files evaluated: {metrics['valid_files']}/{metrics['total_files']}")
    
    if metrics['error_files'] > 0:
        print(f"Files with errors: {metrics['error_files']}")

    print("\nPer-class file accuracy:")
    for label, acc in sorted(metrics['per_class_accuracy'].items()):
        print(f"  {label:10}: {acc:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()