#!/usr/bin/env python3
"""
Result table exporter for FIUS ultrasonic classification.

This script collects metrics from:
- Segment-level validation metrics (from training history or evaluation)
- File-level metrics (from file_level_results.json)
- Latency metrics (from latency_profile.json)

And exports to JSON, CSV, and Markdown formats.

Usage:
    python src/export_result_tables.py
"""

import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd


def load_json_file(filepath):
    """Load JSON file, return None if not found."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect_segment_metrics(results_dir):
    """Collect segment-level validation metrics."""
    # Try to load from evaluation results
    eval_json = results_dir / "evaluation_results.json"
    eval_data = load_json_file(eval_json)

    if eval_data:
        return {
            'accuracy': eval_data.get('val_accuracy', 0),
            'macro_f1': eval_data.get('val_macro_f1', 0),
            'loss': eval_data.get('val_loss', 0),
            'source': 'evaluation_results.json'
        }

    # Try to load from training history
    history_json = results_dir / "training_history.json"
    history_data = load_json_file(history_json)

    if history_data:
        # Get final epoch metrics
        train_loss = history_data.get('train_loss', [])
        val_loss = history_data.get('val_loss', [])
        val_acc = history_data.get('val_accuracy', [])
        val_f1 = history_data.get('val_macro_f1', [])

        if val_acc and val_f1:
            return {
                'accuracy': val_acc[-1],
                'macro_f1': val_f1[-1],
                'loss': val_loss[-1] if val_loss else 0,
                'epochs': len(val_acc),
                'source': 'training_history.json'
            }

    return {'source': 'none'}


def collect_file_metrics(results_dir):
    """Collect file-level evaluation metrics."""
    file_results_json = results_dir / "file_level_results.json"
    data = load_json_file(file_results_json)

    if not data:
        return {'source': 'none'}

    metrics = data.get('metrics', {})
    return {
        'file_accuracy': metrics.get('file_accuracy', 0),
        'valid_files': metrics.get('valid_files', 0),
        'total_files': metrics.get('total_files', 0),
        'error_files': metrics.get('error_files', 0),
        'per_class_accuracy': metrics.get('per_class_accuracy', {}),
        'source': 'file_level_results.json'
    }


def collect_latency_metrics(results_dir):
    """Collect latency profiling metrics."""
    latency_json = results_dir / "latency_profile.json"
    data = load_json_file(latency_json)

    if not data:
        return {'source': 'none'}

    averages = data.get('averages', {})
    return {
        'total_time_ms': averages.get('total_time_ms', 0),
        'preprocess_time_ms': averages.get('preprocess_time_ms', 0),
        'forward_time_ms': averages.get('forward_time_ms', 0),
        'aggregation_time_ms': averages.get('aggregation_time_ms', 0),
        'per_pulse_forward_ms': averages.get('per_pulse_forward_ms', 0),
        'num_files_profiled': data.get('num_files', 0),
        'source': 'latency_profile.json'
    }


def create_result_tables(results_dir):
    """Collect all metrics and create result tables."""
    segment_metrics = collect_segment_metrics(results_dir)
    file_metrics = collect_file_metrics(results_dir)
    latency_metrics = collect_latency_metrics(results_dir)

    # Create summary table
    summary_table = {
        'metric_type': [],
        'metric_name': [],
        'value': [],
        'unit': [],
        'source': []
    }

    # Add segment-level metrics
    if segment_metrics['source'] != 'none':
        summary_table['metric_type'].append('segment_level')
        summary_table['metric_name'].append('validation_accuracy')
        summary_table['value'].append(segment_metrics.get('accuracy', 0))
        summary_table['unit'].append('%')
        summary_table['source'].append(segment_metrics['source'])

        summary_table['metric_type'].append('segment_level')
        summary_table['metric_name'].append('macro_f1')
        summary_table['value'].append(segment_metrics.get('macro_f1', 0))
        summary_table['unit'].append('')
        summary_table['source'].append(segment_metrics['source'])

        summary_table['metric_type'].append('segment_level')
        summary_table['metric_name'].append('validation_loss')
        summary_table['value'].append(segment_metrics.get('loss', 0))
        summary_table['unit'].append('')
        summary_table['source'].append(segment_metrics['source'])

    # Add file-level metrics
    if file_metrics['source'] != 'none':
        summary_table['metric_type'].append('file_level')
        summary_table['metric_name'].append('file_accuracy')
        summary_table['value'].append(file_metrics.get('file_accuracy', 0))
        summary_table['unit'].append('%')
        summary_table['source'].append(file_metrics['source'])

        summary_table['metric_type'].append('file_level')
        summary_table['metric_name'].append('valid_files')
        summary_table['value'].append(file_metrics.get('valid_files', 0))
        summary_table['unit'].append('count')
        summary_table['source'].append(file_metrics['source'])

        # Add per-class accuracies
        for class_name, acc in file_metrics.get('per_class_accuracy', {}).items():
            summary_table['metric_type'].append('file_level_per_class')
            summary_table['metric_name'].append(f'{class_name}_accuracy')
            summary_table['value'].append(acc)
            summary_table['unit'].append('%')
            summary_table['source'].append(file_metrics['source'])

    # Add latency metrics
    if latency_metrics['source'] != 'none':
        summary_table['metric_type'].append('latency')
        summary_table['metric_name'].append('total_inference_time')
        summary_table['value'].append(latency_metrics.get('total_time_ms', 0))
        summary_table['unit'].append('ms')
        summary_table['source'].append(latency_metrics['source'])

        summary_table['metric_type'].append('latency')
        summary_table['metric_name'].append('preprocessing_time')
        summary_table['value'].append(latency_metrics.get('preprocess_time_ms', 0))
        summary_table['unit'].append('ms')
        summary_table['source'].append(latency_metrics['source'])

        summary_table['metric_type'].append('latency')
        summary_table['metric_name'].append('forward_pass_time')
        summary_table['value'].append(latency_metrics.get('forward_time_ms', 0))
        summary_table['unit'].append('ms')
        summary_table['source'].append(latency_metrics['source'])

        summary_table['metric_type'].append('latency')
        summary_table['metric_name'].append('aggregation_time')
        summary_table['value'].append(latency_metrics.get('aggregation_time_ms', 0))
        summary_table['unit'].append('ms')
        summary_table['source'].append(latency_metrics['source'])

        summary_table['metric_type'].append('latency')
        summary_table['metric_name'].append('per_pulse_forward_time')
        summary_table['value'].append(latency_metrics.get('per_pulse_forward_ms', 0))
        summary_table['unit'].append('ms')
        summary_table['source'].append(latency_metrics['source'])

    # Create final result structure
    result_tables = {
        'summary_table': summary_table,
        'segment_metrics': segment_metrics,
        'file_metrics': file_metrics,
        'latency_metrics': latency_metrics,
        'collection_timestamp': pd.Timestamp.now().isoformat()
    }

    return result_tables


def save_result_tables(result_tables, output_dir):
    """Save result tables to multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    json_path = output_dir / "final_result_tables.json"
    with open(json_path, 'w') as f:
        json.dump(result_tables, f, indent=2)

    # Save CSV
    csv_path = output_dir / "final_result_tables.csv"
    df = pd.DataFrame(result_tables['summary_table'])
    df.to_csv(csv_path, index=False)

    # Create Markdown table
    md_path = output_dir / "final_result_tables.md"
    with open(md_path, 'w') as f:
        f.write("# FIUS Classification - Final Result Tables\n\n")
        f.write(f"Generated: {result_tables['collection_timestamp']}\n\n")

        # Summary table
        f.write("## Summary Table\n\n")
        f.write("| Metric Type | Metric Name | Value | Unit | Source |\n")
        f.write("|-------------|-------------|-------|------|--------|\n")

        for i in range(len(df)):
            row = df.iloc[i]
            value_str = ".3f" if isinstance(row['value'], (int, float)) else str(row['value'])
            f.write(f"| {row['metric_type']} | {row['metric_name']} | {value_str} | {row['unit']} | {row['source']} |\n")

        f.write("\n")

        # Detailed sections
        if result_tables['segment_metrics']['source'] != 'none':
            f.write("## Segment-Level Metrics\n\n")
            seg = result_tables['segment_metrics']
            f.write(f"- Validation Accuracy: {seg.get('accuracy', 0):.2f}%\n")
            f.write(f"- Macro F1: {seg.get('macro_f1', 0):.4f}\n")
            f.write(f"- Validation Loss: {seg.get('loss', 0):.4f}\n")
            f.write(f"- Source: {seg['source']}\n\n")

        if result_tables['file_metrics']['source'] != 'none':
            f.write("## File-Level Metrics\n\n")
            file_m = result_tables['file_metrics']
            f.write(f"- File Accuracy: {file_m.get('file_accuracy', 0):.2f}%\n")
            f.write(f"- Valid Files: {file_m.get('valid_files', 0)}\n")
            f.write(f"- Total Files: {file_m.get('total_files', 0)}\n")
            f.write(f"- Error Files: {file_m.get('error_files', 0)}\n\n")

            if file_m.get('per_class_accuracy'):
                f.write("### Per-Class File Accuracy\n\n")
                f.write("| Class | Accuracy |\n")
                f.write("|-------|----------|\n")
                for cls, acc in sorted(file_m['per_class_accuracy'].items()):
                    f.write(f"| {cls} | {acc:.2f}% |\n")
                f.write("\n")

        if result_tables['latency_metrics']['source'] != 'none':
            f.write("## Latency Metrics\n\n")
            lat = result_tables['latency_metrics']
            f.write(f"- Total Inference Time: {lat.get('total_time_ms', 0):.2f} ms\n")
            f.write(f"- Preprocessing Time: {lat.get('preprocess_time_ms', 0):.2f} ms\n")
            f.write(f"- Forward Pass Time: {lat.get('forward_time_ms', 0):.2f} ms\n")
            f.write(f"- Aggregation Time: {lat.get('aggregation_time_ms', 0):.2f} ms\n")
            f.write(f"- Per-Pulse Forward Time: {lat.get('per_pulse_forward_ms', 0):.3f} ms\n")
            f.write(f"- Files Profiled: {lat.get('num_files_profiled', 0)}\n")
            f.write(f"- Source: {lat['source']}\n\n")

            # Check AIS requirement
            total_time = lat.get('total_time_ms', 0)
            if total_time < 10.0:
                f.write("✅ **Meets AIS <10ms latency requirement**\n\n")
            else:
                f.write("❌ **Exceeds AIS 10ms latency requirement**\n\n")

    print(f"Result tables saved to:")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Markdown: {md_path}")


def print_result_summary(result_tables):
    """Print a summary of collected results."""
    print("\n" + "="*50)
    print("RESULT TABLES EXPORT SUMMARY")
    print("="*50)

    # Segment metrics
    seg = result_tables['segment_metrics']
    if seg['source'] != 'none':
        print("Segment-Level Metrics:")
        print(".2%")
        print(".4f")
        print(f"  Source: {seg['source']}")
    else:
        print("Segment-Level Metrics: Not found")

    # File metrics
    file_m = result_tables['file_metrics']
    if file_m['source'] != 'none':
        print("File-Level Metrics:")
        print(".2%")
        print(f"  Valid files: {file_m.get('valid_files', 0)}")
        print(f"  Source: {file_m['source']}")
    else:
        print("File-Level Metrics: Not found")

    # Latency metrics
    lat = result_tables['latency_metrics']
    if lat['source'] != 'none':
        print("Latency Metrics:")
        print(".2f")
        print(f"  Files profiled: {lat.get('num_files_profiled', 0)}")
        print(f"  Source: {lat['source']}")

        if lat.get('total_time_ms', 0) < 10.0:
            print("  ✓ Meets AIS <10ms requirement")
        else:
            print("  ✗ Exceeds AIS 10ms requirement")
    else:
        print("Latency Metrics: Not found")

    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Export result tables for FIUS classification")
    parser.add_argument('--results_dir', default='results',
                       help='Directory containing result files')
    parser.add_argument('--output_dir', default='results',
                       help='Output directory for exported tables')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Collect result tables
    result_tables = create_result_tables(results_dir)

    # Save to multiple formats
    save_result_tables(result_tables, args.output_dir)

    # Print summary
    print_result_summary(result_tables)


if __name__ == "__main__":
    main()