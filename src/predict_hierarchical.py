import os
import time
import numpy as np
import torch
import torch.nn as nn
import argparse
from pathlib import Path

from dataset import UltrasonicDataset
from model import UltrasonicCNN
from data_loader import load_config, process_file, infer_fft_file_path

"""
Hierarchical CNN Prediction Script for FIUS Project

This script performs hierarchical classification on input ADC files.
It loads the 3 trained models and performs end-to-end hierarchical prediction.

Usage:
    python src/predict_hierarchical.py --input path/to/adc_file.txt --energy_percentile 30

Arguments:
    --input: Path to ADC input file (required)
    --fft: Optional explicit FFT file path
    --allow_fft_fallback: Allow computing FFT from ADC if paired file missing
    --energy_percentile: Energy percentile filter (0 disables, default: 0.0)
    --demo_mode: Print concise demo output only

Output:
    Hierarchical classification results with probabilities and final prediction.
    Uses energy-confidence-weighted vote as final prediction method.
"""

def _select_pulses_by_energy(measurements, energy_percentile):
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


def predict_file_hierarchical(file_path, fft_file_path=None, allow_fft_fallback=False, energy_percentile=0.0, demo_mode=False):
    config = load_config()
    # Use merged class list (bigtable merged into wall) — must match training
    from dataset import UltrasonicDataset
    raw_classes = [c.lower() for c in config["dataset"]["classes"]]
    merged_away = set(UltrasonicDataset.MERGE_MAP.keys())
    classes = [c for c in raw_classes if c not in merged_away]
    device = torch.device("cpu")

    # Acoustic grouping — must match train_hierarchical.py
    GROUP0_CLASSES = ["person", "backpack", "plant"]   # soft / absorbing
    GROUP1_CLASSES = ["wall", "chair"]                 # hard / reflective

    # Load models
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    group_model_path  = os.path.join(models_dir, "fius_group_cnn.pth")
    group0_model_path = os.path.join(models_dir, "fius_group0_cnn.pth")
    group1_model_path = os.path.join(models_dir, "fius_group1_cnn.pth")

    if not all(os.path.exists(p) for p in [group_model_path, group0_model_path, group1_model_path]):
        print("Missing trained models. Please run train_hierarchical.py first.")
        return

    group_model = UltrasonicCNN(num_classes=2).to(device)
    group_model.load_state_dict(torch.load(group_model_path, map_location=device))
    group_model.eval()

    group0_model = UltrasonicCNN(num_classes=len(GROUP0_CLASSES)).to(device)
    group0_model.load_state_dict(torch.load(group0_model_path, map_location=device))
    group0_model.eval()

    group1_model = UltrasonicCNN(num_classes=len(GROUP1_CLASSES)).to(device)
    group1_model.load_state_dict(torch.load(group1_model_path, map_location=device))
    group1_model.eval()

    # Define class mappings — derived from merged class list
    group0_classes = GROUP0_CLASSES
    group1_classes = GROUP1_CLASSES

    group0_indices = [i for i, name in enumerate(classes) if name in group0_classes]
    group1_indices = [i for i, name in enumerate(classes) if name in group1_classes]

    # Fine classifier to original class mapping
    group0_to_orig = {local: orig for local, orig in enumerate(group0_indices)}
    group1_to_orig = {local: orig for local, orig in enumerate(group1_indices)}

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

    if measurements is None or len(measurements) == 0:
        print("No valid pulses found in input file.")
        return

    pulses_total = int(measurements.shape[0])
    selected_measurements, selected_idx = _select_pulses_by_energy(measurements, energy_percentile)
    pulses_used = int(selected_measurements.shape[0])

    input_batch = torch.from_numpy(selected_measurements.astype(np.float32)).to(device)

    start_time = time.perf_counter()

    with torch.no_grad():
        # Step 1: Group classification - get probabilities for all pulses
        group_logits = group_model(input_batch)
        group_probs = torch.softmax(group_logits, dim=1)  # Shape: [pulses_used, 2]

        # Step 2: Run both fine classifiers on ALL pulses
        group0_logits = group0_model(input_batch)
        group0_fine_probs = torch.softmax(group0_logits, dim=1)  # Shape: [pulses_used, 3]

        group1_logits = group1_model(input_batch)
        group1_fine_probs = torch.softmax(group1_logits, dim=1)  # Shape: [pulses_used, 3]

        # Step 3: Combine probabilities into 6-class space
        final_probs = torch.zeros((pulses_used, 6), device=device)

        # For group0 classes: final_prob = P(group0) * P(fine_class|group0)
        group0_prob = group_probs[:, 0].unsqueeze(1)  # P(group0) for all pulses
        for local_idx, orig_idx in group0_to_orig.items():
            final_probs[:, orig_idx] = group0_prob.squeeze() * group0_fine_probs[:, local_idx]

        # For group1 classes: final_prob = P(group1) * P(fine_class|group1)
        group1_prob = group_probs[:, 1].unsqueeze(1)  # P(group1) for all pulses
        for local_idx, orig_idx in group1_to_orig.items():
            final_probs[:, orig_idx] = group1_prob.squeeze() * group1_fine_probs[:, local_idx]

    # Aggregate pulse predictions (same as original predict.py)
    mean_probs = final_probs.mean(dim=0)
    mean_confidence, mean_predicted_idx = torch.max(mean_probs, dim=0)

    # Confidence-weighted vote
    pulse_confidences = torch.max(final_probs, dim=1).values
    conf_vote_scores = torch.zeros(6, dtype=torch.float32, device=device)
    conf_vote_scores.scatter_add_(0, torch.argmax(final_probs, dim=1), pulse_confidences)
    conf_vote_predicted_idx = torch.argmax(conf_vote_scores)
    conf_vote_total = torch.sum(conf_vote_scores).item()
    conf_vote_confidence = (
        conf_vote_scores[int(conf_vote_predicted_idx.item())].item() / max(1e-8, conf_vote_total)
    )

    # Energy-confidence weighted vote
    adc_selected = selected_measurements[:, 0, :]
    pulse_energies_np = np.sum(adc_selected.astype(np.float64) ** 2, axis=1)
    pulse_energies_np = pulse_energies_np / (np.sum(pulse_energies_np) + 1e-12)
    pulse_energies = torch.from_numpy(pulse_energies_np.astype(np.float32)).to(device)

    energy_conf_weights = pulse_energies * pulse_confidences
    energy_conf_vote_scores = torch.zeros(6, dtype=torch.float32, device=device)
    energy_conf_vote_scores.scatter_add_(0, torch.argmax(final_probs, dim=1), energy_conf_weights)
    energy_conf_vote_predicted_idx = torch.argmax(energy_conf_vote_scores)
    energy_conf_total = torch.sum(energy_conf_vote_scores).item()
    energy_conf_vote_confidence = (
        energy_conf_vote_scores[int(energy_conf_vote_predicted_idx.item())].item() / max(1e-8, energy_conf_total)
    )

    # Majority vote
    pulse_pred_idx = torch.argmax(final_probs, dim=1)
    counts = torch.bincount(pulse_pred_idx, minlength=6)
    max_count = torch.max(counts)
    tied = torch.where(counts == max_count)[0]
    if tied.numel() == 1:
        majority_predicted_idx = tied[0]
    else:
        tied_probs = mean_probs[tied]
        majority_predicted_idx = tied[torch.argmax(tied_probs)]
    majority_confidence = counts[int(majority_predicted_idx.item())].item() / max(1, pulses_used)

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    mean_predicted_label = classes[int(mean_predicted_idx.item())]
    majority_predicted_label = classes[int(majority_predicted_idx.item())]
    conf_vote_predicted_label = classes[int(conf_vote_predicted_idx.item())]
    energy_conf_vote_predicted_label = classes[int(energy_conf_vote_predicted_idx.item())]

    final_predicted_label = energy_conf_vote_predicted_label
    final_method = "energy_confidence_weighted_vote"

    if demo_mode:
        print(f"Hierarchical final prediction: {final_predicted_label}")
        print(f"METHOD USED: {final_method}")
        print(f"CONFIDENCE: {energy_conf_vote_confidence * 100:.2f}%")
        print(f"PULSES USED: {pulses_used}/{pulses_total}")
        print(f"INFERENCE TIME: {latency_ms:.2f} ms")
        return

    print("\n" + "=" * 40)
    print("FIUS HIERARCHICAL CLASSIFICATION RESULT")
    print("=" * 40)
    print(f"Input file      : {file_path}")
    if fft_mode == "Using FFT file":
        print("Using FFT file")
        print(f"FFT file        : {resolved_fft}")
    elif fft_mode == "Computed FFT from ADC":
        print("Computed FFT from ADC")
    else:
        print(fft_mode if fft_mode else "Unknown FFT mode")
    print(f"Pulses total    : {pulses_total}")
    print(f"Pulses used     : {pulses_used}")
    if energy_percentile and energy_percentile > 0:
        print(f"Energy filter   : >= p{float(energy_percentile):.1f}")
    else:
        print("Energy filter   : disabled")
    print(
        f"Mean-prob class : {mean_predicted_label} "
        f"({mean_confidence.item() * 100:.2f}%)"
    )
    print(
        f"Majority-vote   : {majority_predicted_label} "
        f"({majority_confidence * 100:.2f}% of used pulses)"
    )
    print(
        f"Conf-weighted   : {conf_vote_predicted_label} "
        f"({conf_vote_confidence * 100:.2f}% weighted vote share)"
    )
    print(
        f"Energy+Conf     : {energy_conf_vote_predicted_label} "
        f"({energy_conf_vote_confidence * 100:.2f}% weighted vote share)"
    )
    print(f"Inference time  : {latency_ms:.2f} ms")
    print("-" * 40)

    mean_probs_np = mean_probs.detach().cpu().numpy()
    print("Class probability distribution:")
    for i, cls in enumerate(classes):
        print(f"{cls:12}: {mean_probs_np[i] * 100:5.1f}%")
    print("-" * 40)
    print(f"Hierarchical final prediction: {final_predicted_label}")
    print(f"METHOD USED: {final_method}")
    print(f"PULSES USED: {pulses_used}/{pulses_total}")
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
    parser.add_argument(
        "--energy_percentile",
        type=float,
        default=0.0,
        help="Optional low-energy filtering percentile (0 disables, e.g. 30 keeps pulses >= p30).",
    )
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Print concise live-demo output only.",
    )
    args = parser.parse_args()
    predict_file_hierarchical(args.input, args.fft, args.allow_fft_fallback, args.energy_percentile, args.demo_mode)