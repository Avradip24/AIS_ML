import argparse
import os
import time
import numpy as np
import torch

from model import UltrasonicCNN
from data_loader import load_config, process_file, infer_fft_file_path

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


def predict_file_structured(file_path, fft_file_path=None, allow_fft_fallback=False, energy_percentile=0.0):
    """
    Predict object class for a single file and return structured results.

    Returns a dictionary with all prediction details for evaluation/reporting.
    """
    config = load_config()
    # Use UltrasonicDataset to get the merged class list (bigtable merged into wall)
    from dataset import UltrasonicDataset
    import os
    from pathlib import Path
    _binary_dir = str(Path(__file__).resolve().parents[1] / "data" / "binary")
    _ds = UltrasonicDataset.__new__(UltrasonicDataset)
    _ds.config = config
    raw_classes = [c.lower() for c in config["dataset"]["classes"]]
    merged_away = set(UltrasonicDataset.MERGE_MAP.keys())
    classes = [c for c in raw_classes if c not in merged_away]
    device = torch.device("cpu")

    model = UltrasonicCNN(num_classes=len(classes)).to(device)
    model_path = config["paths"]["model_output"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    resolved_fft = fft_file_path or infer_fft_file_path(file_path)
    if resolved_fft is None and not allow_fft_fallback:
        raise ValueError("No paired FFT file found. Use allow_fft_fallback=True to compute FFT from ADC.")

    try:
        measurements, fft_mode = process_file(
            file_path,
            resolved_fft,
            allow_computed_fft=allow_fft_fallback,
            return_fft_mode=True,
        )
    except Exception as e:
        raise RuntimeError(f"Prediction preprocessing failed: {e}")

    if measurements is None or len(measurements) == 0:
        raise ValueError("No valid pulses found in input file.")

    pulses_total = int(measurements.shape[0])
    selected_measurements, selected_idx = _select_pulses_by_energy(measurements, energy_percentile)
    pulses_used = int(selected_measurements.shape[0])

    input_batch = torch.from_numpy(selected_measurements.astype(np.float32)).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        logits = model(input_batch)
        probs = torch.softmax(logits, dim=1)

    pulse_pred_idx = torch.argmax(probs, dim=1)
    mean_probs = probs.mean(dim=0)
    mean_confidence, mean_predicted_idx = torch.max(mean_probs, dim=0)

    # Confidence-weighted vote.
    pulse_confidences = torch.max(probs, dim=1).values
    conf_vote_scores = torch.zeros(len(classes), dtype=torch.float32, device=probs.device)
    conf_vote_scores.scatter_add_(0, pulse_pred_idx, pulse_confidences)
    conf_vote_predicted_idx = torch.argmax(conf_vote_scores)
    conf_vote_total = torch.sum(conf_vote_scores).item()
    conf_vote_confidence = (
        conf_vote_scores[int(conf_vote_predicted_idx.item())].item() / max(1e-8, conf_vote_total)
    )

    # Energy-confidence weighted vote.
    adc_selected = selected_measurements[:, 0, :]  # normalized ADC waveform
    pulse_energies_np = np.sum(adc_selected.astype(np.float64) ** 2, axis=1)
    pulse_energies_np = pulse_energies_np / (np.sum(pulse_energies_np) + 1e-12)
    pulse_energies = torch.from_numpy(pulse_energies_np.astype(np.float32)).to(probs.device)

    energy_conf_weights = pulse_energies * pulse_confidences
    energy_conf_vote_scores = torch.zeros(len(classes), dtype=torch.float32, device=probs.device)
    energy_conf_vote_scores.scatter_add_(0, pulse_pred_idx, energy_conf_weights)
    energy_conf_vote_predicted_idx = torch.argmax(energy_conf_vote_scores)
    energy_conf_total = torch.sum(energy_conf_vote_scores).item()
    energy_conf_vote_confidence = (
        energy_conf_vote_scores[int(energy_conf_vote_predicted_idx.item())].item() / max(1e-8, energy_conf_total)
    )

    # Majority vote over pulse-level predicted labels.
    counts = torch.bincount(pulse_pred_idx, minlength=len(classes))
    max_count = torch.max(counts)
    tied = torch.where(counts == max_count)[0]
    if tied.numel() == 1:
        majority_predicted_idx = tied[0]
    else:
        # Tie-break using mean probability among tied classes.
        tied_probs = mean_probs[tied]
        majority_predicted_idx = tied[torch.argmax(tied_probs)]
    majority_confidence = counts[int(majority_predicted_idx.item())].item() / max(1, pulses_used)

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    mean_predicted_label = classes[int(mean_predicted_idx.item())]
    majority_predicted_label = classes[int(majority_predicted_idx.item())]
    conf_vote_predicted_label = classes[int(conf_vote_predicted_idx.item())]
    energy_conf_vote_predicted_label = classes[int(energy_conf_vote_predicted_idx.item())]

    # Top-2 mean probability reporting
    sorted_mean_idx = torch.argsort(mean_probs, descending=True)
    top1_idx = int(sorted_mean_idx[0].item())
    top2_idx = int(sorted_mean_idx[1].item()) if mean_probs.numel() > 1 else top1_idx
    top1_prob = float(mean_probs[top1_idx].item())
    top2_prob = float(mean_probs[top2_idx].item())
    margin = top1_prob - top2_prob

    # Final selection logic
    if (
        mean_predicted_label == majority_predicted_label
        and mean_predicted_label == conf_vote_predicted_label
    ):
        final_predicted_label = mean_predicted_label
        final_method = "consensus_mean_majority_confidence_weighted"
    else:
        final_predicted_label = conf_vote_predicted_label
        final_method = "confidence_weighted_fallback"

    # Uncertainty reporting
    status = "CONFIDENT"
    if top1_prob < 0.40 or margin < 0.08:
        status = "UNCERTAIN"

    # Return structured results
    return {
        "file_name": os.path.basename(file_path),
        "true_label": None,  # To be filled by evaluation script
        "predicted_label": final_predicted_label,
        "prediction_status": status,
        "top1_label": classes[top1_idx],
        "top1_prob": top1_prob,
        "top2_label": classes[top2_idx],
        "top2_prob": top2_prob,
        "margin_percent": margin * 100.0,
        "pulses_total": pulses_total,
        "pulses_used": pulses_used,
        "inference_time_ms": latency_ms,
        # Additional details for debugging/analysis
        "mean_predicted_label": mean_predicted_label,
        "mean_confidence": float(mean_confidence.item()),
        "majority_predicted_label": majority_predicted_label,
        "majority_confidence": majority_confidence,
        "conf_vote_predicted_label": conf_vote_predicted_label,
        "conf_vote_confidence": conf_vote_confidence,
        "energy_conf_vote_predicted_label": energy_conf_vote_predicted_label,
        "energy_conf_vote_confidence": energy_conf_vote_confidence,
        "final_method": final_method,
        "fft_mode": fft_mode,
        "energy_percentile": energy_percentile,
    }
def predict_file(file_path, fft_file_path=None, allow_fft_fallback=False, energy_percentile=0.0, demo_mode=False):
    """
    Predict object class for a single file and print formatted results.
    """
    try:
        results = predict_file_structured(file_path, fft_file_path, allow_fft_fallback, energy_percentile)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    if demo_mode:
        print(f"FINAL PREDICTION: {results['predicted_label']}")
        print(f"METHOD USED: {results['final_method']}")
        print(f"STATUS: {results['prediction_status']}")
        print(f"TOP1: {results['top1_label']} ({results['top1_prob'] * 100:.1f}%)")
        print(f"MARGIN: {results['margin_percent']:.1f}%")
        print(f"PULSES USED: {results['pulses_used']}/{results['pulses_total']}")
        print(f"INFERENCE TIME: {results['inference_time_ms']:.2f} ms")
        return

    print("\n" + "=" * 40)
    print("FIUS CLASSIFICATION RESULT")
    print("=" * 40)
    print(f"Input file      : {file_path}")
    if results['fft_mode'] == "Using FFT file":
        print("Using FFT file")
        print(f"FFT file        : {fft_file_path or 'auto-detected'}")
    elif results['fft_mode'] == "Computed FFT from ADC":
        print("Computed FFT from ADC")
    else:
        print(results['fft_mode'] if results['fft_mode'] else "Unknown FFT mode")
    print(f"Pulses total    : {results['pulses_total']}")
    print(f"Pulses used     : {results['pulses_used']}")
    if results['energy_percentile'] and results['energy_percentile'] > 0:
        print(f"Energy filter   : >= p{float(results['energy_percentile']):.1f}")
    else:
        print("Energy filter   : disabled")
    print(
        f"Mean-prob class : {results['mean_predicted_label']} "
        f"({results['mean_confidence'] * 100:.2f}%)"
    )
    print(
        f"Majority-vote   : {results['majority_predicted_label']} "
        f"({results['majority_confidence'] * 100:.2f}% of used pulses)"
    )
    print(
        f"Conf-weighted   : {results['conf_vote_predicted_label']} "
        f"({results['conf_vote_confidence'] * 100:.2f}% weighted vote share)"
    )
    print(
        f"Energy+Conf     : {results['energy_conf_vote_predicted_label']} "
        f"({results['energy_conf_vote_confidence'] * 100:.2f}% weighted vote share)"
    )
    print(f"Inference time  : {results['inference_time_ms']:.2f} ms")
    print("-" * 40)

    print("Mean probability (top-2):")
    print(f"  1) {results['top1_label']}: {results['top1_prob'] * 100:.1f}%")
    print(f"  2) {results['top2_label']}: {results['top2_prob'] * 100:.1f}%")
    print(f"  Margin      : {results['margin_percent']:.1f}%")
    print(f"Prediction status: {results['prediction_status']}")
    print("-" * 40)
    print(f"FINAL PREDICTION: {results['predicted_label']}")
    print(f"METHOD USED: {results['final_method']}")
    print(f"PULSES USED: {results['pulses_used']}/{results['pulses_total']}")
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
    predict_file(args.input, args.fft, args.allow_fft_fallback, args.energy_percentile, args.demo_mode)