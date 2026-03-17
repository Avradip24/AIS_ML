import os
import time
import json
import numpy as np
import torch
import argparse
from pathlib import Path

from dataset import UltrasonicDataset
from model import UltrasonicHierarchicalSoftCNN
from data_loader import load_config, process_file, infer_fft_file_path

"""
Hierarchical CNN Prediction Script (soft routing, single-model)
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

    if selected_idx.size == 0:
        selected_idx = np.array([int(np.argmax(energies))], dtype=np.int64)

    return measurements[selected_idx], selected_idx


def predict_file_hierarchical(
    file_path,
    fft_file_path=None,
    allow_fft_fallback=False,
    energy_percentile=0.0,
    demo_mode=False,
    confidence_threshold=0.0,
):
    config = load_config()
    all_classes = UltrasonicDataset(config["paths"]["raw_dir"], transform=False, preload=False).classes
    assert len(all_classes) == 5, "Hierarchical predict expects 5 merged classes."

    ckpt_path = Path(__file__).resolve().parents[1] / "models" / "fius_hierarchical_soft_cnn.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train_hierarchical.py first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state" not in ckpt or "metadata" not in ckpt:
        raise ValueError("Invalid checkpoint format. Expected keys: 'model_state', 'metadata'.")

    metadata = ckpt["metadata"]
    group0_names = metadata.get("group0")
    group1_names = metadata.get("group1")
    if not group0_names or not group1_names:
        raise ValueError("Checkpoint metadata missing group definitions.")

    group0_indices = [i for i, c in enumerate(all_classes) if c in group0_names]
    group1_indices = [i for i, c in enumerate(all_classes) if c in group1_names]

    model = UltrasonicHierarchicalSoftCNN(num_group0=len(group0_indices), num_group1=len(group1_indices))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    resolved_fft = fft_file_path or infer_fft_file_path(file_path)
    if resolved_fft is None and not allow_fft_fallback:
        raise FileNotFoundError("No paired FFT file found.")

    measurements, fft_mode = process_file(
        file_path,
        resolved_fft,
        allow_computed_fft=allow_fft_fallback,
        return_fft_mode=True,
    )
    if measurements is None or len(measurements) == 0:
        raise ValueError("No valid pulses found in input file.")

    pulses_total = len(measurements)
    selected_measurements, _ = _select_pulses_by_energy(measurements, energy_percentile)
    pulses_used = len(selected_measurements)

    if pulses_used == 0:
        raise ValueError("No pulses selected after energy filtering.")

    input_batch = torch.from_numpy(selected_measurements.astype(np.float32))

    start_time = time.perf_counter()
    with torch.no_grad():
        group_logits, fine0_logits, fine1_logits = model(input_batch)
        group_probs = torch.softmax(group_logits, dim=1)
        fine0_probs = torch.softmax(fine0_logits, dim=1)
        fine1_probs = torch.softmax(fine1_logits, dim=1)

        final_probs = torch.zeros((pulses_used, len(all_classes)))
        if group0_indices:
            final_probs[:, group0_indices] = group_probs[:, 0:1] * fine0_probs
        if group1_indices:
            final_probs[:, group1_indices] = group_probs[:, 1:2] * fine1_probs

        if confidence_threshold > 0.0:
            max_per_pulse = final_probs.max(dim=1).values
            keep_mask = max_per_pulse >= confidence_threshold
            kept = keep_mask.sum().item()
            min_keep = max(1, int(np.ceil(0.05 * pulses_used)))
            if kept < min_keep:
                keep_mask = torch.ones(pulses_used, dtype=torch.bool)
                kept = pulses_used
            else:
                print(f"Filtered pulses >= {confidence_threshold:.3f}: {kept}/{pulses_used}")
            final_probs = final_probs[keep_mask]
            pulses_used = kept

    mean_probs = final_probs.mean(dim=0)
    pulse_preds = final_probs.argmax(dim=1)
    mean_pred_idx = int(mean_probs.argmax().item())
    mean_pred_label = all_classes[mean_pred_idx]

    pulse_confidences = final_probs.max(dim=1).values
    conf_scores = torch.zeros(len(all_classes))
    conf_scores.scatter_add_(0, pulse_preds, pulse_confidences)
    conf_pred_idx = int(conf_scores.argmax().item())
    conf_pred_label = all_classes[conf_pred_idx]

    energies = np.sum(selected_measurements[:, 0, :].astype(np.float64) ** 2, axis=1)
    if energies.sum() <= 0:
        energies = np.ones_like(energies)
    energy_weights = torch.from_numpy((energies / energies.sum()).astype(np.float32))
    energy_conf_scores = torch.zeros(len(all_classes))
    energy_conf_scores.scatter_add_(0, pulse_preds, energy_weights.to(pulse_confidences.device) * pulse_confidences)
    energy_conf_pred_idx = int(energy_conf_scores.argmax().item())
    energy_conf_pred_label = all_classes[energy_conf_pred_idx]

    majority_counts = torch.bincount(pulse_preds, minlength=len(all_classes))
    best = torch.max(majority_counts).item()
    tied = torch.where(majority_counts == best)[0]
    if tied.numel() == 1:
        majority_pred_idx = int(tied[0].item())
    else:
        tied_probs = mean_probs[tied]
        majority_pred_idx = int(tied[torch.argmax(tied_probs)].item())
    majority_pred_label = all_classes[majority_pred_idx]

    top2 = torch.argsort(mean_probs, descending=True)[:2]
    top1, top2 = int(top2[0].item()), int(top2[1].item())
    top1_prob = float(mean_probs[top1].item())
    top2_prob = float(mean_probs[top2].item())
    margin = top1_prob - top2_prob
    status = "CONFIDENT" if (top1_prob >= 0.40 and margin >= 0.08) else "UNCERTAIN"

    # If mean and majority agree, use their consensus by default.
    # For UNCERTAIN cases, allow strong confidence-weighted vote to override.
    if mean_pred_label == majority_pred_label:
        if (
            status == "UNCERTAIN"
            and conf_pred_label != mean_pred_label
            and float(conf_scores[conf_pred_idx]) >= float(conf_scores[mean_pred_idx]) * 1.05
        ):
            final_label = conf_pred_label
            method = "confidence_weighted_fallback_on_uncertain"
        elif (
            status == "UNCERTAIN"
            and energy_conf_pred_label != mean_pred_label
            and float(energy_conf_scores[energy_conf_pred_idx]) >= 0.40
        ):
            final_label = energy_conf_pred_label
            method = "energy_confidence_fallback_on_uncertain"
        else:
            final_label = mean_pred_label
            method = "mean_majority_consensus"
    else:
        # mean and majority disagree: favor the confidence-weighted vote if strong, else energy fallback.
        if (
            conf_pred_label != mean_pred_label
            and float(conf_scores[conf_pred_idx]) >= float(conf_scores[mean_pred_idx]) * 1.05
        ):
            final_label = conf_pred_label
            method = "confidence_weighted_fallback_on_disagreement"
        else:
            final_label = energy_conf_pred_label
            method = "energy_confidence_weighted"

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    print("\n===============================================")
    print("HIERARCHICAL SOFT-ROUTING PREDICTION")
    print("===============================================")
    print(f"Input       : {file_path}")
    print(f"Grouping    : {metadata.get('grouping', 'preset_a')}")
    print(f"group0      : {group0_names}")
    print(f"group1      : {group1_names}")
    print(f"Pulses total: {pulses_total}")
    print(f"Pulses used : {pulses_used}")
    print(f"Mean        : {mean_pred_label} ({top1_prob*100:.2f}%)")
    print(f"Majority    : {majority_pred_label} ({(majority_counts[majority_pred_idx].item()/pulses_used*100):.2f}%)")
    print(f"Confidence  : {conf_pred_label} ({conf_scores[conf_pred_idx].item()*100:.2f}%)")
    print(f"EnergyConf  : {energy_conf_pred_label} ({energy_conf_scores[energy_conf_pred_idx].item()*100:.2f}%)")
    print(f"Final       : {final_label} ({method})")
    print(f"Status      : {status}")
    print(f"Latency     : {latency_ms:.2f} ms")
    print("-----------------------------------------------")
    print("Top-2:")
    print(f"  1) {all_classes[top1]} {top1_prob*100:.2f}%")
    print(f"  2) {all_classes[top2]} {top2_prob*100:.2f}%")
    print("===============================================\n")

    print("Class probability distribution:")
    mean_probs_np = mean_probs.detach().cpu().numpy()
    for i, cls in enumerate(all_classes):
        print(f"  {cls:10}: {mean_probs_np[i] * 100:5.1f}%")

    return {
        "input": str(file_path),
        "resolved_fft": str(resolved_fft) if resolved_fft is not None else None,
        "pulses_total": pulses_total,
        "pulses_used": pulses_used,
        "mean_pred": mean_pred_label,
        "majority_pred": majority_pred_label,
        "confidence_pred": conf_pred_label,
        "energy_conf_pred": energy_conf_pred_label,
        "final_pred": final_label,
        "method": method,
        "status": status,
        "latency_ms": latency_ms,
        "mean_prob": float(top1_prob),
        "second_prob": float(top2_prob),
    }


def batch_predict_hierarchical(
    path_label_pairs,
    allow_fft_fallback=False,
    energy_percentile=0.0,
    confidence_threshold=0.0,
):
    results = []
    for file_path, expected_label in path_label_pairs:
        try:
            result = predict_file_hierarchical(
                file_path,
                fft_file_path=None,
                allow_fft_fallback=allow_fft_fallback,
                energy_percentile=energy_percentile,
                demo_mode=False,
                confidence_threshold=confidence_threshold,
            )
            result["expected_label"] = expected_label
            result["match"] = (result["final_pred"] == expected_label) if expected_label else None
            results.append(result)
        except Exception as ex:
            results.append({
                "input": str(file_path),
                "expected_label": expected_label,
                "error": str(ex),
                "match": False,
            })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Single input ADC file path")
    parser.add_argument("--fft", type=str, default=None, help="Optional explicit FFT txt file.")
    parser.add_argument("--allow_fft_fallback", action="store_true", help="Allow computing FFT from ADC when paired FFT file is missing or invalid.")
    parser.add_argument("--energy_percentile", type=float, default=0.0, help="Optional low-energy filtering percentile (0 disables).")
    parser.add_argument("--demo_mode", action="store_true", help="Print concise demo output only.")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Minimum final-pulse confidence to keep pulses.")
    parser.add_argument("--batch_csv", type=str, default=None, help="Optional CSV file with columns path,expected_label for batch validation.")
    parser.add_argument("--expected_label", type=str, default=None, help="Optional expected label for single-file check.")
    args = parser.parse_args()

    if not args.batch_csv and not args.input:
        raise ValueError("Please provide --input for single prediction or --batch_csv for batch mode.")

    if args.batch_csv:
        pairs = []
        # Read CSV robustly: support UTF-8, UTF-16, or BOM-marked files.
        raw = None
        with open(args.batch_csv, "rb") as f:
            raw = f.read()

        decoded = None
        for enc in ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be"]:
            try:
                decoded = raw.decode(enc)
                break
            except Exception:
                continue

        if decoded is None:
            decoded = raw.decode("utf-8", errors="replace")

        lines = [l.strip() for l in decoded.splitlines() if l.strip() and not l.strip().startswith("#")]
        for line in lines:
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 2:
                pairs.append((cols[0], cols[1]))
            elif len(cols) == 1:
                pairs.append((cols[0], None))
        results = batch_predict_hierarchical(
            pairs,
            allow_fft_fallback=args.allow_fft_fallback,
            energy_percentile=args.energy_percentile,
            confidence_threshold=args.confidence_threshold,
        )
        output_file = Path("results") / "batch_predict_hierarchical.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        correct = sum(1 for r in results if r.get("match") is True)
        total = len(results)
        print(f"Batch prediction complete: {correct}/{total} matches. Report saved to {output_file}")
        exit(0)

    output = predict_file_hierarchical(
        args.input,
        args.fft,
        args.allow_fft_fallback,
        args.energy_percentile,
        args.demo_mode,
        args.confidence_threshold,
    )

    if args.expected_label is not None:
        match = output["final_pred"] == args.expected_label
        print(f"Expected: {args.expected_label} | Predicted: {output['final_pred']} | Match: {match}")
