import argparse
from pathlib import Path
import numpy as np
import joblib

from baseline_models import _extract_file_features
from data_loader import load_config, infer_fft_file_path


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def predict_baseline(adc_input_path, model_path=None):
    cfg = load_config()
    input_size = int(cfg["dataset"]["input_size"])

    if model_path is None:
        model_path = str((Path(cfg["paths"]["model_output"]).resolve().parent / "baseline_best.joblib"))

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Baseline model not found: {model_path}")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    class_names = bundle.get("class_names", [c.lower() for c in cfg["dataset"]["classes"]])

    feats, use_paired_fft, fft_path = _extract_file_features(adc_input_path, input_size, require_fft=False)
    X = feats.reshape(1, -1)

    pred_idx = int(model.predict(X)[0])
    pred_class = class_names[pred_idx]

    print("=" * 40)
    print("BASELINE PREDICTION")
    print("=" * 40)
    print(f"ADC input      : {adc_input_path}")
    if use_paired_fft and fft_path:
        print(f"FFT file       : {fft_path}")
    else:
        print(f"FFT file       : NOT FOUND ({infer_fft_file_path(adc_input_path)})")
        print("FFT source     : Computed from ADC")
    print(f"Predicted class: {pred_class}")

    # Prefer probabilities if the classifier supports them.
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        print("Per-class probabilities:")
        for i, cname in enumerate(class_names):
            print(f"{cname:<10}: {float(probs[i]):.4f}")
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X)
        if np.ndim(dec) == 1:
            # Binary decision function returns one score.
            scores = np.array([-dec[0], dec[0]], dtype=np.float64)
            # Align with available classes if needed.
            if len(class_names) != 2:
                out = np.zeros(len(class_names), dtype=np.float64)
                out[:2] = _softmax(scores)
                scores = out
            else:
                scores = _softmax(scores)
        else:
            scores = _softmax(dec[0])
        print("Per-class scores (softmax(decision_function)):")
        for i, cname in enumerate(class_names):
            val = float(scores[i]) if i < len(scores) else 0.0
            print(f"{cname:<10}: {val:.4f}")
    else:
        print("Per-class scores: not available for this model type.")

    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to ADC txt file.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional path to baseline .joblib (default: models/baseline_best.joblib).",
    )
    args = parser.parse_args()
    predict_baseline(args.input, args.model)