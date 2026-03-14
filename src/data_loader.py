import numpy as np
from pathlib import Path
import yaml
import re


def load_config():
    """Load config relative to repository root, independent of current working directory."""
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths from config location to avoid cwd-dependent behavior.
    base_dir = cfg_path.parent
    for key in ("raw_dir", "processed_dir", "model_output"):
        path_val = config.get("paths", {}).get(key)
        if path_val:
            p = Path(path_val)
            config["paths"][key] = str((base_dir / p).resolve()) if not p.is_absolute() else str(p)

    return config


GLOBAL_CONFIG = load_config()


def _read_txt_pulses(file_path, input_size):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_words = f.read().split()

    numeric_data = []
    for word in raw_words:
        try:
            numeric_data.append(float(word))
        except ValueError:
            continue

    data_array = np.array(numeric_data[16:], dtype=np.float32)
    num_samples = len(data_array) // input_size
    if num_samples == 0:
        return None

    return data_array[: num_samples * input_size].reshape(num_samples, input_size)


def _is_float_token(token):
    try:
        float(token)
        return True
    except ValueError:
        return False


def _parse_fius_fft_file(fft_file_path):
    """
    Parse FIUS FFT export:
    - first line is frequency axis
    - subsequent lines contain metadata + trailing FFT magnitude bins
    Returns: np.ndarray [num_pulses, fft_bins]
    """
    with open(fft_file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"FFT file has insufficient rows: {fft_file_path}")

    version_re = re.compile(r"^V\d+(\.\d+)?$", re.IGNORECASE)
    rows = []
    max_bins = 0

    # Skip first line (frequency axis/header)
    for line in lines[1:]:
        tokens = line.split()
        if not tokens:
            continue

        version_idx = next((i for i, tok in enumerate(tokens) if version_re.match(tok)), None)
        if version_idx is not None:
            # After V0.2, FIUS rows typically keep 2 trailing metadata fields before magnitudes.
            start_idx = min(len(tokens), version_idx + 3)
        else:
            # Fallback: take trailing numeric run.
            j = len(tokens) - 1
            while j >= 0 and _is_float_token(tokens[j]):
                j -= 1
            start_idx = j + 1

        mags = [float(tok) for tok in tokens[start_idx:] if _is_float_token(tok)]
        if not mags:
            continue

        max_bins = max(max_bins, len(mags))
        rows.append(mags)

    if not rows:
        raise ValueError(f"No FFT magnitude rows parsed from file: {fft_file_path}")

    fft_array = np.zeros((len(rows), max_bins), dtype=np.float32)
    for i, row in enumerate(rows):
        fft_array[i, : len(row)] = np.array(row, dtype=np.float32)

    return fft_array


def _build_expected_fft_path(adc_file_path):
    adc_path = Path(adc_file_path)
    fft_name = adc_path.name.replace("adc_", "fft_", 1)
    fft_dir = Path(str(adc_path.parent).replace("adc_measurements", "fft_measurements"))
    return str(fft_dir / fft_name)


def infer_fft_file_path(file_path):
    p = Path(file_path)
    if p.name.lower().startswith("fft_"):
        return str(p) if p.exists() else None

    expected_fft = _build_expected_fft_path(file_path)
    return expected_fft if Path(expected_fft).exists() else None


def _normalize_and_energy(signal):
    norm = signal.astype(np.float32)
    max_val = np.max(np.abs(norm)) + 1e-6
    norm = norm / max_val
    energy = np.cumsum(np.abs(norm))
    energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-6)
    return norm.astype(np.float32), energy.astype(np.float32)


def _fit_to_input_size(arr, input_size):
    if arr.shape[0] < input_size:
        return np.pad(arr, (0, input_size - arr.shape[0]), mode="constant")
    if arr.shape[0] > input_size:
        return arr[:input_size]
    return arr


def process_file(file_path, fft_file_path=None, allow_computed_fft=False, return_fft_mode=False):
    input_size = int(GLOBAL_CONFIG["dataset"]["input_size"])

    try:
        adc_path = str(file_path)
        adc_measurements = _read_txt_pulses(adc_path, input_size)
        if adc_measurements is None:
            raise ValueError(f"ADC parser found no valid samples: {adc_path}")

        expected_fft_path = str(fft_file_path) if fft_file_path else _build_expected_fft_path(adc_path)
        fft_path = expected_fft_path if Path(expected_fft_path).exists() else None

        if fft_path:
            try:
                fft_measurements = _parse_fius_fft_file(fft_path)
                fft_mode = "Using FFT file"
            except Exception as fft_parse_error:
                if allow_computed_fft:
                    print(f"Warning: FFT parse failed ({fft_parse_error}). Falling back to ADC-derived FFT.")
                    fft_measurements = np.abs(np.fft.fft(adc_measurements, axis=1)).astype(np.float32)
                    fft_mode = "Computed FFT from ADC"
                else:
                    raise RuntimeError(
                        f"FFT parsing failed for {fft_path}. "
                        "Use --allow_fft_fallback for ADC-derived FFT."
                    ) from fft_parse_error
        else:
            if allow_computed_fft:
                fft_measurements = np.abs(np.fft.fft(adc_measurements, axis=1)).astype(np.float32)
                fft_mode = "Computed FFT from ADC"
            else:
                raise FileNotFoundError(
                    f"Missing paired FFT file. Expected path: {expected_fft_path} (ADC input: {adc_path})"
                )

        # Debug prints once per processed file.
        print(f"ADC file used: {adc_path}")
        print(f"ADC parsed shape (raw rows): {adc_measurements.shape}")
        if fft_mode == "Using FFT file":
            print(f"FFT file path found: {fft_path}")
        else:
            print("FFT fallback: computed FFT from ADC")
        print(f"FFT parsed shape: {fft_measurements.shape}")

        # Average every 8 ADC rows into one sample
        # _read_txt_pulses returns (N_rows, input_size) where N_rows = recordings * 8.
        # convert_data.py uses ROWS_PER_ADC_SAMPLE=8: one training sample = mean of 8 rows.
        # Prediction MUST do the same averaging or features mismatch the training distribution.
        ROWS_PER_ADC_SAMPLE = 8
        n_adc_samples = len(adc_measurements) // ROWS_PER_ADC_SAMPLE
        if n_adc_samples == 0:
            # Fallback: file has fewer than 8 rows — use what we have averaged together
            adc_averaged = adc_measurements.mean(axis=0, keepdims=True)
        else:
            adc_averaged = np.stack(
                [adc_measurements[i * ROWS_PER_ADC_SAMPLE:(i + 1) * ROWS_PER_ADC_SAMPLE].mean(axis=0)
                 for i in range(n_adc_samples)],
                axis=0,
            )
        print(f"ADC averaged shape (8-row mean, matches training): {adc_averaged.shape}")

        num_samples = min(len(adc_averaged), len(fft_measurements))
        if num_samples == 0:
            raise ValueError(f"No overlapping ADC/FFT samples for file: {adc_path}")

        processed_samples = []

        for i in range(num_samples):
            adc_raw = adc_averaged[i]
            fft_raw = _fit_to_input_size(fft_measurements[i], input_size)

            adc_norm, adc_energy = _normalize_and_energy(adc_raw)
            fft_norm, fft_energy = _normalize_and_energy(fft_raw)

            # 4 channels: ADC norm, ADC energy, FFT norm, FFT energy.
            combined = np.stack([adc_norm, adc_energy, fft_norm, fft_energy], axis=0).astype(np.float32)
            processed_samples.append(combined)

        processed = np.array(processed_samples, dtype=np.float32)
        if return_fft_mode:
            return processed, fft_mode
        return processed
    except Exception as e:
        raise RuntimeError(f"Error in data_loader while processing {file_path}: {e}") from e