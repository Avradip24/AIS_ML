import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from collections import Counter
from data_loader import load_config

class UltrasonicDataset(Dataset):
    """
    Ultrasonic dataset with bigtable->wall merge.

    Wall and bigtable produce acoustically indistinguishable echo waveforms
    when placed at the same distance (both are large flat hard surfaces).
    Bigtable data is therefore merged into the wall class at load time.
    This reduces the class count from 6 to 5.

    Final classes (5): wall, person, chair, backpack, plant
    """

    # Merge map: folder matching key gets relabelled as value class
    MERGE_MAP = {"bigtable": "wall"}

    def __init__(self, root_dir, transform=False, preload=False, augment=False):
        self.config = load_config()

        # Build merged class list: remove merged-away classes
        raw_classes = [c.lower() for c in self.config["dataset"]["classes"]]
        merged_away = set(self.MERGE_MAP.keys())
        self.classes = [c for c in raw_classes if c not in merged_away]

        self.transform = transform
        self.augment = augment
        self.input_size = int(self.config["dataset"]["input_size"])
        self.preload = preload

        self.samples = []
        self.cache_path = None
        self.cache_data = None
        self.preloaded_data = {}

        print(f"Scanning Binary Data: {root_dir}")
        print(f"Merge active: {self.MERGE_MAP}")
        print(f"Final classes ({len(self.classes)}): {self.classes}")

        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".npy"):
                    path_lower = root.lower()

                    # Check merge map first
                    merged_class = None
                    for src, tgt in self.MERGE_MAP.items():
                        if src in path_lower:
                            merged_class = tgt
                            break

                    if merged_class is not None:
                        if merged_class in self.classes:
                            idx = self.classes.index(merged_class)
                            file_path = os.path.join(root, f)
                            data = np.load(file_path, mmap_mode="r")
                            for i in range(len(data)):
                                self.samples.append((file_path, idx, i))
                        continue

                    # Normal class matching
                    for idx, class_name in enumerate(self.classes):
                        if class_name in path_lower:
                            file_path = os.path.join(root, f)
                            data = np.load(file_path, mmap_mode="r")
                            for i in range(len(data)):
                                self.samples.append((file_path, idx, i))
                            break

        random.shuffle(self.samples)
        label_counts = Counter(s[1] for s in self.samples)
        print(f"Ready! {len(self.samples)} segments indexed.")
        for i, cls in enumerate(self.classes):
            print(f"  {cls:<12}: {label_counts.get(i, 0)} segments")

        if self.preload:
            unique_files = sorted({sample[0] for sample in self.samples})
            try:
                print(f"Preloading {len(unique_files)} files into memory...")
                for file_path in unique_files:
                    self.preloaded_data[file_path] = np.load(file_path).astype(np.float32)
                print("Preloading complete.")
            except MemoryError:
                print("Preloading failed due to memory limits. Falling back to on-demand loading.")
                self.preloaded_data = {}
                self.preload = False

    def __len__(self):
        return len(self.samples)

    def _normalize_and_energy(self, signal):
        # Max-abs normalization: identical to data_loader.py _normalize_and_energy.
        # Ensures training features are on the same scale as prediction features.
        # Previously used fixed_scale=1.0 (no normalization) which caused a
        # train/predict mismatch: raw ADC amplitudes [-1912, 1898] during training
        # vs normalized [-1, 1] values during prediction.
        signal = signal.astype(np.float32)
        max_val = np.max(np.abs(signal)) + 1e-6
        norm = signal / max_val
        energy = np.cumsum(np.abs(norm))
        energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-6)
        return norm.astype(np.float32), energy.astype(np.float32)

    def _shift_1d(self, arr, shift):
        """Shift a 1D array with zero padding (no circular wrap)."""
        if shift == 0:
            return arr
        if shift > 0:
            return np.concatenate([np.zeros(shift, dtype=arr.dtype), arr[:-shift]])
        return np.concatenate([arr[-shift:], np.zeros(-shift, dtype=arr.dtype)])

    def _augment_sample(self, features):
        """Apply light training-only augmentation to a sample.

        Augmentation applies to waveform channels (ADC + FFT). Energy channels
        are recomputed after waveform manipulations to stay consistent.
        """
        # features: [4, input_size]
        adc = features[0].copy()
        fft = features[2].copy()

        # 1) Amplitude scaling
        adc *= np.random.uniform(0.95, 1.05)
        fft *= np.random.uniform(0.95, 1.05)

        # 2) Additive noise
        adc += np.random.normal(0.0, np.random.uniform(0.005, 0.02), size=adc.shape).astype(np.float32)
        fft += np.random.normal(0.0, np.random.uniform(0.005, 0.02), size=fft.shape).astype(np.float32)

        # 3) Temporal shift (padding, no wrap)
        shift = int(np.random.randint(-16, 17))
        if shift != 0:
            adc = self._shift_1d(adc, shift)
            fft = self._shift_1d(fft, shift)

        # Recompute energy channels after waveform changes.
        adc_energy = np.cumsum(np.abs(adc))
        adc_energy = (adc_energy - np.mean(adc_energy)) / (np.std(adc_energy) + 1e-6)
        fft_energy = np.cumsum(np.abs(fft))
        fft_energy = (fft_energy - np.mean(fft_energy)) / (np.std(fft_energy) + 1e-6)

        out = features.copy()
        out[0] = adc
        out[1] = adc_energy
        out[2] = fft
        out[3] = fft_energy
        return out

    def _build_features_from_adc_fft(self, adc_raw, fft_raw=None):
        if fft_raw is None:
            fft_raw = np.abs(np.fft.fft(adc_raw)).astype(np.float32)
        adc_norm, adc_energy = self._normalize_and_energy(adc_raw)
        fft_norm, fft_energy = self._normalize_and_energy(fft_raw)
        return np.stack([adc_norm, adc_energy, fft_norm, fft_energy], axis=0).astype(np.float32)

    def __getitem__(self, idx):
        file_path, label, segment_idx = self.samples[idx]

        if self.preload and file_path in self.preloaded_data:
            current_data = self.preloaded_data[file_path]
        else:
            if self.cache_path != file_path:
                self.cache_data = np.load(file_path)
                self.cache_path = file_path
            current_data = self.cache_data

        sample = current_data[segment_idx].astype(np.float32)

        if sample.ndim != 2 or sample.shape[0] != 4:
            raise ValueError(
                f"Expected sample shape [4, input_size], got {sample.shape} from {file_path}. "
                "Regenerate binary data with paired ADC/FFT preprocessing."
            )

        features = sample.copy()
        if not self.transform:
            return torch.from_numpy(features).float(), torch.tensor(label).long()

        if self.augment:
            features = self._augment_sample(features)
        else:
            # Minimal noise augmentation when transform is enabled but --augment not used.
            adc_raw = features[0] + np.random.normal(0, 0.02, features[0].shape).astype(np.float32)
            fft_raw = features[2] + np.random.normal(0, 0.01, features[2].shape).astype(np.float32)
            features = self._build_features_from_adc_fft(adc_raw, fft_raw)

        return torch.from_numpy(features).float(), torch.tensor(label).long()