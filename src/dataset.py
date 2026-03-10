import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from data_loader import load_config

class UltrasonicDataset(Dataset):
    def __init__(self, root_dir, transform=True, preload=False):
        self.config = load_config()
        self.classes = [c.lower() for c in self.config["dataset"]["classes"]]
        self.transform = transform
        self.input_size = int(self.config["dataset"]["input_size"])
        self.preload = preload

        self.samples = []
        self.cache_path = None
        self.cache_data = None
        self.preloaded_data = {}

        print(f"Scanning Binary Data: {root_dir}")
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".npy"):
                    path_lower = root.lower()
                    for idx, class_name in enumerate(self.classes):
                        if class_name in path_lower:
                            file_path = os.path.join(root, f)
                            data = np.load(file_path, mmap_mode="r")
                            for i in range(len(data)):
                                self.samples.append((file_path, idx, i))
                            break

        random.shuffle(self.samples)
        print(f"Ready! {len(self.samples)} segments indexed.")

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
        signal = signal.astype(np.float32)
        max_val = np.max(np.abs(signal)) + 1e-6
        norm = signal / max_val
        energy = np.cumsum(np.abs(norm))
        energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-6)
        return norm.astype(np.float32), energy.astype(np.float32)

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

        # Training data is expected to be pre-converted to 4 channels using paired ADC+FFT files.
        if sample.ndim != 2 or sample.shape[0] != 4:
            raise ValueError(
                f"Expected sample shape [4, input_size], got {sample.shape} from {file_path}. "
                "Regenerate binary data with paired ADC/FFT preprocessing."
            )

        features = sample.copy()
        if not self.transform:
            return torch.from_numpy(features).float(), torch.tensor(label).long()

        # Optional augmentation path (disabled by default in training).
        adc_raw = features[0] + np.random.normal(0, 0.02, features[0].shape).astype(np.float32)
        fft_raw = features[2] + np.random.normal(0, 0.01, features[2].shape).astype(np.float32)
        features = self._build_features_from_adc_fft(adc_raw, fft_raw)

        return torch.from_numpy(features).float(), torch.tensor(label).long()
