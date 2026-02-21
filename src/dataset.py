import torch
from torch.utils.data import Dataset
import os
import numpy as np
from data_loader import load_config

class UltrasonicDataset(Dataset):
    def __init__(self, root_dir, transform=True):
        self.config = load_config()
        self.classes = [c.lower() for c in self.config['dataset']['classes']]
        self.transform = transform
        
        # INCREASED DOWNSAMPLING: Speed up training by 10x
        self.downsample_factor = 100 
        self.fixed_length = 8192  # Plenty of resolution for 1D-CNN
        
        self.samples = []
        self.cache_path = None
        self.cache_data = None

        print(f"📂 Scanning Binary Data: {root_dir}")
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith(".npy"):
                    path_lower = root.lower()
                    for idx, class_name in enumerate(self.classes):
                        if class_name in path_lower:
                            # Load segment count without loading whole file
                            data = np.load(os.path.join(root, f), mmap_mode='r')
                            for i in range(len(data)):
                                self.samples.append((os.path.join(root, f), idx, i))
                            break

        import random
        random.shuffle(self.samples)
        print(f"✅ Ready! {len(self.samples)} segments loaded in memory.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, segment_idx = self.samples[idx]
        
        # Simple caching to avoid re-loading the .npy file for every segment
        if self.cache_path != file_path:
            self.cache_data = np.load(file_path)
            self.cache_path = file_path
            
        # 1. Extract and Normalize raw signal
        signal = self.cache_data[segment_idx].flatten().astype(np.float32)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        
        # 2. Windowing parameters
        window_size = 50
        stride = 25
        feature_length = 256
        
        texture = []
        envelope = []
        spectral_centroid = [] # New: Pitch/Frequency info

        # 3. Feature Extraction Loop
        for i in range(0, len(signal) - window_size, stride):
            window = signal[i:i+window_size]
            
            # Channel 0: Texture (Standard Deviation)
            texture.append(np.std(window))
            
            # Channel 1: Energy Envelope (Absolute Mean)
            envelope.append(np.mean(np.abs(window)))
            
            # Channel 2: Spectral Centroid (Frequency center of gravity)
            # This helps distinguish material types (e.g., cloth vs plastic)
            fft_vals = np.abs(np.fft.rfft(window))
            freqs = np.arange(len(fft_vals))
            sum_fft = np.sum(fft_vals)
            if sum_fft > 0:
                centroid = np.sum(freqs * fft_vals) / sum_fft
            else:
                centroid = 0
            spectral_centroid.append(centroid)

        # 4. Helper to ensure fixed length for CNN
        def pad_or_cut(arr):
            arr = np.array(arr, dtype=np.float32)
            if len(arr) > feature_length:
                return arr[:feature_length]
            return np.pad(arr, (0, feature_length - len(arr)))

        # 5. Stack into 3-channel feature map: shape (3, 256)
        combined = np.stack([
            pad_or_cut(texture), 
            pad_or_cut(envelope),
            pad_or_cut(spectral_centroid)
        ], axis=0)
        
        return torch.from_numpy(combined).float(), torch.tensor(label).long()