import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random
from data_loader import load_config

class UltrasonicDataset(Dataset):
    def __init__(self, root_dir, transform=True):
        self.config = load_config()
        self.classes = [c.lower() for c in self.config['dataset']['classes']]
        self.transform = transform
        
        # INCREASED DOWNSAMPLING: Speed up training
        self.downsample_factor = 100 
        self.fixed_length = 8192  
        
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
                            data = np.load(os.path.join(root, f), mmap_mode='r')
                            for i in range(len(data)):
                                self.samples.append((os.path.join(root, f), idx, i))
                            break

        random.shuffle(self.samples)
        print(f"✅ Ready! {len(self.samples)} segments indexed.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, segment_idx = self.samples[idx]
        
        if self.cache_path != file_path:
            self.cache_data = np.load(file_path)
            self.cache_path = file_path
            
        # 1. Extract and Normalize raw signal
        signal = self.cache_data[segment_idx].flatten().astype(np.float32)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        
        if self.transform:
            noise = np.random.normal(0, 0.02, signal.shape).astype(np.float32)
            signal = signal + noise

        # 2. Windowing parameters
        window_size = 60
        stride = 12
        feature_length = 256
        
        texture = []
        envelope = []
        spectral_centroid = [] 

        # 3. Feature Extraction Loop
        freqs = np.arange(window_size // 2 + 1)

        for i in range(0, len(signal) - window_size, stride):
            window = signal[i:i+window_size]
            
            texture.append(np.std(window))
            envelope.append(np.mean(np.abs(window)))
            
            fft_vals = np.abs(np.fft.rfft(window))
            sum_fft = np.sum(fft_vals)
            centroid = np.sum(freqs * fft_vals) / (sum_fft + 1e-6)
            spectral_centroid.append(centroid)
        
        # 4. Log compression
        texture = np.log1p(np.array(texture, dtype=np.float32))
        envelope = np.log1p(np.array(envelope, dtype=np.float32))
        spectral_centroid = np.array(spectral_centroid, dtype=np.float32)

        # 5. NEW: Per-Channel Standardization
        # This forces the model to look at RELATIVE changes rather than absolute values
        texture = (texture - np.mean(texture)) / (np.std(texture) + 1e-6)
        envelope = (envelope - np.mean(envelope)) / (np.std(envelope) + 1e-6)
        spectral_centroid = (spectral_centroid - np.mean(spectral_centroid)) / (np.std(spectral_centroid) + 1e-6)

        # 6. Helper to ensure fixed length
        def pad_or_cut(arr):
            if len(arr) > feature_length:
                return arr[:feature_length]
            return np.pad(arr, (0, feature_length - len(arr)))

        # 7. Stack into 3-channel feature map
        combined = np.stack([
            pad_or_cut(texture), 
            pad_or_cut(envelope),
            pad_or_cut(spectral_centroid)
        ], axis=0)
        
        return torch.from_numpy(combined).float(), torch.tensor(label).long()