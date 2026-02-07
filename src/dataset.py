import torch
from torch.utils.data import Dataset
import os
import numpy as np
from data_loader import load_config, process_file 

class UltrasonicDataset(Dataset):
    def __init__(self, root_dir):
        self.config = load_config()
        self.classes = self.config['dataset']['classes']
        self.data = []
        self.labels = []
        
        for idx, label_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, label_name.lower(), "adc_measurements")
            if not os.path.exists(class_path): continue
            
            files = [f for f in os.listdir(class_path) if f.endswith(".txt")]
            for file in files:
                measurements = process_file(os.path.join(class_path, file))
                if measurements is not None:
                    for m in measurements:
                        self.data.append(m)
                        self.labels.append(idx)
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].copy()
        
        # 1. Normalization (The "Secret Sauce")
        # Forces the AI to look at shape, not volume
        sample_min = sample.min(axis=1, keepdims=True)
        sample_max = sample.max(axis=1, keepdims=True)
        sample = (sample - sample_min) / (sample_max - sample_min + 1e-8)
        
        # 2. Random Time Shift (Small)
        shift = np.random.randint(-150, 150)
        sample = np.roll(sample, shift, axis=1)
        
        # 3. Very Subtle Noise
        noise = np.random.normal(0, 0.002, sample.shape).astype(np.float32)
        sample = sample + noise
        
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)