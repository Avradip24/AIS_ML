import torch
from torch.utils.data import Dataset
import os
import numpy as np
from data_loader import load_config, process_file 

class UltrasonicDataset(Dataset):
    def __init__(self, root_dir):
        self.config = load_config()
        self.classes = self.config['dataset']['classes']
        
        raw_data = []
        raw_labels = []
        
        print(f"📂 Pre-loading data into RAM (this might take a minute)...")
        
        for idx, label_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, label_name.lower(), "adc_measurements")
            if not os.path.exists(class_path): continue
            
            files = [f for f in os.listdir(class_path) if f.endswith(".txt")]
            for file in files:
                measurements = process_file(os.path.join(class_path, file))
                if measurements is not None:
                    for m in measurements:
                        # Ensure data type is float32
                        m = m.astype(np.float32)
                        
                        # --- NORMALIZATION ---
                        # Channel 0: Raw Signal (Max-Abs)
                        r_max = np.max(np.abs(m[0])) + 1e-8
                        m[0] /= r_max
                        
                        # Channel 1: Energy (Max scaling)
                        e_max = np.max(m[1]) + 1e-8
                        m[1] /= e_max
                        
                        raw_data.append(m)
                        raw_labels.append(idx)
        
        self.data = torch.from_numpy(np.array(raw_data)).float()
        self.labels = torch.tensor(raw_labels).long()
        
        print(f"✅ Loaded {len(self.data)} samples. Ready for high-speed training.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].clone() 
        label = self.labels[idx]
        
        # --- INCREASED DATA AUGMENTATION ---
        # Increased noise from 0.0005 to 0.005 to prevent overfitting
        # on specific "jagged" textures
        noise = torch.randn_like(sample) * 0.005
        sample += noise
        
        return sample, label