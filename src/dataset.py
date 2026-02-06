import torch
from torch.utils.data import Dataset
import os
import numpy as np
from data_loader import process_file, load_config

class UltrasonicDataset(Dataset):
    def __init__(self, root_dir):
        self.config = load_config()
        self.classes = self.config['dataset']['classes']
        self.data = []
        self.labels = []

        print(f"Searching in: {root_dir}")

        for idx, label_name in enumerate(self.classes):
            # Matches your structure: raw/wall/adc_measurements
            class_path = os.path.join(root_dir, label_name.lower(), "adc_measurements")
            
            if not os.path.exists(class_path):
                print(f"Skipping {label_name}: {class_path} not found.")
                continue
            
            print(f"Loading files for {label_name}...")
            files = [f for f in os.listdir(class_path) if f.endswith(".txt")]
            
            for file in files:
                file_path = os.path.join(class_path, file)
                measurements = process_file(file_path) # Returns [50, 2048]
                
                if measurements is not None:
                    for m in measurements:
                        self.data.append(m)
                        self.labels.append(idx)
        
        print(f"Successfully loaded {len(self.data)} total samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert to Tensor [1, 2048] as required by Conv1D
        sample = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label