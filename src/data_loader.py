import numpy as np
import os
import yaml

# Move this outside so it only runs once per execution
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# Global config to save CPU time
GLOBAL_CONFIG = load_config()

def process_file(file_path):
    input_size = GLOBAL_CONFIG['dataset']['input_size'] 
    
    try:
        with open(file_path, 'r') as f:
            raw_words = f.read().split()
        
        numeric_data = []
        for word in raw_words:
            try:
                # The FIUS sends integers, but float conversion is safer for the 0.53 values
                numeric_data.append(float(word))
            except ValueError:
                continue
        
        # Start at 16 to skip the header
        data_array = np.array(numeric_data[16:], dtype=np.float32)
        num_samples = len(data_array) // input_size
        if num_samples == 0:
            return None

        # Reshape to [Samples, 2048]
        raw_measurements = data_array[:num_samples * input_size].reshape(num_samples, input_size)
        
        processed_samples = []
        
        for i in range(len(raw_measurements)):
            # FEATURE 1: Raw Oscillatory Signal
            raw_sig = raw_measurements[i]
            
            # FEATURE 2: Cumulative Energy (Reflection Power)
            # This is great for distinguishing BigTable/Wall from soft objects.
            energy = np.cumsum(np.abs(raw_sig))
            
            # --- STANDARDIZATION ---
            # Max-Abs scaling is often better for raw ultrasonic waves
            max_val = np.max(np.abs(raw_sig)) + 1e-6
            raw_sig = raw_sig / max_val
            
            # Standardize Energy
            energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-6)
            
            # Stack into 2 channels: [2, 2048]
            combined = np.stack([raw_sig, energy], axis=0)
            processed_samples.append(combined)
            
        return np.array(processed_samples)
        
    except Exception as e:
        print(f"Error in data_loader while processing {file_path}: {e}")
        return None