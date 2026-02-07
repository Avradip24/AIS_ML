import numpy as np
import os
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def process_file(file_path):
    config = load_config()
    input_size = config['dataset']['input_size'] 
    
    try:
        with open(file_path, 'r') as f:
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

        # Base measurements: [Samples, 2048]
        raw_measurements = data_array[:num_samples * input_size].reshape(num_samples, input_size)
        
        processed_samples = []
        
        for i in range(len(raw_measurements)):
            # FEATURE 1: Absolute Envelope (Shape)
            env = np.abs(raw_measurements[i])
            
            # FEATURE 2: Cumulative Energy (Size/Reflection Power)
            # This helps distinguish BigTable vs Plant
            energy = np.cumsum(env)
            
            # Standardization for both features
            env = (env - np.mean(env)) / (np.std(env) + 1e-6)
            energy = (energy - np.mean(energy)) / (np.std(energy) + 1e-6)
            
            # Stack into 2 channels: [2, 2048]
            combined = np.stack([env, energy], axis=0)
            processed_samples.append(combined)
            
        return np.array(processed_samples) # Returns [Samples, 2, 2048]
        
    except Exception as e:
        print(f"Error in data_loader while processing {file_path}: {e}")
        return None