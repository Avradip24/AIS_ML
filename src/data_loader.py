import numpy as np
import os
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def process_file(file_path):
    config = load_config()
    input_size = config['dataset']['input_size']
    samples_per_file = config['dataset']['samples_per_file']
    
    # Load raw text data
    raw_data = np.genfromtxt(file_path)
    
    # Skip the 16-word header found in your Red Pitaya files
    data_payload = raw_data[16:]
    
    # Slice into [50, 2048]
    # We use 'reshape' to verify the file has the correct amount of data
    try:
        measurements = data_payload.reshape(samples_per_file, input_size)
        # Normalization: Scale data between -1 and 1 for the CNN
        max_val = np.max(np.abs(measurements))
        if max_val > 0:
            measurements = measurements / max_val
        return measurements
    except ValueError:
        print(f"Warning: File {file_path} does not match expected size. Skipping.")
        return None