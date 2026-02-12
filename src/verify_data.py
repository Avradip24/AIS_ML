import os
import numpy as np

def run_dataset_audit(base_folder):
    # Constants based on your sensor configuration
    POINTS_PER_SAMPLE = 2048 
    
    print(f"\n{'OBJECT':<12} | {'FILE NAME':<35} | {'SAMPLES':<8} | {'STATUS'}")
    print("-" * 85)

    if not os.path.exists(base_folder):
        print(f"Error: Folder '{base_folder}' not found!")
        return

    total_samples = 0

    for root, dirs, files in os.walk(base_folder):
        # Skip FFT folders as we use Raw ADC for the 2D Spectrogram
        if "fft" in root.lower():
            continue
            
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                # Usually root is '.../backpack/adc_measurements', so label is 2 levels up
                parts = root.split(os.sep)
                label = parts[-2] if len(parts) > 1 else os.path.basename(root)
                
                try:
                    with open(path, 'r') as f:
                        content = f.read().split()
                    
                    # Instead of a fixed header size, we find the first 
                    # clearly numeric data point after the 'V0.2' string
                    # Or we just find all valid numbers and count them.
                    numeric_data = []
                    for word in content:
                        try:
                            # Skip strings like 'V0.2'
                            numeric_data.append(float(word))
                        except ValueError:
                            continue
                    
                    # The first ~12-15 numeric values are header (distance, freq, etc.)
                    # We subtract those to get the raw pulse data
                    effective_data_len = len(numeric_data) - 16 
                    num_samples = effective_data_len // POINTS_PER_SAMPLE
                    
                    if num_samples >= 50:
                        status = "✅ OK"
                    elif num_samples > 0:
                        status = "⚠️ LOW"
                    else:
                        status = "❌ EMPTY"

                    print(f"{label:<12} | {file[:35]:<35} | {int(num_samples):<8} | {status}")
                    total_samples += num_samples
                
                except Exception as e:
                    print(f"{label:<12} | {file[:35]:<35} | ERROR    | 💥 {str(e)[:15]}")

    print("-" * 85)
    print(f"TOTAL VALID SAMPLES IN DATASET: {int(total_samples)}")
    print(f"Goal for Presentation: >1250 samples (250 per class)")

if __name__ == "__main__":
    run_dataset_audit("data/raw")