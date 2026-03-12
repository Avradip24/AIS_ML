import os
import numpy as np
from pathlib import Path
# Import the specific internal functions for parsing and preprocessing
from data_loader import (
    _read_txt_pulses, 
    _parse_fius_fft_file, 
    GLOBAL_CONFIG, 
    _normalize_and_energy, 
    _fit_to_input_size
)

def convert_to_binary():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, "data", "raw")
    bin_dir = os.path.join(project_root, "data", "binary")
    
    # Configuration from yaml
    input_size = int(GLOBAL_CONFIG["dataset"]["input_size"])
    # Based on your data structure: 8 ADC rows = 1 temporal snapshot
    ROWS_PER_ADC_SAMPLE = 8  
    
    converted_count = 0
    
    print(f"🚀 Scanning Raw Data: {raw_dir}")

    for root, dirs, files in os.walk(raw_dir):
        if "adc_measurements" not in root:
            continue

        for f in files:
            if f.startswith("adc_") and f.endswith(".txt"):
                adc_path = os.path.join(root, f)
                
                # Pair with FFT file
                fft_folder = root.replace("adc_measurements", "fft_measurements")
                fft_filename = f.replace("adc_", "fft_")
                fft_path = os.path.join(fft_folder, fft_filename)
                
                if not os.path.exists(fft_path):
                    continue

                try:
                    # 1. Load Raw Data
                    adc_data = _read_txt_pulses(adc_path, input_size)
                    fft_data = _parse_fius_fft_file(fft_path)
                    
                    if adc_data is None or fft_data is None:
                        continue
                    
                    # 2. Sync sample counts
                    num_adc_samples = len(adc_data) // ROWS_PER_ADC_SAMPLE
                    num_fft_samples = len(fft_data)
                    num_samples = min(num_adc_samples, num_fft_samples)
                    
                    if num_samples == 0:
                        continue

                    processed_file_samples = []

                    for i in range(num_samples):
                        # Get raw pieces
                        # We use the first pulse of the 8-row block for the 4-channel stack
                        adc_raw = adc_data[i * ROWS_PER_ADC_SAMPLE]
                        fft_raw = _fit_to_input_size(fft_data[i], input_size)

                        # 3. Apply the 4-channel Preprocessing Logic
                        adc_norm, adc_energy = _normalize_and_energy(adc_raw)
                        fft_norm, fft_energy = _normalize_and_energy(fft_raw)

                        # Stack into shape: (4, input_size) -> (4, 2048)
                        combined = np.stack([adc_norm, adc_energy, fft_norm, fft_energy], axis=0)
                        processed_file_samples.append(combined.astype(np.float32))

                    # 4. Save as a standard NumPy array (N, 4, 2048)
                    category = os.path.basename(os.path.dirname(root))
                    save_dir = os.path.join(bin_dir, category)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    out_name = f.replace("adc_", "").replace(".txt", ".npy")
                    output_array = np.array(processed_file_samples)
                    
                    np.save(os.path.join(save_dir, out_name), output_array)
                    
                    print(f"✅ {category}/{out_name} | Preprocessed Samples: {num_samples}")
                    converted_count += 1

                except Exception as e:
                    print(f"❌ Failed to process {f}: {str(e)}")

    print(f"\n🎉 Binary dataset ready for Training! Total files: {converted_count}")

if __name__ == "__main__":
    convert_to_binary()