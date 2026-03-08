import os
import numpy as np

def run_dataset_audit(base_folder, set_name="DATASET"):
    # Constants based on your sensor configuration
    POINTS_PER_SAMPLE = 2048 
    
    print(f"\n--- AUDIT FOR {set_name}: {base_folder} ---")
    print(f"{'LABEL/FILE':<25} | {'SAMPLES':<8} | {'STATUS'}")
    print("-" * 55)

    if not os.path.exists(base_folder):
        print(f"Error: Folder '{base_folder}' not found!")
        return 0

    total_samples = 0

    for root, dirs, files in os.walk(base_folder):
        # Skip FFT folders as the model uses Raw ADC
        if "fft" in root.lower():
            continue
            
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                
                # Logic to handle your nested 'test' folder
                parts = root.split(os.sep)
                if 'test' in parts:
                    # If it's in the test folder, show the filename (e.g., adc_testdata1.txt)
                    display_name = f"TEST: {file[:18]}"
                else:
                    # Otherwise, show the category name (e.g., backpack, person)
                    display_name = parts[-1] 
                
                try:
                    with open(path, 'r') as f:
                        content = f.read().split()
                    
                    # Count valid numeric data points to verify file integrity
                    numeric_data_count = sum(1 for word in content if word.replace('.','',1).isdigit())
                    
                    # Subtract header (16) and calculate total pulses in file
                    effective_data_len = numeric_data_count - 16 
                    num_samples = effective_data_len // POINTS_PER_SAMPLE
                    
                    if num_samples > 0:
                        status = "✅ OK"
                    else:
                        status = "❌ EMPTY"

                    print(f"{display_name:<25} | {int(num_samples):<8} | {status}")
                    total_samples += num_samples
                
                except Exception as e:
                    print(f"{display_name:<25} | ERROR    | 💥 {str(e)[:15]}")

    print("-" * 55)
    print(f"TOTAL SAMPLES DETECTED: {int(total_samples)}")
    return total_samples

if __name__ == "__main__":
    # This audits all subfolders (backpack, wall, etc.) AND the test folder inside raw
    final_count = run_dataset_audit("data/raw", "FULL DATASET")
    
    if final_count >= 1250:
        print("🏆 SUCCESS: Dataset meets presentation requirements!")
    else:
        print(f"⚠️  NOTICE: You need {1250 - final_count} more samples to hit your target.")