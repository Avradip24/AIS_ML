import os

def run_dataset_audit(base_folder):
    # Constants based on your Red Pitaya settings
    HEADER_SIZE = 16
    POINTS_PER_SAMPLE = 2048
    
    print(f"\n{'OBJECT':<12} | {'FILE NAME':<30} | {'SAMPLES':<8} | {'STATUS'}")
    print("-" * 75)

    if not os.path.exists(base_folder):
        print(f"Error: Folder '{base_folder}' not found!")
        return

    for root, dirs, files in os.walk(base_folder):
        # We skip the FFT measurements folder as we only need Raw ADC for training
        if "fft" in root.lower():
            continue
            
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                
                try:
                    with open(path, 'r') as f:
                        # Split handles the tabs and V0.2 text safely
                        content = f.read().split()
                        total_elements = len(content)
                    
                    # MATH: (Total - Header) / 2048
                    # This gives us the exact number of pulses captured
                    num_samples = (total_elements - HEADER_SIZE) / POINTS_PER_SAMPLE
                    
                    # We flag anything less than 50 as SHORT
                    if num_samples >= 50:
                        status = "✅ OK"
                    else:
                        status = "❌ SHORT"

                    # Print the results with the new SAMPLES column
                    print(f"{label:<12} | {file[:30]:<30} | {int(num_samples):<8} | {status}")
                
                except Exception as e:
                    print(f"{label:<12} | {file[:30]:<30} | ERROR    | 💥 Read Failed")

if __name__ == "__main__":
    # Ensure your data is in the 'data/raw' folder
    run_dataset_audit("data/raw")