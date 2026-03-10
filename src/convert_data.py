import os
import numpy as np
from data_loader import load_config, process_file

def convert_to_binary():
    config = load_config()
    # Let's use the absolute path to be 100% sure
    raw_dir = os.path.abspath("./data/raw")
    bin_dir = os.path.abspath("./data/binary")
    
    print(f"🚀 Searching in: {raw_dir}")
    
    if not os.path.exists(raw_dir):
        print(f"❌ ERROR: The path {raw_dir} does not exist!")
        return

    converted_count = 0
    
    for root, dirs, files in os.walk(raw_dir):
        # We check if 'adc_measurements' is in the path to keep data clean
        if "adc_measurements" in root.lower():
            for f in files:
                if f.endswith(".txt"):
                    txt_path = os.path.join(root, f)
                    
                    # Create matching structure in binary folder
                    rel_path = os.path.relpath(root, raw_dir)
                    target_folder = os.path.join(bin_dir, rel_path)
                    
                    if not os.path.exists(target_folder):
                        os.makedirs(target_folder)
                    
                    target_file = os.path.join(target_folder, f.replace('.txt', '.npy'))
                    
                    try:
                        measurements = process_file(txt_path)
                        # Convert list to numpy array and save
                        np.save(target_file, np.array(measurements, dtype=np.float32))
                        converted_count += 1
                        print(f"✅ [{converted_count}] Converted: {f}")
                    except Exception as e:
                        print(f"⚠️ Failed {f}: {e}")

    if converted_count == 0:
        print("\n❓ Still 0 files. Let's list what I saw:")
        # Diagnostic: Show the first few folders found
        for name in os.listdir(raw_dir):
            print(f"  Found folder: {name}")
    else:
        print(f"\n🎉 Done! Converted {converted_count} files to: {bin_dir}")

if __name__ == "__main__":
    convert_to_binary()
