import time
import torch
import json
from pathlib import Path
from data_loader import load_config
# Import both model classes
from model import UltrasonicCNN, UltrasonicHierarchicalSoftCNN 
from dataset import UltrasonicDataset

def measure_latency():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up paths
    models_dir = Path(__file__).resolve().parents[1] / "models"
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(exist_ok=True) 

    # --- 1. Load Flat Model ---
    model_flat = UltrasonicCNN(num_classes=5).to(device)
    model_path = config["paths"]["model_output"]
    
    flat_checkpoint = torch.load(model_path, map_location=device)
    if isinstance(flat_checkpoint, dict) and "model_state" in flat_checkpoint:
        model_flat.load_state_dict(flat_checkpoint["model_state"])
    else:
        model_flat.load_state_dict(flat_checkpoint)
    model_flat.eval()

    # --- 2. Load Hierarchical Model ---
    model_hier = UltrasonicHierarchicalSoftCNN().to(device) 
    hier_path = models_dir / "fius_hierarchical_soft_cnn.pth"

    hier_checkpoint = torch.load(hier_path, map_location=device)
    if isinstance(hier_checkpoint, dict) and "model_state" in hier_checkpoint:
        model_hier.load_state_dict(hier_checkpoint["model_state"])
    else:
        model_hier.load_state_dict(hier_checkpoint)
    model_hier.eval()

    # --- 3. Get a Sample ---
    dataset = UltrasonicDataset(config["paths"]["raw_dir"], transform=False, preload=True)
    sample, _ = dataset[0]
    sample = sample.unsqueeze(0).to(device) 

    # --- 4. Warm up ---
    print("Warming up models...")
    with torch.no_grad():
        for _ in range(20):
            _ = model_flat(sample)
            _ = model_hier(sample)

    # --- 5. Measure Flat Latency ---
    print("Measuring Flat Model Latency...")
    times_flat = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model_flat(sample)
            t1 = time.perf_counter()
            times_flat.append((t1 - t0) * 1000)

    # --- 6. Measure Hierarchical Latency ---
    print("Measuring Hierarchical Model Latency...")
    times_hier = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model_hier(sample) 
            t1 = time.perf_counter()
            times_hier.append((t1 - t0) * 1000)

    avg_flat = sum(times_flat) / len(times_flat)
    avg_hier = sum(times_hier) / len(times_hier)

    # --- 7. SAVE RESULTS TO DISK (Formatted for Exporter Compatibility) ---
    latency_profile = {
        "averages": {
            "total_time_ms": avg_hier,
            "forward_time_ms": avg_hier,
            "preprocess_time_ms": 0.0,
            "per_pulse_forward_ms": avg_hier,  # KEY: Needed for Exporter Summary
            "flat_baseline_ms": avg_flat
        },
        "num_files": 1,  # KEY: Prevents "0 files profiled" error
        "metadata": {
            "device": str(device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    with open(results_dir / "latency_profile.json", "w") as f:
        json.dump(latency_profile, f, indent=4)

    print("\n" + "="*50)
    print("LATENCY RESULTS (Inference only)")
    print("="*50)
    print(f"Flat Model Latency:         {avg_flat:.3f} ms")
    print(f"Hierarchical Model Latency: {avg_hier:.3f} ms")
    print("-" * 50)
    print(f"✅ Saved results to {results_dir / 'latency_profile.json'}")
    
    if avg_hier < 10.0:
        print("✅ Meets AIS <10ms requirement.")
    else:
        print("❌ Exceeds AIS <10ms requirement.")
    print("="*50)

if __name__ == "__main__":
    measure_latency()