import time
import torch
from pathlib import Path
from data_loader import load_config
from model import UltrasonicCNN
from dataset import UltrasonicDataset
from torch.utils.data import DataLoader

def measure_latency():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load flat model
    model = UltrasonicCNN(num_classes=5).to(device)
    model_path = config["paths"]["model_output"]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load hierarchical models
    models_dir = Path(__file__).resolve().parents[1] / "models"
    group_model = UltrasonicCNN(num_classes=2).to(device)
    group0_model = UltrasonicCNN(num_classes=3).to(device)
    group1_model = UltrasonicCNN(num_classes=2).to(device)
    group_model.load_state_dict(torch.load(models_dir / "fius_group_cnn.pth", map_location=device))
    group0_model.load_state_dict(torch.load(models_dir / "fius_group0_cnn.pth", map_location=device))
    group1_model.load_state_dict(torch.load(models_dir / "fius_group1_cnn.pth", map_location=device))
    group_model.eval()
    group0_model.eval()
    group1_model.eval()

    # Get a sample
    dataset = UltrasonicDataset(config["paths"]["raw_dir"], transform=False, preload=True)
    sample, _ = dataset[0]
    sample = sample.unsqueeze(0).to(device)  # Add batch dim

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            model(sample)
            group_model(sample)
            group0_model(sample)
            group1_model(sample)

    # Measure flat latency
    times_flat = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            _ = model(sample)
            t1 = time.perf_counter()
            times_flat.append((t1 - t0) * 1000)

    # Measure hierarchical latency
    times_hier = []
    with torch.no_grad():
        for _ in range(100):
            t0 = time.perf_counter()
            group_logits = group_model(sample)
            group_pred = group_logits.argmax(dim=1)
            if group_pred == 0:
                _ = group0_model(sample)
            else:
                _ = group1_model(sample)
            t1 = time.perf_counter()
            times_hier.append((t1 - t0) * 1000)

    avg_flat = sum(times_flat) / len(times_flat)
    avg_hier = sum(times_hier) / len(times_hier)

    print(f"Flat model forward-only latency: {avg_flat:.2f} ms")
    print(f"Hierarchical model end-to-end latency: {avg_hier:.2f} ms")

    # Note: AIS requirement is for pure model inference per pulse, not full pipeline
    print("Note: AIS <10ms requirement applies to pure model forward-pass per pulse.")
    print("Use profile_latency.py for detailed per-pulse measurements.")

if __name__ == "__main__":
    measure_latency()