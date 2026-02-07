import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os

from dataset import UltrasonicDataset
# Updated the import to match the new class name in model.py
from model import UltrasonicCNN 
from data_loader import load_config

def weights_init(m):
    # Updated to initialize Conv2d as well
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)

def train():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = UltrasonicDataset(config['paths']['raw_dir'])
    
    if len(full_dataset) == 0:
        print("❌ Dataset is empty. Check your data paths.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'])

    num_classes = len(config['dataset']['classes'])
    
    # Updated to initialize UltrasonicCNN
    model = UltrasonicCNN(num_classes=num_classes).to(device)
    model.apply(weights_init)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Training Started (2D Spectrogram CNN) ---")
    
    epochs = config['training']['epochs']
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)

            # Ensure signals are [Batch, 2, 2048] for the Spectrogram transform
            if signals.dim() == 2:
                signals = signals.unsqueeze(1) # Add channel dim if missing

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                if signals.dim() == 2: signals = signals.unsqueeze(1)
                
                outputs = model(signals)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {train_loss/len(train_loader):.3f} | "
              f"Acc: {100.*correct/total:.1f}% | Val Acc: {100.*val_correct/val_total:.1f}% | LR: {curr_lr}")

    model_out_path = config['paths']['model_output']
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    torch.save(model.state_dict(), model_out_path)
    print(f"\n✅ Training Complete! Model saved to {model_out_path}")

if __name__ == "__main__":
    train()