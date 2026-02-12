import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np

from dataset import UltrasonicDataset
from model import UltrasonicCNN 
from data_loader import load_config

def weights_init(m):
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
    
    # Using config values for batch_size
    # Set num_workers=0 if you encounter 'BrokenPipe' errors on Windows
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=0, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['training']['batch_size'],
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Weights match YAML: ["Wall", "Person", "Chair", "Backpack", "Plant", "BigTable"]
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)

    num_classes = len(config['dataset']['classes'])
    model = UltrasonicCNN(num_classes=num_classes).to(device)
    model.apply(weights_init)
    
    # UPDATE: Added weight_decay from your config.yaml
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0001) 
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=7, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\n--- Training Started (Parallel Pipeline) ---")
    print(f"Targeting {config['training']['epochs']} epochs with LR: {config['training']['learning_rate']}")
    
    epochs = config['training']['epochs']
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss/len(val_loader)
        scheduler.step(avg_val_loss)
        
        curr_lr = optimizer.param_groups[0]['lr']
        val_acc = 100.*val_correct/val_total
        
        # Formatting Epoch as 001/150 for clarity
        print(f"Epoch [{epoch+1:03d}/{epochs}] | Loss: {train_loss/len(train_loader):.3f} | "
              f"Acc: {100.*correct/total:.1f}% | Val Acc: {val_acc:.1f}% | LR: {curr_lr}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_out_path = config['paths']['model_output']
            os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
            torch.save(model.state_dict(), model_out_path)

    print(f"\n✅ Training Complete! Best Val Acc: {best_val_acc:.1f}%")

if __name__ == "__main__":
    # Essential for Windows Multiprocessing
    train()