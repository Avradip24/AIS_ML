import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np

from dataset import UltrasonicDataset
from model import UltrasonicCNN 
from data_loader import load_config

# --- NEW: Early Stopping Utility ---
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
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

    num_classes = len(config['dataset']['classes'])
    model = UltrasonicCNN(num_classes=num_classes).to(device)
    model.apply(weights_init)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.0001) 
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=10)

    # --- NEW: Class Weighting for Safety ---
    # We give 'Person' (index 1) a higher weight to ensure the model prioritizes avoiding human collisions.
    weights = torch.ones(num_classes).to(device)
    weights[1] = 2.0  # Double the importance of correctly identifying people
    criterion_cls = nn.CrossEntropyLoss(weight=weights)
    
    criterion_range = nn.MSELoss()

    print(f"\n--- Perfected Multi-Task Training Started ---")
    print(f"Tasks: Classification (Weighted) + Range Estimation")
    
    epochs = config['training']['epochs']
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            range_targets = labels.float().unsqueeze(1) 

            optimizer.zero_grad()
            class_logits, range_preds = model(signals)
            
            loss_cls = criterion_cls(class_logits, labels)
            loss_range = criterion_range(range_preds, range_targets)
            
            # Combine losses with a task-weight (0.5 for range)
            combined_loss = loss_cls + (0.5 * loss_range) 
            
            combined_loss.backward()
            optimizer.step()
            
            train_loss += combined_loss.item()
            _, predicted = class_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                range_targets = labels.float().unsqueeze(1)

                class_logits, range_preds = model(signals)
                v_loss = criterion_cls(class_logits, labels) + (0.5 * criterion_range(range_preds, range_targets))
                val_loss += v_loss.item()
                
                _, predicted = class_logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss/len(val_loader)
        val_acc = 100.*val_correct/val_total
        
        # Step Scheduler and Early Stopping
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        
        print(f"Epoch [{epoch+1:03d}/{epochs}] | Loss: {train_loss/len(train_loader):.3f} | "
              f"Val Loss: {avg_val_loss:.3f} | Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config['paths']['model_output'])

        if early_stopping.early_stop:
            print("🛑 Early stopping triggered. Model has reached peak performance.")
            break

    print(f"\n✅ Training Complete! Best Val Acc: {best_val_acc:.1f}%")

if __name__ == "__main__":
    train()