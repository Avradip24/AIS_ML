import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np

from dataset import UltrasonicDataset
from model import UltrasonicCNN 
from data_loader import load_config

# --- Early Stopping Utility (Preserved) ---
class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.001): 
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

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def train():
    # 1. Setup & Config
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Dataset Loading (Transform=True enables your noise logic)
    full_dataset = UltrasonicDataset(config['paths']['raw_dir'], transform=True)
    print(f"Detected Classes: {full_dataset.classes}")
    
    if len(full_dataset) == 0:
        print("❌ Dataset is empty. Check your data paths.")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Keeping your batch size and shuffle logic
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 3. Model Initialization
    num_classes = len(full_dataset.classes)
    model = UltrasonicCNN(num_classes=num_classes).to(device)
    model.apply(init_weights)
    
    # Preserved Adam config with Weight Decay
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    
    # Scheduler: Slightly more patience for the higher-res data
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=25)

    # 4. Loss Function (Aligned Weights)
    # Order: [wall, person, chair, backpack, plant, bigtable]
    # Person (1) and Bigtable (5) are the priorities here
    weights = torch.tensor([1.5, 2.0, 1.5, 1.5, 1.0, 1.8]).to(device)

    # Preserving Label Smoothing
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.15) # Increased from 0.1

    epochs = config['training']['epochs']
    best_val_acc = 0.0

    # 5. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        print(f"🚀 Starting Epoch {epoch+1}...")
        
        # Batch processing print logic preserved
        batch_idx = 0
        for signals, labels in train_loader:
            batch_idx += 1
            if batch_idx % 10 == 0:
                print(f"📦 Batch {batch_idx}/{len(train_loader)} processing...", end='\r')

            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            
            class_logits, _ = model(signals) 
            loss = criterion(class_logits, labels)
            loss.backward()
            
            # Gradient Clipping (Your safety logic)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = class_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                class_logits, _ = model(signals)
                v_loss = criterion(class_logits, labels)
                val_loss += v_loss.item()
                
                _, predicted = class_logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss)
        
        print(f"Epoch [{epoch+1:03d}/{epochs}] | Loss: {train_loss/len(train_loader):.3f} | "
              f"Val Loss: {avg_val_loss:.3f} | Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config['paths']['model_output'])

        if early_stopping.early_stop:
            print("🛑 Early stopping triggered. Generalization limit reached.")
            break

    print(f"\n✅ Training Complete! Best Val Acc: {best_val_acc:.1f}%")

    # 6. Final Evaluation (Preserved Per-Class Accuracy logic)
    model.eval()
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    
    with torch.no_grad():
        for signals, labels in val_loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs, _ = model(signals)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("\n📊 --- Final Per-Class Accuracy ---")
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"Accuracy of {full_dataset.classes[i]:<10}: {acc:>5.1f}%")

if __name__ == "__main__":
    train()