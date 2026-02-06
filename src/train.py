import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import UltrasonicDataset  # You need to create this file next
from model import FIUS_CNN
from data_loader import load_config
import os

def train():
    config = load_config()
    
    # 1. Load the data using the folder structure we discussed
    # This will go into data/raw/wall/adc_measurements/ etc.
    full_dataset = UltrasonicDataset(config['paths']['raw_dir'])
    
    if len(full_dataset) == 0:
        print("Error: No data found! Check your folder paths.")
        return

    # 2. Split into Training (80%) and Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'])

    # 3. Initialize Model, Loss, and Optimizer
    model = FIUS_CNN(num_classes=len(config['dataset']['classes']))
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Starting training on {len(full_dataset)} samples...")

    # 4. The actual Training Loop
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        for signals, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{config['training']['epochs']}] - Loss: {running_loss/len(train_loader):.4f}")

    # 5. Save the 'brain'
    os.makedirs(os.path.dirname(config['paths']['model_output']), exist_ok=True)
    torch.save(model.state_dict(), config['paths']['model_output'])
    print(f"Model successfully saved to {config['paths']['model_output']}")

if __name__ == "__main__":
    train()