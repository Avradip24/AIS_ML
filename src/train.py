import torch
import torch.optim as optim
from data_loader import process_file, load_config
from model import FIUS_CNN
import os

def train():
    config = load_config()
    model = FIUS_CNN(num_classes=len(config['dataset']['classes']))
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    print("Starting training development with Wall and Person data...")
    
    # Logic to loop through data/raw/wall and data/raw/person
    # (In a full script, you would create a PyTorch Dataset class here)
    
    # Save the 'brain' after training
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), config['paths']['model_save_path'])
    print(f"Model saved to {config['paths']['model_save_path']}")

if __name__ == "__main__":
    train()