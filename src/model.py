import torch
import torch.nn as nn
import torch.nn.functional as F

class UltrasonicCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(UltrasonicCNN, self).__init__()
        
        # Channel 0: Texture, Channel 1: Envelope, Channel 2: Spectral Centroid
        self.conv1 = nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # --- ENHANCED FC LAYERS ---
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.5) # Increased from 0.3 to 0.5
        
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Convolutional Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global Average Pooling
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification Head with Heavy Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x) # Prevents memorization of "Wall" vs "Table" noise
        
        logits = self.fc2(x)
        return logits, x