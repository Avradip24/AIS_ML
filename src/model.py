import torch
import torch.nn as nn
import torch.nn.functional as F

class UltrasonicCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(UltrasonicCNN, self).__init__()
        # Input: (2, 256)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1) # New Layer
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # 256 -> pool -> 128 -> pool -> 64 -> pool -> 32
        # 256 channels * 32 length = 8192
        self.fc1 = nn.Linear(256 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x)))) # New Layer in forward
        
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x), None