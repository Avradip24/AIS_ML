import torch
import torch.nn as nn
import torchaudio.transforms as T

class UltrasonicCNN(nn.Module):
    def __init__(self, num_classes):
        super(UltrasonicCNN, self).__init__()
        
        # Increased n_fft for better frequency resolution
        self.spectrogram = T.Spectrogram(n_fft=256, hop_length=64)
        
        self.features = nn.Sequential(
            # Input: [Batch, 2, 129, 33] (Frequency x Time)
            nn.Conv2d(2, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            # Calculation: 129/2/2 = 32; 33/2/2 = 8 -> 64 * 32 * 8
            nn.Linear(64 * 32 * 8, 256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [Batch, 2, 2048]
        with torch.no_grad():
            spec = self.spectrogram(x) # [Batch, 2, 129, 33]
            spec = torch.log1p(spec)   # Log scale helps AI see faint echoes
        return self.features(spec)