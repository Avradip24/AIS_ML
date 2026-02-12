import torch
import torch.nn as nn
import torchaudio.transforms as T

class UltrasonicCNN(nn.Module):
    def __init__(self, num_classes):
        super(UltrasonicCNN, self).__init__()
        
        # Increased frequency resolution
        self.spectrogram = T.Spectrogram(n_fft=256, hop_length=64)
        
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(2, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # NEW Layer 3: Adds the "depth" needed to break the 20% barrier
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Forces a consistent size for the Linear layer
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), 
            nn.ReLU(),
            nn.Dropout(0.4), # Slightly higher dropout to prevent memorizing the Wall
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Removed torch.no_grad() so the model can actually learn from the signal!
        spec = self.spectrogram(x) 
        
        # Log scaling helps the model "see" quiet echoes behind the loud ones
        spec = torch.log1p(spec) 
        
        x = self.conv_layers(spec)
        return self.fc_layers(x)