import torch
import torch.nn as nn
import torchaudio.transforms as T

class UltrasonicCNN(nn.Module):
    def __init__(self, num_classes):
        super(UltrasonicCNN, self).__init__()
        self.spectrogram = T.Spectrogram(n_fft=256, hop_length=64)
        
        # Keep your previous 3-layer CNN backbone intact
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Head 1: Object Classification (Multi-class)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), 
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

        # Head 2: Range Estimation (Regression for distance)
        # This fulfills the "Multi-Task Learning" optional extension 
        self.range_estimator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs a single continuous value (distance)
        )

    def forward(self, x):
        spec = torch.log1p(self.spectrogram(x)) 
        features = self.conv_layers(spec)
        
        class_output = self.classifier(features)
        range_output = self.range_estimator(features)
        
        return class_output, range_output