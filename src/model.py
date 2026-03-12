import torch
import torch.nn as nn
import torch.nn.functional as F

class UltrasonicCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(UltrasonicCNN, self).__init__()

        # --- ADC Branch (Time Domain / Distance) ---
        self.adc_conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3)
        self.adc_bn1 = nn.BatchNorm1d(16)
        self.adc_conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.adc_bn2 = nn.BatchNorm1d(32)
        self.adc_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.adc_bn3 = nn.BatchNorm1d(64)

        # --- FFT Branch (Frequency Domain / Material) ---
        self.fft_conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3)
        self.fft_bn1 = nn.BatchNorm1d(16)
        self.fft_conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.fft_bn2 = nn.BatchNorm1d(32)
        self.fft_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fft_bn3 = nn.BatchNorm1d(64)

        # CHANGE: Use MaxPool to preserve the sharp peaks of the echoes
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Final Decision Layers
        # 64 (ADC) + 64 (FFT) = 128 inputs
        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.4) # Slightly lower dropout for stability
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: [batch, 4, length] 
        # (Assuming your dataset loader puts ADC in 0:2 and FFT in 2:4)
        adc_in = x[:, 0:2, :]
        fft_in = x[:, 2:4, :]

        # Feature extraction - ADC
        adc = F.leaky_relu(self.adc_bn1(self.adc_conv1(adc_in)))
        adc = F.leaky_relu(self.adc_bn2(self.adc_conv2(adc)))
        adc = F.leaky_relu(self.adc_bn3(self.adc_conv3(adc)))
        adc = self.pool(adc).view(adc.size(0), -1)

        # Feature extraction - FFT
        fft = F.leaky_relu(self.fft_bn1(self.fft_conv1(fft_in)))
        fft = F.leaky_relu(self.fft_bn2(self.fft_conv2(fft)))
        fft = F.leaky_relu(self.fft_bn3(self.fft_conv3(fft)))
        fft = self.pool(fft).view(fft.size(0), -1)

        # Concatenate: Fusion of Time and Frequency
        combined = torch.cat([adc, fft], dim=1)

        # Classification
        combined = F.relu(self.fc1(combined))
        combined = self.dropout1(combined)
        logits = self.fc2(combined)
        
        return logits