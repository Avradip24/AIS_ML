import torch
import torch.nn as nn
import torch.nn.functional as F

class UltrasonicCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(UltrasonicCNN, self).__init__()

        # ADC branch (channels 0,1) and FFT branch (channels 2,3).
        self.adc_conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3)
        self.adc_bn1 = nn.BatchNorm1d(16)
        self.adc_conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.adc_bn2 = nn.BatchNorm1d(32)
        self.adc_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.adc_bn3 = nn.BatchNorm1d(64)

        self.fft_conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3)
        self.fft_bn1 = nn.BatchNorm1d(16)
        self.fft_conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.fft_bn2 = nn.BatchNorm1d(32)
        self.fft_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fft_bn3 = nn.BatchNorm1d(64)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        adc = x[:, 0:2, :]
        fft = x[:, 2:4, :]

        adc = F.relu(self.adc_bn1(self.adc_conv1(adc)))
        adc = F.relu(self.adc_bn2(self.adc_conv2(adc)))
        adc = F.relu(self.adc_bn3(self.adc_conv3(adc)))
        adc = self.pool(adc).view(adc.size(0), -1)

        fft = F.relu(self.fft_bn1(self.fft_conv1(fft)))
        fft = F.relu(self.fft_bn2(self.fft_conv2(fft)))
        fft = F.relu(self.fft_bn3(self.fft_conv3(fft)))
        fft = self.pool(fft).view(fft.size(0), -1)

        x = torch.cat([adc, fft], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        logits = self.fc2(x)
        return logits
