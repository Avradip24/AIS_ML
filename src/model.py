import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Learns to re-weight each channel based on global context.
    Helps the model decide: 'for THIS sample, is the ADC or FFT channel more informative?'
    Negligible parameter cost (~128 extra params per branch).
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(1, channels // reduction))
        self.fc2 = nn.Linear(max(1, channels // reduction), channels)

    def forward(self, x):
        # x: [batch, channels, length]
        s = x.mean(dim=2)                      # Squeeze: global avg pool -> [batch, channels]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))         # Gate: [batch, channels]
        return x * s.unsqueeze(2)              # Excite: rescale each channel


class UltrasonicCNN(nn.Module):
    """
    Dual-branch 1D CNN with SE attention for ultrasonic object classification.
    Branch 1 (ADC): time-domain waveform + cumulative energy envelope.
    Branch 2 (FFT): frequency-domain magnitude + energy envelope.
    Each branch has a Squeeze-and-Excitation block after the conv layers.
    """
    def __init__(self, num_classes=6, dropout_rate=0.4):
        super(UltrasonicCNN, self).__init__()

        # --- ADC Branch (Time Domain / Distance) ---
        self.adc_conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3)
        self.adc_bn1 = nn.BatchNorm1d(16)
        self.adc_conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.adc_bn2 = nn.BatchNorm1d(32)
        self.adc_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.adc_bn3   = nn.BatchNorm1d(64)
        self.adc_se    = SEBlock(64, reduction=4)   # channel attention after conv

        # --- FFT Branch (Frequency Domain / Material) ---
        self.fft_conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=2, padding=3)
        self.fft_bn1   = nn.BatchNorm1d(16)
        self.fft_conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.fft_bn2   = nn.BatchNorm1d(32)
        self.fft_conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fft_bn3   = nn.BatchNorm1d(64)
        self.fft_se    = SEBlock(64, reduction=4)   # channel attention after conv

        # MaxPool preserves sharp echo peaks better than AvgPool
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Classifier head: 64 (ADC) + 64 (FFT) = 128 fused features
        self.fc1      = nn.Linear(128, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2      = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch, 4, length]
        # Channels: 0=ADC norm, 1=ADC energy, 2=FFT norm, 3=FFT energy
        adc_in = x[:, 0:2, :]
        fft_in = x[:, 2:4, :]

        # ADC branch
        adc = F.leaky_relu(self.adc_bn1(self.adc_conv1(adc_in)))
        adc = F.leaky_relu(self.adc_bn2(self.adc_conv2(adc)))
        adc = F.leaky_relu(self.adc_bn3(self.adc_conv3(adc)))
        adc = self.adc_se(adc)                      # SE attention
        adc = self.pool(adc).view(adc.size(0), -1)  # [batch, 64]

        # FFT branch
        fft = F.leaky_relu(self.fft_bn1(self.fft_conv1(fft_in)))
        fft = F.leaky_relu(self.fft_bn2(self.fft_conv2(fft)))
        fft = F.leaky_relu(self.fft_bn3(self.fft_conv3(fft)))
        fft = self.fft_se(fft)                      # SE attention
        fft = self.pool(fft).view(fft.size(0), -1)  # [batch, 64]

        # Fuse branches and classify
        combined = torch.cat([adc, fft], dim=1)     # [batch, 128]
        combined = F.relu(self.fc1(combined))
        combined = self.dropout1(combined)
        return self.fc2(combined)