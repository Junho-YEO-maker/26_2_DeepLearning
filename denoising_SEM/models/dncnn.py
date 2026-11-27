"""
DnCNN - Denoising CNN with ResNet-style architecture.
Based on Zhang et al. 2017, modified to use residual blocks.
Lightweight: 8 ResNet blocks.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class DnCNN(nn.Module):
    """ResNet-based DnCNN for SEM image denoising.
    Predicts noise residual: output = input - predicted_noise.
    """
    def __init__(self, in_channels=1, num_blocks=8, features=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            *[ResBlock(features) for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(features, in_channels, 3, padding=1, bias=False)

    def forward(self, x):
        residual = self.head(x)
        residual = self.body(residual)
        noise = self.tail(residual)
        return x - noise


if __name__ == "__main__":
    model = DnCNN(in_channels=1, num_blocks=8, features=64)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DnCNN parameters: {n:,}")
    x = torch.randn(1, 1, 256, 256)
    print(f"Input: {x.shape} -> Output: {model(x).shape}")
