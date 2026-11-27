"""
CycleGAN for unpaired SEM denoising.
Generator: ResNet-based (6 residual blocks).
Discriminator: PatchGAN.
Learns noisy<->clean mapping without paired data.
"""

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator(nn.Module):
    """ResNet-based generator. 6 ResNet blocks, ~2.3M params."""
    def __init__(self, in_channels=1, out_channels=1, features=64, num_residual=6):
        super().__init__()
        f = features
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, f, 7, bias=False),
            nn.InstanceNorm2d(f),
            nn.ReLU(inplace=True),
        ]
        # Downsampling
        for mult in [2, 4]:
            model += [
                nn.Conv2d(f * mult // 2, f * mult, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(f * mult),
                nn.ReLU(inplace=True),
            ]
        # Residual blocks
        for _ in range(num_residual):
            model.append(ResNetBlock(f * 4))
        # Upsampling
        for mult in [4, 2]:
            model += [
                nn.ConvTranspose2d(f * mult, f * mult // 2, 3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(f * mult // 2),
                nn.ReLU(inplace=True),
            ]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(f, out_channels, 7), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CycleGANDiscriminator(nn.Module):
    """PatchGAN discriminator (3-layer)."""
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        f = features
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, f, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f, f * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 2, f * 4, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(f * 4, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    G = CycleGANGenerator(1, 1, 64, num_residual=6)
    D = CycleGANDiscriminator(1, 64)
    print(f"Generator: {sum(p.numel() for p in G.parameters()):,} params")
    print(f"Discriminator: {sum(p.numel() for p in D.parameters()):,} params")
