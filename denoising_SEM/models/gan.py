"""
Pix2Pix-style GAN for SEM denoising.
Generator: U-Net with skip connections (4 downsamples).
Discriminator: PatchGAN (70x70 receptive field).
"""

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class Generator(nn.Module):
    """U-Net Generator. 4 down + 4 up with skip connections. ~2M params."""
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super().__init__()
        f = features
        self.d1 = UNetDown(in_channels, f, normalize=False)
        self.d2 = UNetDown(f, f * 2)
        self.d3 = UNetDown(f * 2, f * 4)
        self.d4 = UNetDown(f * 4, f * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, 4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.u1 = UNetUp(f * 8, f * 8, dropout=True)
        self.u2 = UNetUp(f * 16, f * 4)
        self.u3 = UNetUp(f * 8, f * 2)
        self.u4 = UNetUp(f * 4, f)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        bn = self.bottleneck(d4)
        u1 = self.u1(bn, d4)
        u2 = self.u2(u1, d3)
        u3 = self.u3(u2, d2)
        u4 = self.u4(u3, d1)
        return self.final(u4)


class Discriminator(nn.Module):
    """PatchGAN discriminator."""
    def __init__(self, in_channels=2, features=64):
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

    def forward(self, noisy, target):
        x = torch.cat([noisy, target], dim=1)
        return self.model(x)


if __name__ == "__main__":
    G = Generator(1, 1, 64)
    D = Discriminator(2, 64)
    print(f"Generator: {sum(p.numel() for p in G.parameters()):,} params")
    print(f"Discriminator: {sum(p.numel() for p in D.parameters()):,} params")
