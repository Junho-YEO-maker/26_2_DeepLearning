"""
Lightweight DDPM for SEM denoising.
Small U-Net with time embedding, conditioned on noisy SEM image.
500 timesteps. Supports multiple samples for uncertainty estimation.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def gn(ch):
    """GroupNorm that handles small channel counts."""
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(gn(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.conv2 = nn.Sequential(gn(out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class DiffusionUNet(nn.Module):
    """Lightweight U-Net for conditional DDPM. ~4M params."""
    def __init__(self, in_channels=1, cond_channels=1, base_channels=64):
        super().__init__()
        ch = base_channels
        time_dim = ch * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(ch), nn.Linear(ch, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        c_in = in_channels + cond_channels
        self.enc1 = ResBlock(c_in, ch, time_dim)
        self.down1 = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
        self.enc2 = ResBlock(ch, ch * 2, time_dim)
        self.down2 = nn.Conv2d(ch * 2, ch * 2, 4, stride=2, padding=1)
        self.enc3 = ResBlock(ch * 2, ch * 4, time_dim)
        self.down3 = nn.Conv2d(ch * 4, ch * 4, 4, stride=2, padding=1)
        self.mid = ResBlock(ch * 4, ch * 4, time_dim)
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1)
        self.dec3 = ResBlock(ch * 8, ch * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(ch * 4, ch * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)
        self.dec1 = ResBlock(ch * 2, ch, time_dim)
        self.out = nn.Conv2d(ch, in_channels, 1)

    def forward(self, x, t, condition):
        t_emb = self.time_mlp(t)
        x = torch.cat([x, condition], dim=1)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        m = self.mid(self.down3(e3), t_emb)
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out(d1)


class GaussianDiffusion:
    """DDPM scheduler. 500 timesteps."""
    def __init__(self, num_timesteps=500, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.T = num_timesteps
        self.device = device
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alpha_bar[t][:, None, None, None]
        b = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return a * x0 + b * noise, noise

    def p_sample(self, model, xt, t, condition):
        with torch.no_grad():
            t_tensor = torch.full((xt.shape[0],), t, device=self.device, dtype=torch.long)
            pred_noise = model(xt, t_tensor, condition)
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bar[t]
            beta = self.betas[t]
            mean = (1.0 / alpha.sqrt()) * (xt - beta / (1.0 - alpha_bar).sqrt() * pred_noise)
            if t > 0:
                mean = mean + beta.sqrt() * torch.randn_like(xt)
            return mean

    @torch.no_grad()
    def sample(self, model, condition, num_samples=1):
        B, C, H, W = condition.shape
        all_samples = []
        for _ in range(num_samples):
            x = torch.randn(B, C, H, W, device=self.device)
            for t in reversed(range(self.T)):
                x = self.p_sample(model, x, t, condition)
            all_samples.append(x.clamp(0, 1))
        return all_samples[0] if num_samples == 1 else torch.stack(all_samples)


if __name__ == "__main__":
    model = DiffusionUNet(1, 1, 64)
    print(f"Diffusion U-Net: {sum(p.numel() for p in model.parameters()):,} params")
