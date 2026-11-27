"""
Training script for real SEM data (NFFA-EUROPE).
Reuses model definitions from models/ but loads real SEM image pairs.

Usage:
  python train_real.py --model dncnn --noise medium --epochs 100 --batch-size 16
  python train_real.py --model all --noise all --epochs 100 --batch-size 16
"""

import os
import sys
import json
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Import models
from models.dncnn import DnCNN
from models.gan import Generator, Discriminator
from models.cyclegan import CycleGANGenerator, CycleGANDiscriminator
from models.diffusion import DiffusionUNet, GaussianDiffusion


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealSEMDataset(Dataset):
    """Load paired noisy/clean real SEM images."""

    def __init__(self, data_dir, noise_level='medium', split='train'):
        self.noisy_dir = os.path.join(data_dir, noise_level, split, 'noisy')
        self.clean_dir = os.path.join(data_dir, noise_level, split, 'clean')

        self.noisy_files = sorted(glob.glob(os.path.join(self.noisy_dir, '*.png')))
        self.clean_files = sorted(glob.glob(os.path.join(self.clean_dir, '*.png')))

        assert len(self.noisy_files) == len(self.clean_files), \
            f"Mismatch: {len(self.noisy_files)} noisy vs {len(self.clean_files)} clean"
        assert len(self.noisy_files) > 0, \
            f"No images found in {self.noisy_dir}"

        print(f"  [{split}] Loaded {len(self.noisy_files)} pairs from {data_dir}/{noise_level}/{split}/")

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy = np.array(Image.open(self.noisy_files[idx]), dtype=np.float32) / 255.0
        clean = np.array(Image.open(self.clean_files[idx]), dtype=np.float32) / 255.0

        # Add channel dimension: (H,W) -> (1,H,W)
        noisy = torch.from_numpy(noisy).unsqueeze(0)
        clean = torch.from_numpy(clean).unsqueeze(0)

        return noisy, clean


# ---------------------------------------------------------------------------
# Training functions (adapted from train.py)
# ---------------------------------------------------------------------------

def train_dncnn(train_loader, test_loader, device, epochs=100, lr=1e-3):
    model = DnCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  E{epoch}/{epochs} | Loss: {avg_loss:.6f}")

    return model


def train_gan(train_loader, test_loader, device, epochs=100, lr=2e-4):
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    lambda_l1 = 100.0

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        g_losses, d_losses = [], []

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            fake = generator(noisy)

            # Train Discriminator
            opt_d.zero_grad()
            real_pair = torch.cat([noisy, clean], dim=1)
            fake_pair = torch.cat([noisy, fake.detach()], dim=1)
            pred_real = discriminator(real_pair)
            pred_fake = discriminator(fake_pair)
            loss_d = 0.5 * (
                criterion_gan(pred_real, torch.ones_like(pred_real)) +
                criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            )
            loss_d.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            fake_pair = torch.cat([noisy, fake], dim=1)
            pred_fake = discriminator(fake_pair)
            loss_g_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = criterion_l1(fake, clean)
            loss_g = loss_g_gan + lambda_l1 * loss_g_l1
            loss_g.backward()
            opt_g.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  E{epoch}/{epochs} | G: {np.mean(g_losses):.4f} | D: {np.mean(d_losses):.4f}")

    return generator


def train_cyclegan(train_loader, test_loader, device, epochs=100, lr=2e-4):
    G_n2c = CycleGANGenerator().to(device)   # noisy -> clean
    G_c2n = CycleGANGenerator().to(device)   # clean -> noisy
    D_clean = CycleGANDiscriminator().to(device)
    D_noisy = CycleGANDiscriminator().to(device)

    opt_g = optim.Adam(
        list(G_n2c.parameters()) + list(G_c2n.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )
    opt_d = optim.Adam(
        list(D_clean.parameters()) + list(D_noisy.parameters()),
        lr=lr, betas=(0.5, 0.999)
    )

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    lambda_cycle = 10.0
    lambda_identity = 5.0

    for epoch in range(1, epochs + 1):
        G_n2c.train(); G_c2n.train()
        g_losses, d_losses = [], []

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Generator forward
            fake_clean = G_n2c(noisy)
            fake_noisy = G_c2n(clean)
            rec_noisy = G_c2n(fake_clean)
            rec_clean = G_n2c(fake_noisy)

            # Identity
            id_clean = G_n2c(clean)
            id_noisy = G_c2n(noisy)

            # Generator loss
            opt_g.zero_grad()
            loss_g_clean = criterion_gan(D_clean(fake_clean), torch.ones_like(D_clean(fake_clean)))
            loss_g_noisy = criterion_gan(D_noisy(fake_noisy), torch.ones_like(D_noisy(fake_noisy)))
            loss_cycle = criterion_cycle(rec_noisy, noisy) + criterion_cycle(rec_clean, clean)
            loss_id = criterion_identity(id_clean, clean) + criterion_identity(id_noisy, noisy)
            loss_g = loss_g_clean + loss_g_noisy + lambda_cycle * loss_cycle + lambda_identity * loss_id
            loss_g.backward()
            opt_g.step()

            # Discriminator loss
            opt_d.zero_grad()
            loss_d_clean = 0.5 * (
                criterion_gan(D_clean(clean), torch.ones_like(D_clean(clean))) +
                criterion_gan(D_clean(fake_clean.detach()), torch.zeros_like(D_clean(fake_clean.detach())))
            )
            loss_d_noisy = 0.5 * (
                criterion_gan(D_noisy(noisy), torch.ones_like(D_noisy(noisy))) +
                criterion_gan(D_noisy(fake_noisy.detach()), torch.zeros_like(D_noisy(fake_noisy.detach())))
            )
            loss_d = loss_d_clean + loss_d_noisy
            loss_d.backward()
            opt_d.step()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

        if epoch % 10 == 0 or epoch == 1:
            print(f"  E{epoch}/{epochs} | G: {np.mean(g_losses):.4f} | D: {np.mean(d_losses):.4f}")

    return G_n2c


def train_diffusion(train_loader, test_loader, device, epochs=100, lr=2e-4):
    unet = DiffusionUNet().to(device)
    diffusion = GaussianDiffusion(unet, T=500).to(device)
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        unet.train()
        total_loss = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            loss = diffusion.train_loss(clean, noisy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  E{epoch}/{epochs} | Loss: {avg_loss:.6f}")

    return diffusion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TRAINERS = {
    'dncnn': train_dncnn,
    'gan': train_gan,
    'cyclegan': train_cyclegan,
    'diffusion': train_diffusion,
}

def main():
    parser = argparse.ArgumentParser(description='Train on real SEM data')
    parser.add_argument('--model', type=str, required=True,
                        choices=['dncnn', 'gan', 'cyclegan', 'diffusion', 'all'])
    parser.add_argument('--noise', type=str, default='medium',
                        choices=['low', 'medium', 'high', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', type=str, default='data/nffa_sem',
                        help='Path to processed NFFA dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_real')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    models = list(TRAINERS.keys()) if args.model == 'all' else [args.model]
    noises = ['low', 'medium', 'high'] if args.noise == 'all' else [args.noise]

    for noise_level in noises:
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()} on {noise_level} noise (REAL data)")
            print(f"{'='*60}")

            # Adjust lr for different models
            lr = args.lr
            if model_name in ['gan', 'cyclegan', 'diffusion']:
                lr = min(lr, 2e-4)

            # Load data
            train_dataset = RealSEMDataset(args.data_dir, noise_level, 'train')
            test_dataset = RealSEMDataset(args.data_dir, noise_level, 'test')

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=2, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=2, pin_memory=True)

            # Train
            t0 = time.time()
            trainer = TRAINERS[model_name]
            model = trainer(train_loader, test_loader, device, args.epochs, lr)
            elapsed = time.time() - t0
            print(f"  Training time: {elapsed/60:.1f} min")

            # Save checkpoint
            ckpt_dir = os.path.join(args.checkpoint_dir, f"{model_name}_{noise_level}")
            os.makedirs(ckpt_dir, exist_ok=True)

            if model_name == 'cyclegan':
                # model is G_n2c
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'generator_n2c.pth'))
            elif model_name == 'diffusion':
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'diffusion.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{model_name}.pth'))

            print(f"  Checkpoint saved: {ckpt_dir}/")

    print(f"\n{'='*60}")
    print("ALL TRAINING DONE (REAL DATA)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
