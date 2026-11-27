"""
Unified training script for 4 denoising models.
Usage: python train.py --model dncnn --noise medium --epochs 100
"""

import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from models.dncnn import DnCNN
from models.gan import Generator as GANGen, Discriminator as GANDisc
from models.cyclegan import CycleGANGenerator, CycleGANDiscriminator
from models.diffusion import DiffusionUNet, GaussianDiffusion


class SEMDataset(Dataset):
    def __init__(self, data_dir, noise_level="medium", split="train"):
        self.clean_dir = Path(data_dir) / split / "clean"
        self.noisy_dir = Path(data_dir) / split / f"noisy_{noise_level}"
        self.files = sorted(list(self.clean_dir.glob("*.png")))
        assert len(self.files) > 0, f"No files in {self.clean_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx].name
        clean = np.array(Image.open(self.clean_dir / fname)).astype(np.float32) / 255.0
        noisy = np.array(Image.open(self.noisy_dir / fname)).astype(np.float32) / 255.0
        return torch.from_numpy(noisy).unsqueeze(0), torch.from_numpy(clean).unsqueeze(0), fname


def train_dncnn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(SEMDataset(args.data_dir, args.noise, "train"),
                        batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(SEMDataset(args.data_dir, args.noise, "val"),
                            batch_size=args.batch_size, num_workers=4)
    model = DnCNN(1, 8, 64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    save_dir = Path(args.save_dir) / f"dncnn_{args.noise}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        tloss = 0
        for noisy, clean, _ in tqdm(loader, desc=f"E{epoch+1}/{args.epochs}"):
            noisy, clean = noisy.to(device), clean.to(device)
            loss = criterion(model(noisy), clean)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tloss += loss.item()
        scheduler.step()
        model.eval()
        vloss = sum(criterion(model(n.to(device)), c.to(device)).item()
                    for n, c, _ in val_loader) / len(val_loader)
        print(f"  Train: {tloss/len(loader):.6f}, Val: {vloss:.6f}")
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), save_dir / "best.pth")
    torch.save(model.state_dict(), save_dir / "last.pth")


def train_gan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(SEMDataset(args.data_dir, args.noise, "train"),
                        batch_size=args.batch_size, shuffle=True, num_workers=4)
    G = GANGen(1, 1, 64).to(device)
    D = GANDisc(2, 64).to(device)
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss(); l1 = nn.L1Loss()
    save_dir = Path(args.save_dir) / f"gan_{args.noise}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        G.train(); D.train()
        for noisy, clean, _ in tqdm(loader, desc=f"E{epoch+1}"):
            noisy, clean = noisy.to(device), clean.to(device)
            fake = G(noisy)
            loss_D = (bce(D(noisy, clean), torch.ones_like(D(noisy, clean))) +
                      bce(D(noisy, fake.detach()), torch.zeros_like(D(noisy, fake.detach())))) / 2
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
            loss_G = bce(D(noisy, fake), torch.ones_like(D(noisy, fake))) + 100 * l1(fake, clean)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    torch.save(G.state_dict(), save_dir / "G_last.pth")


def train_cyclegan(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(SEMDataset(args.data_dir, args.noise, "train"),
                        batch_size=args.batch_size, shuffle=True, num_workers=4)
    G_N2C = CycleGANGenerator(1, 1, 64, 6).to(device)
    G_C2N = CycleGANGenerator(1, 1, 64, 6).to(device)
    D_C = CycleGANDiscriminator(1, 64).to(device)
    D_N = CycleGANDiscriminator(1, 64).to(device)
    opt_G = optim.Adam(list(G_N2C.parameters()) + list(G_C2N.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(list(D_C.parameters()) + list(D_N.parameters()), lr=args.lr, betas=(0.5, 0.999))
    mse = nn.MSELoss(); cyc = nn.L1Loss()
    save_dir = Path(args.save_dir) / f"cyclegan_{args.noise}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        for noisy, clean, _ in tqdm(loader, desc=f"E{epoch+1}"):
            noisy, clean = noisy.to(device), clean.to(device)
            fc, fn = G_N2C(noisy), G_C2N(clean)
            loss_G = (mse(D_C(fc), torch.ones_like(D_C(fc))) + mse(D_N(fn), torch.ones_like(D_N(fn))) +
                      10 * (cyc(G_C2N(fc), noisy) + cyc(G_N2C(fn), clean)) +
                      5 * (cyc(G_N2C(clean), clean) + cyc(G_C2N(noisy), noisy)))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
            loss_D = ((mse(D_C(clean), torch.ones_like(D_C(clean))) + mse(D_C(fc.detach()), torch.zeros_like(D_C(fc)))) +
                      (mse(D_N(noisy), torch.ones_like(D_N(noisy))) + mse(D_N(fn.detach()), torch.zeros_like(D_N(fn))))) / 2
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
    torch.save(G_N2C.state_dict(), save_dir / "G_N2C_last.pth")


def train_diffusion(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(SEMDataset(args.data_dir, args.noise, "train"),
                        batch_size=args.batch_size, shuffle=True, num_workers=4)
    model = DiffusionUNet(1, 1, 64).to(device)
    diffusion = GaussianDiffusion(500, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    save_dir = Path(args.save_dir) / f"diffusion_{args.noise}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for noisy, clean, _ in tqdm(loader, desc=f"E{epoch+1}"):
            noisy, clean = noisy.to(device), clean.to(device)
            t = torch.randint(0, diffusion.T, (clean.shape[0],), device=device)
            noisy_t, noise = diffusion.q_sample(clean, t)
            loss = nn.MSELoss()(model(noisy_t, t, noisy), noise)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        print(f"  Loss: {np.mean(losses):.6f}")
    torch.save(model.state_dict(), save_dir / "model_last.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["dncnn", "gan", "cyclegan", "diffusion"])
    parser.add_argument("--data-dir", default="data/synthetic_sem_dataset")
    parser.add_argument("--noise", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=2021147017)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Training {args.model} on {args.noise} noise...")
    {"dncnn": train_dncnn, "gan": train_gan, "cyclegan": train_cyclegan, "diffusion": train_diffusion}[args.model](args)
