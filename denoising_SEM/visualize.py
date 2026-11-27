"""
Visualization: denoising comparison images + metric charts.
Usage: python visualize.py --noise medium
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from train import SEMDataset
from models.dncnn import DnCNN
from models.gan import Generator as GANGen
from models.cyclegan import CycleGANGenerator
from models.diffusion import DiffusionUNet, GaussianDiffusion
from metrics.metrology import measure_edge_profile, detect_edges


def load_model(name, noise, ckpt_dir="checkpoints", device="cuda"):
    paths = {
        "dncnn": (DnCNN(1, 8, 64), f"dncnn_{noise}/best.pth"),
        "gan": (GANGen(1, 1, 64), f"gan_{noise}/G_last.pth"),
        "cyclegan": (CycleGANGenerator(1, 1, 64, 6), f"cyclegan_{noise}/G_N2C_last.pth"),
        "diffusion": (DiffusionUNet(1, 1, 64), f"diffusion_{noise}/model_last.pth"),
    }
    model, p = paths[name]
    model = model.to(device)
    path = Path(ckpt_dir) / p
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def denoise_one(model, noisy_tensor, name, device, diff_sched=None):
    with torch.no_grad():
        x = noisy_tensor.unsqueeze(0).to(device)
        if name == "diffusion":
            return diff_sched.sample(model, x, 1).squeeze().cpu().numpy()
        return model(x).squeeze().cpu().numpy()


def make_comparison_grid(args):
    """Generate side-by-side denoising comparison images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SEMDataset(args.data_dir, args.noise, "test")
    models_list = ["dncnn", "gan", "cyclegan", "diffusion"]
    loaded = {m: load_model(m, args.noise, args.ckpt_dir, device) for m in models_list}
    diff_sched = GaussianDiffusion(500, device=device)

    out_dir = Path(args.save_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick a few representative samples
    indices = [0, 5, 10, 15, 20] if len(dataset) > 20 else list(range(min(5, len(dataset))))

    for idx in indices:
        noisy, clean, fname = dataset[idx]
        clean_np = clean.squeeze().numpy()
        noisy_np = noisy.squeeze().numpy()

        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
        titles = ["Noisy", "DnCNN", "GAN", "CycleGAN", "Diffusion", "Clean (GT)"]
        images = [noisy_np]
        for m in models_list:
            dn = denoise_one(loaded[m], noisy, m, device, diff_sched)
            images.append(np.clip(dn, 0, 1))
        images.append(clean_np)

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f"Denoising Comparison - {args.noise} noise (sample {idx})", fontsize=16)
        plt.tight_layout()
        plt.savefig(out_dir / f"comparison_{args.noise}_{idx:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison_{args.noise}_{idx:04d}.png")


def make_edge_profile_plot(args):
    """Plot edge profiles from each model vs ground truth."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SEMDataset(args.data_dir, args.noise, "test")
    with open(Path(args.data_dir) / "test" / "ground_truth.json") as f:
        all_gt = json.load(f)

    models_list = ["dncnn", "gan", "cyclegan", "diffusion"]
    loaded = {m: load_model(m, args.noise, args.ckpt_dir, device) for m in models_list}
    diff_sched = GaussianDiffusion(500, device=device)
    colors = {"dncnn": "#1f77b4", "gan": "#ff7f0e", "cyclegan": "#2ca02c", "diffusion": "#d62728"}

    out_dir = Path(args.save_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    noisy, clean, fname = dataset[0]
    gt = all_gt[0]
    if len(gt["lines"]) == 0:
        print("No lines in GT, skipping edge plot")
        return

    line_gt = gt["lines"][0]
    left_nom = np.mean(line_gt["left_edge"])
    gt_left = np.array(line_gt["left_edge"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    rows = np.arange(len(gt_left))
    ax.plot(rows, gt_left, 'k-', linewidth=2, label='Ground Truth', alpha=0.7)

    for m in models_list:
        dn = denoise_one(loaded[m], noisy, m, device, diff_sched)
        dn_img = np.clip(dn * 255, 0, 255)
        profile = measure_edge_profile(dn_img, left_nom)
        ax.plot(rows, profile, color=colors[m], linewidth=1.2, label=m, alpha=0.8)

    ax.set_xlabel("Row (pixel)", fontsize=12)
    ax.set_ylabel("Edge Position (pixel)", fontsize=12)
    ax.set_title(f"Left Edge Profile Comparison - {args.noise} noise", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"edge_profile_{args.noise}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved edge_profile_{args.noise}.png")


def make_metric_charts(args):
    """Create PSNR vs CD error scatter plot + bar charts."""
    out_dir = Path(args.save_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for noise in ["low", "medium", "high"]:
        rpath = Path(args.save_dir) / f"results_{noise}.json"
        if rpath.exists():
            with open(rpath) as f:
                all_data[noise] = json.load(f)["summary"]

    if not all_data:
        print("No results found")
        return

    models = ["dncnn", "gan", "cyclegan", "diffusion"]
    colors = {"dncnn": "#1f77b4", "gan": "#ff7f0e", "cyclegan": "#2ca02c", "diffusion": "#d62728"}
    markers = {"low": "o", "medium": "s", "high": "^"}

    # 1. PSNR vs CD error scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for noise, data in all_data.items():
        for m in models:
            if m in data:
                ax.scatter(data[m]["psnr"], data[m]["cd_error"],
                          color=colors[m], marker=markers[noise], s=120,
                          label=f"{m} ({noise})", edgecolors='black', linewidth=0.5)
    ax.set_xlabel("PSNR (dB) - higher is better", fontsize=12)
    ax.set_ylabel("CD Error (px) - lower is better", fontsize=12)
    ax.set_title("Image Quality vs Metrology Accuracy", fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "psnr_vs_cd_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved psnr_vs_cd_scatter.png")

    # 2. Bar chart per noise level
    for noise, data in all_data.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        vals_psnr = [data[m]["psnr"] for m in models]
        vals_cd = [data[m]["cd_error"] for m in models]
        c = [colors[m] for m in models]

        ax1.bar(models, vals_psnr, color=c, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel("PSNR (dB)", fontsize=12)
        ax1.set_title(f"Image Quality ({noise} noise)", fontsize=13)

        ax2.bar(models, vals_cd, color=c, edgecolor='black', linewidth=0.5)
        ax2.set_ylabel("CD Error (px)", fontsize=12)
        ax2.set_title(f"Metrology Accuracy ({noise} noise)", fontsize=13)

        plt.suptitle(f"Noise Level: {noise}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_dir / f"bars_{noise}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved bars_{noise}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic_sem_dataset")
    parser.add_argument("--noise", default="medium")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--save-dir", default="results")
    args = parser.parse_args()

    print("1. Generating comparison images...")
    make_comparison_grid(args)
    print("2. Generating edge profile plots...")
    make_edge_profile_plot(args)
    print("3. Generating metric charts...")
    make_metric_charts(args)
    print("Done! Check results/figures/")
