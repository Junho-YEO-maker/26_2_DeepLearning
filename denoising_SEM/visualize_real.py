"""
Visualize real SEM denoising results.
Generates comparison grids, metric charts, and PSNR-vs-CD scatter plots.

Usage:
  python visualize_real.py
"""

import os
import json
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

from models.dncnn import DnCNN
from models.gan import Generator
from models.cyclegan import CycleGANGenerator
from models.diffusion import DiffusionUNet, GaussianDiffusion


def load_model(name, noise, ckpt_dir="checkpoints_real", device="cuda"):
    """Load trained model."""
    d = os.path.join(ckpt_dir, f"{name}_{noise}")
    if name == 'dncnn':
        m = DnCNN(1, 8, 64).to(device)
        m.load_state_dict(torch.load(os.path.join(d, 'best.pth'), map_location=device, weights_only=True))
    elif name == 'gan':
        m = Generator(1, 1, 64).to(device)
        m.load_state_dict(torch.load(os.path.join(d, 'G_last.pth'), map_location=device, weights_only=True))
    elif name == 'cyclegan':
        m = CycleGANGenerator(1, 1, 64, 6).to(device)
        m.load_state_dict(torch.load(os.path.join(d, 'G_N2C_last.pth'), map_location=device, weights_only=True))
    elif name == 'diffusion':
        m = DiffusionUNet(1, 1, 64).to(device)
        m.load_state_dict(torch.load(os.path.join(d, 'model_last.pth'), map_location=device, weights_only=True))
    m.eval()
    return m


def denoise_one(model, noisy_np, name, device):
    """Denoise a single image."""
    with torch.no_grad():
        x = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).float().to(device)
        if name == 'diffusion':
            diff = GaussianDiffusion(500, device=device)
            out = diff.sample(model, x, 1).squeeze().cpu().numpy()
        else:
            out = model(x).squeeze().cpu().numpy()
    return np.clip(out, 0, 1)


def make_comparison_grid(data_dir, ckpt_dir, save_dir, device):
    """Generate side-by-side comparison grids for each noise level."""
    models = ['dncnn', 'gan', 'cyclegan', 'diffusion']
    noises = ['low', 'medium', 'high']
    n_examples = 4  # number of example images per grid

    for noise in noises:
        test_noisy = sorted(glob.glob(os.path.join(data_dir, noise, 'test', 'noisy', '*.png')))
        test_clean = sorted(glob.glob(os.path.join(data_dir, noise, 'test', 'clean', '*.png')))

        if len(test_noisy) == 0:
            continue

        # Pick evenly spaced examples
        indices = np.linspace(0, len(test_noisy) - 1, n_examples, dtype=int)

        fig, axes = plt.subplots(n_examples, 6, figsize=(18, 3 * n_examples))
        col_titles = ['Noisy', 'DnCNN', 'GAN', 'CycleGAN', 'Diffusion', 'Clean (ref)']

        # Load models
        loaded = {}
        for m_name in models:
            try:
                loaded[m_name] = load_model(m_name, noise, ckpt_dir, device)
            except Exception as e:
                print(f"  [SKIP] {m_name}: {e}")

        for row, idx in enumerate(indices):
            noisy_np = np.array(Image.open(test_noisy[idx]), dtype=np.float32) / 255.0
            clean_np = np.array(Image.open(test_clean[idx]), dtype=np.float32) / 255.0

            axes[row, 0].imshow(noisy_np, cmap='gray', vmin=0, vmax=1)
            axes[row, 5].imshow(clean_np, cmap='gray', vmin=0, vmax=1)

            for col, m_name in enumerate(models, 1):
                if m_name in loaded:
                    denoised = denoise_one(loaded[m_name], noisy_np, m_name, device)
                    axes[row, col].imshow(denoised, cmap='gray', vmin=0, vmax=1)
                else:
                    axes[row, col].text(0.5, 0.5, 'N/A', ha='center', va='center')

            for col in range(6):
                axes[row, col].axis('off')
                if row == 0:
                    axes[row, col].set_title(col_titles[col], fontsize=12, fontweight='bold')

        plt.suptitle(f'Real SEM Denoising — {noise.upper()} noise', fontsize=16, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(save_dir, f'comparison_real_{noise}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def make_metric_bar_charts(results_dir, save_dir):
    """Generate bar charts comparing PSNR, SSIM, CD error across models."""
    models = ['dncnn', 'gan', 'cyclegan', 'diffusion']
    noises = ['low', 'medium', 'high']
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']

    all_data = {}
    for noise in noises:
        fpath = os.path.join(results_dir, f'results_real_{noise}.json')
        if os.path.exists(fpath):
            with open(fpath) as f:
                all_data[noise] = json.load(f)

    if not all_data:
        print("  No results found!")
        return

    # --- PSNR bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, noise in enumerate(noises):
        if noise not in all_data:
            continue
        data = all_data[noise]
        vals = [data.get(m, {}).get('psnr_mean', 0) for m in models]
        stds = [data.get(m, {}).get('psnr_std', 0) for m in models]
        bars = axes[i].bar(models, vals, color=colors, yerr=stds, capsize=5)
        axes[i].set_title(f'{noise.upper()} noise', fontweight='bold')
        axes[i].set_ylabel('PSNR (dB)')
        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    plt.suptitle('PSNR Comparison — Real SEM Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'psnr_comparison_real.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- CD error bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, noise in enumerate(noises):
        if noise not in all_data:
            continue
        data = all_data[noise]
        vals = [data.get(m, {}).get('cd_error_mean', 0) for m in models]
        stds = [data.get(m, {}).get('cd_error_std', 0) for m in models]
        bars = axes[i].bar(models, vals, color=colors, yerr=stds, capsize=5)
        axes[i].set_title(f'{noise.upper()} noise', fontweight='bold')
        axes[i].set_ylabel('CD Error (%)')
        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.suptitle('CD Error Comparison — Real SEM Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'cd_error_comparison_real.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def make_psnr_vs_cd_scatter(results_dir, save_dir):
    """PSNR vs CD error scatter — the key plot showing ranking mismatch."""
    models = ['dncnn', 'gan', 'cyclegan', 'diffusion']
    noises = ['low', 'medium', 'high']
    markers = {'dncnn': 'o', 'gan': 's', 'cyclegan': '^', 'diffusion': 'D'}
    noise_colors = {'low': '#4CAF50', 'medium': '#FF9800', 'high': '#F44336'}

    fig, ax = plt.subplots(figsize=(8, 6))

    for noise in noises:
        fpath = os.path.join(results_dir, f'results_real_{noise}.json')
        if not os.path.exists(fpath):
            continue
        with open(fpath) as f:
            data = json.load(f)

        for m in models:
            if m not in data or 'cd_error_mean' not in data[m]:
                continue
            psnr = data[m]['psnr_mean']
            cd_err = data[m]['cd_error_mean']
            ax.scatter(psnr, cd_err, marker=markers[m], c=noise_colors[noise],
                      s=120, edgecolors='black', linewidth=0.5, zorder=5)
            ax.annotate(f'{m}\n({noise})', (psnr, cd_err),
                       textcoords="offset points", xytext=(8, 5), fontsize=7)

    # Legend
    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], marker=markers[m], color='gray', linestyle='',
                           markersize=8, label=m.upper()) for m in models]
    noise_handles = [Line2D([0], [0], marker='o', color=noise_colors[n], linestyle='',
                           markersize=8, label=f'{n} noise') for n in noises]
    ax.legend(handles=model_handles + noise_handles, loc='upper right', fontsize=8)

    ax.set_xlabel('PSNR (dB) — higher = better image quality', fontsize=11)
    ax.set_ylabel('CD Error (%) — lower = better metrology', fontsize=11)
    ax.set_title('PSNR vs CD Error — Real SEM Data\n"Looking good ≠ Measuring right"',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, 'psnr_vs_cd_scatter_real.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/nffa_sem')
    parser.add_argument('--ckpt-dir', default='checkpoints_real')
    parser.add_argument('--results-dir', default='results_real')
    parser.add_argument('--save-dir', default='results_real/figures')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 50)
    print("Generating visualizations (Real SEM data)")
    print("=" * 50)

    print("\n[1/3] Comparison grids...")
    make_comparison_grid(args.data_dir, args.ckpt_dir, args.save_dir, device)

    print("\n[2/3] Metric bar charts...")
    make_metric_bar_charts(args.results_dir, args.save_dir)

    print("\n[3/3] PSNR vs CD scatter...")
    make_psnr_vs_cd_scatter(args.results_dir, args.save_dir)

    print("\n" + "=" * 50)
    print("FIGURES DONE (Real SEM)")
    print("=" * 50)


if __name__ == '__main__':
    main()
