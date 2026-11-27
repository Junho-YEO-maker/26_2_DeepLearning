"""
Evaluate all 4 models on real SEM data.
Computes PSNR/SSIM and CD/LER metrology metrics, same as synthetic evaluation.

Usage:
  python evaluate_real.py
  python evaluate_real.py --noise medium
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
from PIL import Image

from models.dncnn import DnCNN
from models.gan import UNetGenerator
from models.cyclegan import ResNetGenerator
from models.diffusion import DiffusionUNet, GaussianDiffusion
from metrics.image_quality import compute_psnr, compute_ssim
from metrics.metrology import MetrologyEvaluator


def load_model(model_name, checkpoint_dir, noise_level, device):
    """Load a trained model from checkpoint."""
    ckpt_dir = os.path.join(checkpoint_dir, f"{model_name}_{noise_level}")

    if model_name == 'dncnn':
        model = DnCNN().to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'dncnn.pth'), map_location=device))
    elif model_name == 'gan':
        model = UNetGenerator().to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'gan.pth'), map_location=device))
    elif model_name == 'cyclegan':
        model = ResNetGenerator().to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'generator_n2c.pth'), map_location=device))
    elif model_name == 'diffusion':
        unet = DiffusionUNet().to(device)
        model = GaussianDiffusion(unet, T=500).to(device)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'diffusion.pth'), map_location=device))

    model.eval()
    return model


def denoise_image(model, model_name, noisy_tensor, device):
    """Run denoising on a single image."""
    with torch.no_grad():
        noisy = noisy_tensor.unsqueeze(0).to(device)

        if model_name == 'diffusion':
            output = model.sample(noisy, n_samples=1)
        else:
            output = model(noisy)

        return output.squeeze(0).squeeze(0).cpu().numpy()


def evaluate_all(data_dir, checkpoint_dir, noise_levels, results_dir, device):
    """Evaluate all models on all noise levels."""

    model_names = ['dncnn', 'gan', 'cyclegan', 'diffusion']
    metrology = MetrologyEvaluator()

    all_results = {}

    for noise_level in noise_levels:
        print(f"\n{'='*60}")
        print(f"Evaluating on {noise_level} noise (REAL data)")
        print(f"{'='*60}")

        # Load test images
        test_noisy_dir = os.path.join(data_dir, noise_level, 'test', 'noisy')
        test_clean_dir = os.path.join(data_dir, noise_level, 'test', 'clean')

        noisy_files = sorted(glob.glob(os.path.join(test_noisy_dir, '*.png')))
        clean_files = sorted(glob.glob(os.path.join(test_clean_dir, '*.png')))

        print(f"  Test images: {len(noisy_files)}")

        level_results = {}

        for model_name in model_names:
            print(f"\n  --- {model_name.upper()} ---")

            try:
                model = load_model(model_name, checkpoint_dir, noise_level, device)
            except Exception as e:
                print(f"  [SKIP] Could not load {model_name}: {e}")
                continue

            psnr_list, ssim_list = [], []
            cd_errors, ler_errors = [], []

            for i, (nf, cf) in enumerate(zip(noisy_files, clean_files)):
                # Load images
                noisy_np = np.array(Image.open(nf), dtype=np.float32) / 255.0
                clean_np = np.array(Image.open(cf), dtype=np.float32) / 255.0

                noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0).float()

                # Denoise
                denoised = denoise_image(model, model_name, noisy_tensor, device)
                denoised = np.clip(denoised, 0, 1)

                # Image quality metrics
                psnr_val = compute_psnr(clean_np, denoised)
                ssim_val = compute_ssim(clean_np, denoised)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

                # Metrology metrics (CD and LER)
                try:
                    cd_clean = metrology.measure_cd(clean_np)
                    cd_denoised = metrology.measure_cd(denoised)
                    ler_clean = metrology.measure_ler(clean_np)
                    ler_denoised = metrology.measure_ler(denoised)

                    if cd_clean > 0 and cd_denoised > 0:
                        cd_errors.append(abs(cd_denoised - cd_clean) / cd_clean * 100)
                    if ler_clean > 0 and ler_denoised > 0:
                        ler_errors.append(abs(ler_denoised - ler_clean) / ler_clean * 100)
                except Exception:
                    pass  # Some images may not have clear line patterns for metrology

            # Aggregate
            result = {
                'psnr_mean': float(np.mean(psnr_list)),
                'psnr_std': float(np.std(psnr_list)),
                'ssim_mean': float(np.mean(ssim_list)),
                'ssim_std': float(np.std(ssim_list)),
                'n_images': len(psnr_list),
            }

            if cd_errors:
                result['cd_error_mean'] = float(np.mean(cd_errors))
                result['cd_error_std'] = float(np.std(cd_errors))
                result['n_cd_measured'] = len(cd_errors)

            if ler_errors:
                result['ler_error_mean'] = float(np.mean(ler_errors))
                result['ler_error_std'] = float(np.std(ler_errors))
                result['n_ler_measured'] = len(ler_errors)

            level_results[model_name] = result

            print(f"    PSNR: {result['psnr_mean']:.2f} ± {result['psnr_std']:.2f}")
            print(f"    SSIM: {result['ssim_mean']:.4f} ± {result['ssim_std']:.4f}")
            if cd_errors:
                print(f"    CD error: {result['cd_error_mean']:.2f}% ± {result['cd_error_std']:.2f}% ({len(cd_errors)} measured)")
            if ler_errors:
                print(f"    LER error: {result['ler_error_mean']:.2f}% ± {result['ler_error_std']:.2f}% ({len(ler_errors)} measured)")

        all_results[noise_level] = level_results

        # Save per-level results
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, f'results_real_{noise_level}.json'), 'w') as f:
            json.dump(level_results, f, indent=2)
        print(f"\n  Results saved: {results_dir}/results_real_{noise_level}.json")

    # Print ranking comparison
    print(f"\n{'='*60}")
    print("RANKING COMPARISON (REAL DATA)")
    print(f"{'='*60}")

    for noise_level in noise_levels:
        if noise_level not in all_results:
            continue
        lr = all_results[noise_level]

        print(f"\n--- {noise_level.upper()} noise ---")

        # PSNR ranking
        psnr_ranking = sorted(lr.items(), key=lambda x: x[1]['psnr_mean'], reverse=True)
        print(f"  PSNR ranking:  {' > '.join([f'{m}({v[\"psnr_mean\"]:.1f})' for m, v in psnr_ranking])}")

        # SSIM ranking
        ssim_ranking = sorted(lr.items(), key=lambda x: x[1]['ssim_mean'], reverse=True)
        print(f"  SSIM ranking:  {' > '.join([f'{m}({v[\"ssim_mean\"]:.3f})' for m, v in ssim_ranking])}")

        # CD ranking (lower error = better)
        cd_items = [(m, v) for m, v in lr.items() if 'cd_error_mean' in v]
        if cd_items:
            cd_ranking = sorted(cd_items, key=lambda x: x[1]['cd_error_mean'])
            print(f"  CD ranking:    {' > '.join([f'{m}({v[\"cd_error_mean\"]:.1f}%)' for m, v in cd_ranking])}")

        # Check if rankings differ
        if cd_items and len(psnr_ranking) >= 2 and len(cd_ranking) >= 2:
            psnr_best = psnr_ranking[0][0]
            cd_best = cd_ranking[0][0]
            if psnr_best != cd_best:
                print(f"  *** RANKING MISMATCH: PSNR best={psnr_best}, CD best={cd_best} ***")

    print(f"\n{'='*60}")
    print("EVALUATION DONE (REAL DATA)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate on real SEM data')
    parser.add_argument('--data-dir', type=str, default='data/nffa_sem')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_real')
    parser.add_argument('--results-dir', type=str, default='results_real')
    parser.add_argument('--noise', type=str, default='all',
                        choices=['low', 'medium', 'high', 'all'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    noise_levels = ['low', 'medium', 'high'] if args.noise == 'all' else [args.noise]

    evaluate_all(args.data_dir, args.checkpoint_dir, noise_levels, args.results_dir, device)


if __name__ == '__main__':
    main()
