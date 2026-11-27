"""
Evaluation: image quality (PSNR/SSIM) vs metrology accuracy (CD/LER error).
Core hypothesis: image quality ranking != metrology accuracy ranking.
Usage: python evaluate.py --noise medium
"""

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from train import SEMDataset
from models.dncnn import DnCNN
from models.gan import Generator as GANGen
from models.cyclegan import CycleGANGenerator
from models.diffusion import DiffusionUNet, GaussianDiffusion
from metrics.image_quality import compute_psnr, compute_ssim
from metrics.metrology import MetrologyEvaluator


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
    else:
        print(f"WARNING: {path} not found")
    model.eval()
    return model


def denoise(model, noisy, name, device="cuda", diffusion=None, n_samples=1):
    with torch.no_grad():
        x = noisy.unsqueeze(0).to(device) if noisy.dim() == 3 else noisy.to(device)
        if name == "diffusion":
            if diffusion is None:
                diffusion = GaussianDiffusion(500, device=device)
            if n_samples > 1:
                samples = diffusion.sample(model, x, n_samples)
                return np.mean(samples.squeeze(2).cpu().numpy(), axis=0).squeeze(), float(np.mean(np.std(samples.cpu().numpy(), axis=0)))
            return diffusion.sample(model, x, 1).squeeze().cpu().numpy(), None
        return model(x).squeeze().cpu().numpy(), None


def evaluate_all(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(Path(args.data_dir) / "test" / "ground_truth.json") as f:
        all_gt = json.load(f)
    dataset = SEMDataset(args.data_dir, args.noise, "test")
    models_list = ["dncnn", "gan", "cyclegan", "diffusion"]
    results = {m: [] for m in models_list}
    loaded = {m: load_model(m, args.noise, args.ckpt_dir, device) for m in models_list}
    diff_sched = GaussianDiffusion(500, device=device)

    for idx in tqdm(range(len(dataset))):
        noisy, clean, fname = dataset[idx]
        clean_np = (clean.squeeze().numpy() * 255).astype(np.float64)
        gt = all_gt[idx]
        evaluator = MetrologyEvaluator(gt)

        for m in models_list:
            dn, unc = denoise(loaded[m], noisy, m, device, diff_sched,
                              args.diffusion_samples if m == "diffusion" else 1)
            dn = np.clip(dn * 255, 0, 255)
            psnr = compute_psnr(clean_np, dn, 255.0)
            ssim = compute_ssim(clean_np, dn, 255.0)
            metro = evaluator.evaluate(dn)
            r = {"fname": fname, "psnr": psnr, "ssim": ssim,
                 "cd_error": metro["mean_cd_error"], "ler_error": metro["mean_ler_error"]}
            if unc is not None:
                r["uncertainty"] = unc
            results[m].append(r)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"{'Model':<12} {'PSNR':>7} {'SSIM':>7} {'CD err':>9} {'LER err':>9}")
    print("=" * 65)
    summary = {}
    for m in models_list:
        s = {k: np.mean([r[k] for r in results[m]]) for k in ["psnr","ssim","cd_error","ler_error"]}
        summary[m] = s
        print(f"{m:<12} {s['psnr']:>7.2f} {s['ssim']:>7.4f} {s['cd_error']:>9.4f} {s['ler_error']:>9.4f}")

    psnr_rank = sorted(models_list, key=lambda m: summary[m]["psnr"], reverse=True)
    cd_rank = sorted(models_list, key=lambda m: summary[m]["cd_error"])
    print("\n  PSNR ranking: " + " > ".join(psnr_rank))
    print("  CD accuracy:  " + " > ".join(cd_rank))
    if psnr_rank != cd_rank:
        print("\n  ** PSNR ranking != CD accuracy ranking **")
        print("  ** 'Looking good' != 'Measuring right' **")

    with open(save_dir / f"results_{args.noise}.json", "w") as f:
        json.dump({"summary": summary, "details": {m: results[m] for m in models_list}}, f, indent=2, default=str)
    print(f"\nSaved to {save_dir}/results_{args.noise}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/synthetic_sem_dataset")
    parser.add_argument("--noise", default="medium")
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--save-dir", default="results")
    parser.add_argument("--diffusion-samples", type=int, default=5)
    args = parser.parse_args()
    evaluate_all(args)
