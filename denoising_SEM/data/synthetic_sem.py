"""
Synthetic SEM image generator for metrology evaluation.
Generates line patterns with controllable CD, LER, and SEM noise.
Ground truth dimensions are known exactly.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import json


def generate_rough_edge(length, roughness_rms, correlation_length=20.0, seed=None):
    """Generate rough edge profile using PSD method."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.fftfreq(length)
    psd = np.exp(-2 * (np.pi * correlation_length * freqs) ** 2)
    psd[0] = 0
    phases = rng.uniform(0, 2 * np.pi, length)
    spectrum = np.sqrt(psd) * np.exp(1j * phases)
    profile = np.fft.ifft(spectrum).real
    if profile.std() > 0:
        profile = profile / profile.std() * roughness_rms
    return profile


def render_line_pattern(img_size=256, num_lines=5, cd_pixels=20.0,
                        pitch_pixels=50.0, ler_rms=1.5, correlation_length=20.0,
                        brightness_line=200, brightness_space=50, edge_width=2.0, seed=None):
    """Render synthetic SEM line pattern with known ground truth."""
    rng = np.random.default_rng(seed)
    image = np.full((img_size, img_size), brightness_space, dtype=np.float64)
    gt_params = {"lines": [], "nominal_cd": cd_pixels, "nominal_ler_rms": ler_rms}
    x = np.arange(img_size)
    start_offset = (img_size - (num_lines - 1) * pitch_pixels) / 2

    for i in range(num_lines):
        center = start_offset + i * pitch_pixels
        s1, s2 = rng.integers(0, 2**31), rng.integers(0, 2**31)
        left_roughness = generate_rough_edge(img_size, ler_rms / 2, correlation_length, seed=s1)
        right_roughness = generate_rough_edge(img_size, ler_rms / 2, correlation_length, seed=s2)
        left_edge = center - cd_pixels / 2 + left_roughness
        right_edge = center + cd_pixels / 2 + right_roughness
        actual_cds = right_edge - left_edge

        gt_params["lines"].append({
            "center": float(center), "cd_mean": float(np.mean(actual_cds)),
            "cd_std": float(np.std(actual_cds)),
            "ler_left_rms": float(np.std(left_roughness) * 2),
            "ler_right_rms": float(np.std(right_roughness) * 2),
            "left_edge": left_edge.tolist(), "right_edge": right_edge.tolist(),
        })

        for row in range(img_size):
            le, re = left_edge[row], right_edge[row]
            left_trans = 1.0 / (1.0 + np.exp(-(x - le) / (edge_width / 4)))
            right_trans = 1.0 / (1.0 + np.exp((x - re) / (edge_width / 4)))
            line_profile = left_trans * right_trans
            edge_boost = 40 * (np.exp(-((x - le)**2) / (2*edge_width**2)) +
                               np.exp(-((x - re)**2) / (2*edge_width**2)))
            image[row] += line_profile * (brightness_line - brightness_space) + edge_boost

    return np.clip(image, 0, 255), gt_params


def add_sem_noise(clean_image, noise_level="medium", seed=None):
    """Add SEM noise: Poisson + Gaussian + scan lines."""
    rng = np.random.default_rng(seed)
    cfgs = {
        "low":    {"ps": 0.8, "gs": 8,  "ss": 2},
        "medium": {"ps": 0.5, "gs": 15, "ss": 4},
        "high":   {"ps": 0.3, "gs": 25, "ss": 6},
    }
    c = cfgs[noise_level] if isinstance(noise_level, str) else {"ps": 0.5, "gs": float(noise_level), "ss": 3}
    img = clean_image.copy().astype(np.float64)
    scaled = np.maximum(img * c["ps"], 0)
    img = rng.poisson(scaled).astype(np.float64) / c["ps"]
    img += rng.normal(0, c["gs"], img.shape)
    img += rng.normal(0, c["ss"], img.shape[0])[:, np.newaxis]
    return np.clip(img, 0, 255), {"noise_level": noise_level}


def generate_dataset(output_dir, num_images=200, img_size=256,
                     noise_levels=("low", "medium", "high"), cd_range=(15, 25),
                     ler_range=(0.5, 2.5), seed=42):
    """Generate full synthetic SEM dataset with clean/noisy pairs."""
    out = Path(output_dir)
    rng = np.random.default_rng(seed)

    for split, n in [("train", int(num_images * 0.7)),
                     ("val", int(num_images * 0.15)),
                     ("test", int(num_images * 0.15))]:
        split_dir = out / split
        (split_dir / "clean").mkdir(parents=True, exist_ok=True)
        all_gt = []

        for i in range(n):
            cd = rng.uniform(*cd_range)
            ler = rng.uniform(*ler_range)
            num_lines = rng.integers(3, 8)
            pitch = cd * rng.uniform(2.0, 3.5)
            clean, gt = render_line_pattern(img_size=img_size, num_lines=int(num_lines),
                cd_pixels=cd, pitch_pixels=pitch, ler_rms=ler, seed=seed + i * 1000)
            fname = f"{i:04d}.png"
            Image.fromarray(clean.astype(np.uint8)).save(split_dir / "clean" / fname)
            gt["filename"] = fname
            all_gt.append(gt)
            for nl in noise_levels:
                nd = split_dir / f"noisy_{nl}"
                nd.mkdir(parents=True, exist_ok=True)
                noisy, _ = add_sem_noise(clean, nl, seed=seed + i * 1000 + hash(nl) % 10000)
                Image.fromarray(noisy.astype(np.uint8)).save(nd / fname)

        with open(split_dir / "ground_truth.json", "w") as f:
            json.dump(all_gt, f, indent=2)
        print(f"  {split}: {n} images x {len(noise_levels)} noise levels")
    print(f"Dataset saved to {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_sem_dataset")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2021147017)
    args = parser.parse_args()
    print("Generating synthetic SEM dataset...")
    generate_dataset(args.output, args.num_images, args.img_size, seed=args.seed)
